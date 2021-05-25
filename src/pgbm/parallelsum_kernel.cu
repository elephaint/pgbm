// 
//   Copyright (c) 2021 Olivier Sprangers as part of Airlab Amsterdam
//
//   Licensed under the Apache License, Version 2.0 (the "License");
//   you may not use this file except in compliance with the License.
 //  You may obtain a copy of the License at
//
//       http://www.apache.org/licenses/LICENSE-2.0
//
//   Unless required by applicable law or agreed to in writing, software
//   distributed under the License is distributed on an "AS IS" BASIS,
//   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//   See the License for the specific language governing permissions and
//   limitations under the License.
//
//   https://github.com/elephaint/pgbm/blob/main/LICENSE

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

__global__ void _kernel_psum(
  const torch::PackedTensorAccessor32<unsigned char,2,torch::RestrictPtrTraits> X, 
  const torch::PackedTensorAccessor32<float,1,torch::RestrictPtrTraits> gradient,
  const torch::PackedTensorAccessor32<float,1,torch::RestrictPtrTraits> hessian,
  torch::PackedTensorAccessor32<at::BFloat16,3,torch::RestrictPtrTraits> Gl_block,
  torch::PackedTensorAccessor32<at::BFloat16,3,torch::RestrictPtrTraits> Hl_block) {
  
  // Sample index
  unsigned int ti_x = threadIdx.x;
  unsigned int blockSize_x = blockDim.x;
  unsigned int gridSize_x = blockSize_x * gridDim.x;
  unsigned int i = blockIdx.x * blockSize_x + ti_x; 
  // Feature index
  unsigned int j = blockIdx.y * blockDim.y + threadIdx.y; 
  // Bin index
  unsigned int k = blockIdx.z * blockDim.z + threadIdx.z;
  // Thread block identifier
  // https://developer.nvidia.com/blog/cooperative-groups/
  cg::thread_block cta = cg::this_thread_block(); 
  cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(cta);
  
  // Temporary floats
  float Gl_temp = 0;
  float Hl_temp = 0;

  // Load values and perform first reduction
  // Formulation of multiplication instead of if-condition improves speed as no divergent branches are created.
  while (i < X.size(1) && j < X.size(0) && k < Gl_block.size(2)){
    float flag_add = k < X[j][i];
    Gl_temp += gradient[i] * flag_add;
    Hl_temp += hessian[i] * flag_add;
    i += gridSize_x;  
  }

  // Reduce within warp using shuffle: https://developer.nvidia.com/blog/faster-parallel-reductions-kepler/
  for (int i = 16; i > 0; i /= 2) {
    Gl_temp += tile32.shfl_down(Gl_temp, i);
    Hl_temp += tile32.shfl_down(Hl_temp, i);
  }

  // First thread contains blocksum
  if (ti_x == 0){
    Gl_block[blockIdx.x][j][k] = Gl_temp;
    Hl_block[blockIdx.x][j][k] = Hl_temp;
  }
}

__global__ void _kernel_Atomic(
  const torch::PackedTensorAccessor32<int,2,torch::RestrictPtrTraits> X,
  const torch::PackedTensorAccessor32<float,1,torch::RestrictPtrTraits> gradient,
  const torch::PackedTensorAccessor32<float,1,torch::RestrictPtrTraits> hessian,
  torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> Gl,
  torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> Hl) {
  // feature index
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  // sample index
  unsigned int ti_j = threadIdx.y;
  unsigned int j = blockIdx.y * blockDim.y + ti_j;

  if (i < X.size(0) && j < X.size(1)){
  // Create shared gradient and hessian variables
  __shared__ int Xs[1024];
  __shared__ float gs[1024];
  __shared__ float hs[1024];
  // Fill gradient and hessian variables
  Xs[ti_j] = X[i][j];
  gs[ti_j] = gradient[j];
  hs[ti_j] = hessian[j];
  // Sync threads 
  __syncthreads();    
  // Loop over bins
  auto Xsc = Xs[ti_j];
  auto grad = gs[ti_j];
  auto hess = hs[ti_j];
  for (int k = 0; k < Xsc; k++){
    atomicAdd(&Gl[i][k], grad);
    atomicAdd(&Hl[i][k], hess);
  }
  // Sync threads 
  __syncthreads(); 
  }
}

std::vector<torch::Tensor> _split_decision_cuda(
    torch::Tensor X,
    torch::Tensor gradient,
    torch::Tensor hessian,
    int n_bins) {

  // Define number of samples and number of features
  // NB: Typically n_samples >> n_features, such that dimension 1 of X should be the sample dimension, to ensure optimal accesss speed in that
  // direction (as tensors are stored in row-major order)
  int n_samples = X.size(1);
  int n_features = X.size(0);
  auto Gl = torch::zeros({n_features, n_bins}, torch::dtype(torch::kF32).device(X.device()));
  auto Hl = torch::zeros({n_features, n_bins}, torch::dtype(torch::kF32).device(X.device()));
  
  // Invoke kernel
  if ( n_bins <= 32 || n_bins == 64 || n_bins == 128 || n_bins == 256){
    // Number of threadblocks. As our kernel uses only warpshuffle reductions, we can't exceed 32 threads in the n_samples (x) dimension.
    // The disadvantage is that this requires relatively large amount of GPU memory when n_samples >>, as bpg_x will become huge.
    // Note that samples are now the first dimension due to the maximum gridsize of CUDA (> 3.5) being (2^31 - 1, 65535, 65535). 
    // If samples would be the y-dimension, for a problem with e.g. 10.000.000 samples we could not have a sufficiently large grid size (because 10.000.000 / 32 > 65535)
    const dim3 threadsPerBlock(32, 1, std::min(32, n_bins));
    int bpg_x = (n_samples + threadsPerBlock.x - 1) / threadsPerBlock.x;
    int bpg_y = (n_features + threadsPerBlock.y - 1) / threadsPerBlock.y;
    int bpg_z = (n_bins + threadsPerBlock.z - 1) / threadsPerBlock.z;
    const dim3 numBlocks(bpg_x, bpg_y, bpg_z);
    // Temporary tensor array containing blocksums. Results in high GPU memory requirement when n_samples >>.
    // Use bfloat16 to reduce memory consumption.
    auto Gl_block = torch::zeros({bpg_x, n_features, n_bins}, torch::dtype(at::ScalarType::BFloat16).device(X.device()));
    auto Hl_block = torch::zeros({bpg_x, n_features, n_bins}, torch::dtype(at::ScalarType::BFloat16).device(X.device()));
    //  Run kernel
    _kernel_psum<<<numBlocks, threadsPerBlock>>>(
      X.packed_accessor32<unsigned char,2,torch::RestrictPtrTraits>(),
      gradient.packed_accessor32<float,1,torch::RestrictPtrTraits>(),
      hessian.packed_accessor32<float,1,torch::RestrictPtrTraits>(),
      Gl_block.packed_accessor32<at::BFloat16,3,torch::RestrictPtrTraits>(),
      Hl_block.packed_accessor32<at::BFloat16,3,torch::RestrictPtrTraits>());
    // Return Gl, Hl
    Gl = Gl_block.sum(0).to(torch::kF32);
    Hl = Hl_block.sum(0).to(torch::kF32);
    }
  else { 
    // Constraint: this solution works up to ~67M n_samples (but then you'll probably have a GPU OOM error anyways...)
    const dim3 threadsPerBlock(1, 1024);
    int bpg_x = (n_features + threadsPerBlock.x - 1) / threadsPerBlock.x;
    int bpg_y = (n_samples + threadsPerBlock.y - 1) / threadsPerBlock.y;
    const dim3 numBlocks(bpg_x, bpg_y);

    _kernel_Atomic<<<numBlocks, threadsPerBlock>>>(
      X.packed_accessor32<int,2,torch::RestrictPtrTraits>(),
      gradient.packed_accessor32<float,1,torch::RestrictPtrTraits>(),
      hessian.packed_accessor32<float,1,torch::RestrictPtrTraits>(),
      Gl.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
      Hl.packed_accessor32<float,2,torch::RestrictPtrTraits>());
    }
  // Return sum reduced over the block-dimension.
  return {Gl, Hl};
}
