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

// Cuda kernel to compute split gain given short int input X
__global__ void _kernel_Atomic_long(
  const torch::PackedTensorAccessor32<short int,2,torch::RestrictPtrTraits> X,
  const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> gradient_hessian,
  torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> GlHlGlc) {
  // feature index
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  // sample index
  unsigned int ti_j = threadIdx.y;
  unsigned int j = blockIdx.y * blockDim.y + ti_j;

  if (i < X.size(0) && j < X.size(1)){
  // Create shared gradient and hessian variables
  __shared__ short int Xs[1024];
  __shared__ float gs[1024];
  __shared__ float hs[1024];
  // Fill gradient and hessian variables
  Xs[ti_j] = X[i][j];
  gs[ti_j] = gradient_hessian[j][0];
  hs[ti_j] = gradient_hessian[j][1];
  // Sync threads 
  __syncthreads();    
  // Loop over bins
  auto Xsc = Xs[ti_j];
  auto grad = gs[ti_j];
  auto hess = hs[ti_j];
  atomicAdd(&GlHlGlc[0][i][Xsc], grad);
  atomicAdd(&GlHlGlc[1][i][Xsc], hess);
  atomicAdd(&GlHlGlc[2][i][Xsc], 1);
  // Sync threads 
  __syncthreads(); 
  }
}

// Cuda kernel to compute split gain given uint8 input X
__global__ void _kernel_Atomic_short(
  const torch::PackedTensorAccessor32<unsigned char,2,torch::RestrictPtrTraits> X,
  const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> gradient_hessian,
  torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> GlHlGlc) {
  // feature index
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  // sample index
  unsigned int ti_j = threadIdx.y;
  unsigned int j = blockIdx.y * blockDim.y + ti_j;

  if (i < X.size(0) && j < X.size(1)){
  // Create shared gradient and hessian variables
  __shared__ unsigned char Xs[1024];
  __shared__ float gs[1024];
  __shared__ float hs[1024];
  // Fill gradient and hessian variables
  Xs[ti_j] = X[i][j];
  gs[ti_j] = gradient_hessian[j][0];
  hs[ti_j] = gradient_hessian[j][1];
  // Sync threads 
  __syncthreads();    
  // Loop over bins
  auto Xsc = Xs[ti_j];
  auto grad = gs[ti_j];
  auto hess = hs[ti_j];
  atomicAdd(&GlHlGlc[0][i][Xsc], grad);
  atomicAdd(&GlHlGlc[1][i][Xsc], hess);
  atomicAdd(&GlHlGlc[2][i][Xsc], 1);
  // Sync threads 
  __syncthreads(); 
  }
}

// C++ function to call kernels
std::vector<torch::Tensor> _split_gain_cuda(
    torch::Tensor X,
    torch::Tensor gradient_hessian,
    int64_t n_bins) {

  // Define number of samples and number of features
  // NB: Typically n_samples >> n_features, such that dimension 1 of X should be the sample dimension, to ensure optimal accesss speed in that
  // direction (as tensors are stored in row-major order)
  int n_samples = X.size(1);
  int n_features = X.size(0);
  auto GlHlGlc = torch::zeros({3, n_features, n_bins}, torch::dtype(torch::kF32).device(X.device()));
  
  // Constraint: this solution works up to ~67M n_samples (but then you'll probably have a GPU OOM error anyways...)
  const dim3 threadsPerBlock(1, 1024);
  int bpg_x = (n_features + threadsPerBlock.x - 1) / threadsPerBlock.x;
  int bpg_y = (n_samples + threadsPerBlock.y - 1) / threadsPerBlock.y;
  const dim3 numBlocks(bpg_x, bpg_y);

  if (n_bins <= 256) {
    // Run kernel uint8
    _kernel_Atomic_short<<<numBlocks, threadsPerBlock>>>(
      X.packed_accessor32<unsigned char,2,torch::RestrictPtrTraits>(),
      gradient_hessian.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
      GlHlGlc.packed_accessor32<float,3,torch::RestrictPtrTraits>());
    } else {
    // Run kernel short int
    _kernel_Atomic_long<<<numBlocks, threadsPerBlock>>>(
      X.packed_accessor32<short int,2,torch::RestrictPtrTraits>(),
      gradient_hessian.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
      GlHlGlc.packed_accessor32<float,3,torch::RestrictPtrTraits>());
    }
  
  // Cumsum over n_bins dimension, inplace
  GlHlGlc.cumsum_(-1);
  return {GlHlGlc[0], GlHlGlc[1], GlHlGlc[2]};
}
