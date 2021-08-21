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
#include <vector>
#include <c10/cuda/CUDAGuard.h>

// CUDA function: split decision
std::vector<torch::Tensor> _split_decision_cuda(
    torch::Tensor X,
    torch::Tensor gradient,
    torch::Tensor hessian,
    int n_bins);

// Wrapper with CUDA guard
std::vector<torch::Tensor> split_decision_cuda(
    torch::Tensor X,
    torch::Tensor gradient,
    torch::Tensor hessian,
    int n_bins) {
    const at::cuda::OptionalCUDAGuard device_guard(device_of(X));
    return _split_decision_cuda(X, gradient, hessian, n_bins); 
    }

// CPU function
std::vector<torch::Tensor> split_decision_cpu(
    torch::Tensor X,
    torch::Tensor gradient,
    torch::Tensor hessian,
    int n_bins) {
  
  auto device = X.device();
  auto range = torch::arange(n_bins, torch::device(device));
  auto left_idx = torch::gt(X.unsqueeze(-1), range).to(torch::dtype(torch::kF32));
  auto Glc = left_idx.sum(1);
  auto Gl = torch::einsum("i, jik -> jk", {gradient, left_idx});
  auto Hl = torch::einsum("i, jik -> jk", {hessian, left_idx});

  return {Gl, Hl, Glc};
}

// Switch function. NB: should be able to define this with a TorchLibrary but that does not work :(
std::vector<torch::Tensor> split_decision(
    torch::Tensor X,
    torch::Tensor gradient,
    torch::Tensor hessian,
    int n_bins) {

if (X.is_cuda()) {return split_decision_cuda(X, gradient, hessian, n_bins);}
else {return split_decision_cpu(X, gradient, hessian, n_bins);}
}

// Bind to python
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("split_decision", &split_decision, "PGBM split decision");
}