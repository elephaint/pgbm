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

#include <torch/script.h>
#include <vector>

// CPU function
std::vector<torch::Tensor> split_gain_cpu(
    torch::Tensor X,
    torch::Tensor grad_hess,
    int64_t n_bins) {
  
  auto device = X.device();
  auto range = torch::arange(n_bins, torch::device(device));
  auto left_idx = torch::le(X.unsqueeze(-1), range).to(torch::dtype(torch::kF32));
  auto Glc = left_idx.sum(1);
  auto GlHl = torch::einsum("ij, kil -> jkl", {grad_hess, left_idx});
  
  return {GlHl[0], GlHl[1], Glc};
}

// Bind to Torch Library
TORCH_LIBRARY(pgbm, m) {
  m.def("split_gain", &split_gain_cpu);
}