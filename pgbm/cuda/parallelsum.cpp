
#include <torch/extension.h>
// #include <iostream>
#include <vector>
#include <c10/cuda/CUDAGuard.h>
// Tutorial: https://pytorch.org/tutorials/advanced/cpp_extension.html

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
  auto Gl = torch::einsum("i, jik -> jk", {gradient, left_idx});
  auto Hl = torch::einsum("i, jik -> jk", {hessian, left_idx});

  return {Gl, Hl};
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