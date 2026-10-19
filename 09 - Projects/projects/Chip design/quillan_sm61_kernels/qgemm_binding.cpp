// qgemm_binding.cpp
//
// PyTorch extension entry point. Bridges torch::Tensor <-> raw CUDA pointers
// and calls into the sm_61 DP4A kernel launcher defined in sm61_qgemm.cu.

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <ATen/cuda/CUDAContext.h>

// Declared in sm61_qgemm.cu
extern "C" void qgemm_i8_dp4a_launch(
    const int8_t* X, const int8_t* W,
    const float* x_scale, const float* w_scale,
    void* Y_half_ptr, int M, int N, int K, cudaStream_t stream);

torch::Tensor qgemm_forward(
    torch::Tensor x_int8,   // [M, K] int8, contiguous, CUDA
    torch::Tensor w_int8,   // [N, K] int8, contiguous, CUDA (pre-transposed weight)
    torch::Tensor x_scale,  // [M]    float32, CUDA
    torch::Tensor w_scale)  // [N]    float32, CUDA
{
    TORCH_CHECK(x_int8.is_cuda() && w_int8.is_cuda(), "x and w must be CUDA tensors");
    TORCH_CHECK(x_scale.is_cuda() && w_scale.is_cuda(), "scales must be CUDA tensors");
    TORCH_CHECK(x_int8.scalar_type() == torch::kInt8, "x must be int8");
    TORCH_CHECK(w_int8.scalar_type() == torch::kInt8, "w must be int8");
    TORCH_CHECK(x_scale.scalar_type() == torch::kFloat32, "x_scale must be float32");
    TORCH_CHECK(w_scale.scalar_type() == torch::kFloat32, "w_scale must be float32");
    TORCH_CHECK(x_int8.is_contiguous() && w_int8.is_contiguous(), "x and w must be contiguous");
    TORCH_CHECK(x_int8.dim() == 2 && w_int8.dim() == 2, "x and w must be 2D");

    const int64_t M = x_int8.size(0);
    const int64_t K = x_int8.size(1);
    const int64_t N = w_int8.size(0);
    TORCH_CHECK(w_int8.size(1) == K, "K mismatch: x is [M,", K, "], w is [N,", w_int8.size(1), "]");
    TORCH_CHECK(x_scale.numel() == M, "x_scale must have M elements");
    TORCH_CHECK(w_scale.numel() == N, "w_scale must have N elements");

    auto y = torch::empty({M, N}, x_int8.options().dtype(torch::kFloat16));

    qgemm_i8_dp4a_launch(
        x_int8.data_ptr<int8_t>(),
        w_int8.data_ptr<int8_t>(),
        x_scale.data_ptr<float>(),
        w_scale.data_ptr<float>(),
        y.data_ptr(),
        static_cast<int>(M), static_cast<int>(N), static_cast<int>(K),
        at::cuda::getCurrentCUDAStream());

    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("qgemm_forward", &qgemm_forward,
          "Quillan sm_61 DP4A quantized INT8 GEMM -> fp16 (CUDA)");
}
