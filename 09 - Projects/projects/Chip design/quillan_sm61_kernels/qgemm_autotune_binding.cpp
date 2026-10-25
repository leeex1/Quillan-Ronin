// qgemm_autotune_binding.cpp
//
// PyTorch binding for the autotuning dispatch kernels.
// Exports all kernel variants as Python-callable functions.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>

#include <stdexcept>

// Forward declarations from sm61_qgemm_autotune.cu
extern "C" void qgemm_launch_64_64_32_1_1(
    const int8_t* X, const int8_t* W,
    const float* x_scale, const float* w_scale,
    void* Y_half_ptr, int M, int N, int K, cudaStream_t stream);

extern "C" void qgemm_launch_128_64_32_1_1(
    const int8_t* X, const int8_t* W,
    const float* x_scale, const float* w_scale,
    void* Y_half_ptr, int M, int N, int K, cudaStream_t stream);

extern "C" void qgemm_launch_128_64_32_2_2(
    const int8_t* X, const int8_t* W,
    const float* x_scale, const float* w_scale,
    void* Y_half_ptr, int M, int N, int K, cudaStream_t stream);

extern "C" void qgemm_launch_256_64_32_1_1(
    const int8_t* X, const int8_t* W,
    const float* x_scale, const float* w_scale,
    void* Y_half_ptr, int M, int N, int K, cudaStream_t stream);

extern "C" void qgemm_launch_256_64_32_2_2(
    const int8_t* X, const int8_t* W,
    const float* x_scale, const float* w_scale,
    void* Y_half_ptr, int M, int N, int K, cudaStream_t stream);

extern "C" void qgemm_launch_128_128_32_1_1(
    const int8_t* X, const int8_t* W,
    const float* x_scale, const float* w_scale,
    void* Y_half_ptr, int M, int N, int K, cudaStream_t stream);

extern "C" void qgemm_launch_256_32_32_1_1(
    const int8_t* X, const int8_t* W,
    const float* x_scale, const float* w_scale,
    void* Y_half_ptr, int M, int N, int K, cudaStream_t stream);

extern "C" void qgemm_launch_64_256_32_1_1(
    const int8_t* X, const int8_t* W,
    const float* x_scale, const float* w_scale,
    void* Y_half_ptr, int M, int N, int K, cudaStream_t stream);

// Generic wrapper function
torch::Tensor qgemm_launch_wrapper(
    const std::string& config_key,
    torch::Tensor x_int8,
    torch::Tensor w_int8,
    torch::Tensor x_scale,
    torch::Tensor w_scale)
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
    TORCH_CHECK(w_int8.size(1) == K, "K mismatch");
    TORCH_CHECK(x_scale.numel() == M, "x_scale must have M elements");
    TORCH_CHECK(w_scale.numel() == N, "w_scale must have N elements");

    auto y = torch::empty({M, N}, x_int8.options().dtype(torch::kFloat16));

    const int Mi = static_cast<int>(M);
    const int Ni = static_cast<int>(N);
    const int Ki = static_cast<int>(K);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    // Dispatch to appropriate kernel
    if (config_key == "64_64_32_1_1") {
        qgemm_launch_64_64_32_1_1(
            x_int8.data_ptr<int8_t>(), w_int8.data_ptr<int8_t>(),
            x_scale.data_ptr<float>(), w_scale.data_ptr<float>(),
            y.data_ptr<at::Half>(), Mi, Ni, Ki, stream);
    } else if (config_key == "128_64_32_1_1") {
        qgemm_launch_128_64_32_1_1(
            x_int8.data_ptr<int8_t>(), w_int8.data_ptr<int8_t>(),
            x_scale.data_ptr<float>(), w_scale.data_ptr<float>(),
            y.data_ptr<at::Half>(), Mi, Ni, Ki, stream);
    } else if (config_key == "128_64_32_2_2") {
        qgemm_launch_128_64_32_2_2(
            x_int8.data_ptr<int8_t>(), w_int8.data_ptr<int8_t>(),
            x_scale.data_ptr<float>(), w_scale.data_ptr<float>(),
            y.data_ptr<at::Half>(), Mi, Ni, Ki, stream);
    } else if (config_key == "256_64_32_1_1") {
        qgemm_launch_256_64_32_1_1(
            x_int8.data_ptr<int8_t>(), w_int8.data_ptr<int8_t>(),
            x_scale.data_ptr<float>(), w_scale.data_ptr<float>(),
            y.data_ptr<at::Half>(), Mi, Ni, Ki, stream);
    } else if (config_key == "256_64_32_2_2") {
        qgemm_launch_256_64_32_2_2(
            x_int8.data_ptr<int8_t>(), w_int8.data_ptr<int8_t>(),
            x_scale.data_ptr<float>(), w_scale.data_ptr<float>(),
            y.data_ptr<at::Half>(), Mi, Ni, Ki, stream);
    } else if (config_key == "128_128_32_1_1") {
        qgemm_launch_128_128_32_1_1(
            x_int8.data_ptr<int8_t>(), w_int8.data_ptr<int8_t>(),
            x_scale.data_ptr<float>(), w_scale.data_ptr<float>(),
            y.data_ptr<at::Half>(), Mi, Ni, Ki, stream);
    } else if (config_key == "256_32_32_1_1") {
        qgemm_launch_256_32_32_1_1(
            x_int8.data_ptr<int8_t>(), w_int8.data_ptr<int8_t>(),
            x_scale.data_ptr<float>(), w_scale.data_ptr<float>(),
            y.data_ptr<at::Half>(), Mi, Ni, Ki, stream);
    } else if (config_key == "64_256_32_1_1") {
        qgemm_launch_64_256_32_1_1(
            x_int8.data_ptr<int8_t>(), w_int8.data_ptr<int8_t>(),
            x_scale.data_ptr<float>(), w_scale.data_ptr<float>(),
            y.data_ptr<at::Half>(), Mi, Ni, Ki, stream);
    } else {
        throw std::runtime_error("Unknown config key: " + config_key);
    }

    return y;
}

// Individual exports for direct access
torch::Tensor qgemm_launch_64_64_32_1_1_py(
    torch::Tensor x_int8, torch::Tensor w_int8,
    torch::Tensor x_scale, torch::Tensor w_scale)
{
    return qgemm_launch_wrapper("64_64_32_1_1", x_int8, w_int8, x_scale, w_scale);
}

torch::Tensor qgemm_launch_128_64_32_1_1_py(
    torch::Tensor x_int8, torch::Tensor w_int8,
    torch::Tensor x_scale, torch::Tensor w_scale)
{
    return qgemm_launch_wrapper("128_64_32_1_1", x_int8, w_int8, x_scale, w_scale);
}

torch::Tensor qgemm_launch_128_64_32_2_2_py(
    torch::Tensor x_int8, torch::Tensor w_int8,
    torch::Tensor x_scale, torch::Tensor w_scale)
{
    return qgemm_launch_wrapper("128_64_32_2_2", x_int8, w_int8, x_scale, w_scale);
}

torch::Tensor qgemm_launch_256_64_32_1_1_py(
    torch::Tensor x_int8, torch::Tensor w_int8,
    torch::Tensor x_scale, torch::Tensor w_scale)
{
    return qgemm_launch_wrapper("256_64_32_1_1", x_int8, w_int8, x_scale, w_scale);
}

torch::Tensor qgemm_launch_256_64_32_2_2_py(
    torch::Tensor x_int8, torch::Tensor w_int8,
    torch::Tensor x_scale, torch::Tensor w_scale)
{
    return qgemm_launch_wrapper("256_64_32_2_2", x_int8, w_int8, x_scale, w_scale);
}

torch::Tensor qgemm_launch_128_128_32_1_1_py(
    torch::Tensor x_int8, torch::Tensor w_int8,
    torch::Tensor x_scale, torch::Tensor w_scale)
{
    return qgemm_launch_wrapper("128_128_32_1_1", x_int8, w_int8, x_scale, w_scale);
}

torch::Tensor qgemm_launch_256_32_32_1_1_py(
    torch::Tensor x_int8, torch::Tensor w_int8,
    torch::Tensor x_scale, torch::Tensor w_scale)
{
    return qgemm_launch_wrapper("256_32_32_1_1", x_int8, w_int8, x_scale, w_scale);
}

torch::Tensor qgemm_launch_64_256_32_1_1_py(
    torch::Tensor x_int8, torch::Tensor w_int8,
    torch::Tensor x_scale, torch::Tensor w_scale)
{
    return qgemm_launch_wrapper("64_256_32_1_1", x_int8, w_int8, x_scale, w_scale);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("qgemm_launch_64_64_32_1_1", &qgemm_launch_64_64_32_1_1_py,
          "Small matrix kernel (64x64)");
    m.def("qgemm_launch_128_64_32_1_1", &qgemm_launch_128_64_32_1_1_py,
          "Medium matrix baseline (128x64)");
    m.def("qgemm_launch_128_64_32_2_2", &qgemm_launch_128_64_32_2_2_py,
          "Medium matrix with 2x2 register blocking (128x64)");
    m.def("qgemm_launch_256_64_32_1_1", &qgemm_launch_256_64_32_1_1_py,
          "Large matrix kernel (256x64)");
    m.def("qgemm_launch_256_64_32_2_2", &qgemm_launch_256_64_32_2_2_py,
          "Large matrix with 2x2 register blocking (256x64)");
    m.def("qgemm_launch_128_128_32_1_1", &qgemm_launch_128_128_32_1_1_py,
          "Square matrix kernel (128x128)");
    m.def("qgemm_launch_256_32_32_1_1", &qgemm_launch_256_32_32_1_1_py,
          "Tall-and-skinny kernel (256x32)");
    m.def("qgemm_launch_64_256_32_1_1", &qgemm_launch_64_256_32_1_1_py,
          "Wide-and-short kernel (64x256)");
}
