#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// -------------------------------------------------------------------------
// Quillan-Ronin v5.4.0 ONI Native CUDA BitNet 1.58b Kernel
// Supports Pascal sm_61 (GTX 1050) through Ada/Hopper sm_90
// -------------------------------------------------------------------------

__global__ void bitnet_weight_quant_cuda_kernel(
    const float* __restrict__ w,
    float* __restrict__ out,
    int64_t rows,
    int64_t cols,
    float eps
) {
    int64_t r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) return;

    const float* row_in = w + r * cols;
    float* row_out = out + r * cols;

    float sum = 0.0f;
    for (int64_t c = 0; c < cols; ++c) {
        sum += fabsf(row_in[c]);
    }
    float mean = sum / (float)cols;
    float scale = (mean < eps) ? (1.0f / eps) : (1.0f / mean);
    float inv_scale = (mean < eps) ? eps : mean;

    for (int64_t c = 0; c < cols; ++c) {
        float val = row_in[c] * scale;
        float q = roundf(fminf(1.0f, fmaxf(-1.0f, val)));
        row_out[c] = q * inv_scale;
    }
}

torch::Tensor bitnet_weight_quant_cuda(torch::Tensor w, float eps) {
    auto w_contig = w.contiguous();
    auto out = torch::empty_like(w_contig);

    int64_t rows = w_contig.size(0);
    int64_t cols = w_contig.size(1);

    int threads = 256;
    int blocks = (rows + threads - 1) / threads;

    bitnet_weight_quant_cuda_kernel<<<blocks, threads>>>(
        w_contig.data_ptr<float>(),
        out.data_ptr<float>(),
        rows, cols, eps
    );

    return out;
}

torch::Tensor nine_vector_prism_forward_cuda(torch::Tensor x, torch::Tensor w_stacked, torch::Tensor w_gate) {
    // Parallel CUDA tensor contraction across 9 semantic vectors
    auto prism = at::einsum("bld,ned->ble", {x, w_stacked}) / 9.0f;
    return at::linear(prism, w_gate);
}
