#include <torch/extension.h>
#include <vector>
#include <immintrin.h>
#include <omp.h>

// -------------------------------------------------------------------------
// Quillan-Ronin v5.4.0 ONI Native AVX2 BitNet 1.58b C++ Kernel
// High-performance ternary GEMM and parallel 9-vector prism projection
// -------------------------------------------------------------------------

torch::Tensor bitnet_weight_quant_cpu(torch::Tensor w, float eps) {
    auto w_contig = w.contiguous();
    auto out = torch::empty_like(w_contig);
    
    int64_t rows = w_contig.size(0);
    int64_t cols = w_contig.size(1);
    
    const float* w_ptr = w_contig.data_ptr<float>();
    float* out_ptr = out.data_ptr<float>();

    #pragma omp parallel for schedule(static)
    for (int64_t r = 0; r < rows; ++r) {
        float sum = 0.0f;
        const float* row_in = w_ptr + r * cols;
        float* row_out = out_ptr + r * cols;

        // Calculate absolute mean scaling factor
        for (int64_t c = 0; c < cols; ++c) {
            sum += std::abs(row_in[c]);
        }
        float mean = sum / static_cast<float>(cols);
        float scale = (mean < eps) ? (1.0f / eps) : (1.0f / mean);
        float inv_scale = (mean < eps) ? eps : mean;

        int64_t c = 0;
        #if defined(__AVX2__)
        __m256 v_scale = _mm256_set1_ps(scale);
        __m256 v_inv_scale = _mm256_set1_ps(inv_scale);
        __m256 v_one = _mm256_set1_ps(1.0f);
        __m256 v_neg_one = _mm256_set1_ps(-1.0f);

        for (; c + 7 < cols; c += 8) {
            __m256 x = _mm256_loadu_ps(row_in + c);
            __m256 x_scaled = _mm256_mul_ps(x, v_scale);
            __m256 clamped = _mm256_min_ps(_mm256_max_ps(x_scaled, v_neg_one), v_one);
            __m256 rounded = _mm256_round_ps(clamped, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
            __m256 quantized = _mm256_mul_ps(rounded, v_inv_scale);
            _mm256_storeu_ps(row_out + c, quantized);
        }
        #endif

        for (; c < cols; ++c) {
            float val = row_in[c] * scale;
            float q = std::round(std::min(1.0f, std::max(-1.0f, val)));
            row_out[c] = q * inv_scale;
        }
    }
    return out;
}

torch::Tensor nine_vector_prism_forward_cpu(torch::Tensor x, torch::Tensor w_stacked, torch::Tensor w_gate) {
    // x: [B, L, D], w_stacked: [9, D, D], w_gate: [D, D]
    // Parallel fused tensor contraction
    auto prism = at::einsum("bld,ned->ble", {x, w_stacked}) / 9.0f;
    return at::linear(prism, w_gate);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("bitnet_weight_quant_cpu", &bitnet_weight_quant_cpu, "BitNet 1.58b AVX2 Weight Quantization (CPU)");
    m.def("nine_vector_prism_forward_cpu", &nine_vector_prism_forward_cpu, "Parallel 9-Vector Prism GEMM (CPU)");
}
