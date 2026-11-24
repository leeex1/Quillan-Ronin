// quillan_avx2_ternary.cpp — Fast AVX2 BitNet 1.58b GEMV Kernel (Q13 Solution)
// Compile with: g++ -O3 -mavx2 -mfma -fopenmp quillan_avx2_ternary.cpp
#include <immintrin.h>
#include <cstdint>

// Weights are packed 4 per byte as unsigned 2-bit offset values: w_u2 = (w + 1) in {0, 1, 2}
// Activations x_q are INT8 quantized in [-127, 127].
// Identity used: sum_j (w_j + 1) * x_j - sum_j x_j == sum_j w_j * x_j
void bitnet_gemv_avx2_i2s(
    const uint8_t* __restrict__ w_packed,    // [out_dim, in_dim / 4]
    const int8_t*  __restrict__ x_quant,     // [in_dim]
    float*         __restrict__ out,         // [out_dim]
    const float*   __restrict__ row_scales,
    float                       x_scale,
    int                         out_dim,
    int                         in_dim
) {
    // 1. Precompute sum_j(x_j) ONCE per vector (cancels the +1 weight offset for every row!)
    int32_t x_sum = 0;
    for (int j = 0; j < in_dim; ++j) {
        x_sum += static_cast<int32_t>(x_quant[j]);
    }

    const __m256i mask_2bit = _mm256_set1_epi8(0x03);
    const __m256i ones_16   = _mm256_set1_epi16(1);

    #pragma omp parallel for schedule(static)
    for (int row = 0; row < out_dim; ++row) {
        const uint8_t* w_row = w_packed + row * (in_dim / 4);
        __m256i acc_i32 = _mm256_setzero_si256();

        // Process 128 ternary weights (32 packed bytes) per loop iteration
        int j = 0;
        for (; j + 127 < in_dim; j += 128) {
            __m256i w_bytes = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(w_row + (j >> 2)));

            // Unpack 4 streams of 32 x 2-bit unsigned weights in {0, 1, 2}
            __m256i w0 = _mm256_and_si256(w_bytes, mask_2bit);
            __m256i w1 = _mm256_and_si256(_mm256_srli_epi16(w_bytes, 2), mask_2bit);
            __m256i w2 = _mm256_and_si256(_mm256_srli_epi16(w_bytes, 4), mask_2bit);
            __m256i w3 = _mm256_and_si256(_mm256_srli_epi16(w_bytes, 6), mask_2bit);

            __m256i x0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x_quant + j));
            __m256i x1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x_quant + j + 32));
            __m256i x2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x_quant + j + 64));
            __m256i x3 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x_quant + j + 96));

            // vpmaddubsw: multiplies 32 uint8 weights by 32 int8 activations -> 16 int16 sums in 1 cycle!
            __m256i p01 = _mm256_add_epi16(_mm256_maddubs_epi16(w0, x0), _mm256_maddubs_epi16(w1, x1));
            __m256i p23 = _mm256_add_epi16(_mm256_maddubs_epi16(w2, x2), _mm256_maddubs_epi16(w3, x3));
            __m256i p_all = _mm256_add_epi16(p01, p23);

            // Widen int16 pairs to int32 and accumulate
            acc_i32 = _mm256_add_epi32(acc_i32, _mm256_madd_epi16(p_all, ones_16));
        }

        // Horizontal reduction of 8 int32 lanes
        __m128i lo = _mm256_castsi256_si128(acc_i32);
        __m128i hi = _mm256_extracti128_si256(acc_i32, 1);
        __m128i sum128 = _mm_add_epi32(lo, hi);
        sum128 = _mm_add_epi32(sum128, _mm_shuffle_epi32(sum128, _MM_SHUFFLE(1, 0, 3, 2)));
        sum128 = _mm_add_epi32(sum128, _mm_shuffle_epi32(sum128, _MM_SHUFFLE(2, 3, 0, 1)));
        int32_t dot_offset = _mm_cvtsi128_si32(sum128);

        // Subtract x_sum to remove the +1 weight offset and scale to FP32
        int32_t true_ternary_dot = dot_offset - x_sum;
        out[row] = static_cast<float>(true_ternary_dot) * row_scales[row] * x_scale;
    }
}
