// sm61_qgemm_autotune.cu
//
// Template-based kernel generator for autotuning different configurations
// This file defines multiple kernel variants with different tile sizes and
// register blocking factors for runtime dispatch.

#include <cstdint>
#include <cuda_fp16.h>
#include <sm_61_intrinsics.h>

// Base kernel template with configurable tile sizes and register blocking
template<int TILE_M, int TILE_N, int TILE_K, int BLOCK_M, int BLOCK_N>
__global__ void qgemm_i8_dp4a_kernel_tiled(
    const int8_t* __restrict__ X,
    const int8_t* __restrict__ W,
    const float*  __restrict__ x_scale,
    const float*  __restrict__ w_scale,
    half*         __restrict__ Y,
    int M, int N, int K)
{
    __align__(4) __shared__ int8_t sX[TILE_M][TILE_K];
    __align__(4) __shared__ int8_t sW[TILE_N][TILE_K];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    // Each thread computes BLOCK_M x BLOCK_N outputs (register blocking)
    const int row_start = blockIdx.y * TILE_M + ty * BLOCK_M;
    const int col_start = blockIdx.x * TILE_N + tx * BLOCK_N;

    int32_t acc[BLOCK_M][BLOCK_N] = {0};

    const int kTiles = (K + TILE_K - 1) / TILE_K;

    for (int t = 0; t < kTiles; ++t) {
        const int kBase = t * TILE_K;

        // Load X tile with coalescing
        #pragma unroll
        for (int bm = 0; bm < BLOCK_M; ++bm) {
            int row = row_start + bm;
            if (row < M) {
                for (int kk = tx; kk < TILE_K; kk += TILE_N) {
                    const int k = kBase + kk;
                    sX[ty * BLOCK_M + bm][kk] = (k < K) ? X[row * (long long)K + k] : 0;
                }
            } else {
                for (int kk = tx; kk < TILE_K; kk += TILE_N) {
                    sX[ty * BLOCK_M + bm][kk] = 0;
                }
            }
        }

        // Load W tile with coalescing
        #pragma unroll
        for (int bn = 0; bn < BLOCK_N; ++bn) {
            int col = col_start + bn;
            if (col < N) {
                for (int kk = ty; kk < TILE_K; kk += TILE_M) {
                    const int k = kBase + kk;
                    sW[tx * BLOCK_N + bn][kk] = (k < K) ? W[col * (long long)K + k] : 0;
                }
            } else {
                for (int kk = ty; kk < TILE_K; kk += TILE_M) {
                    sW[tx * BLOCK_N + bn][kk] = 0;
                }
            }
        }

        __syncthreads();

        // Compute with register blocking
        #pragma unroll
        for (int kk = 0; kk < TILE_K; kk += 4) {
            #pragma unroll
            for (int bm = 0; bm < BLOCK_M; ++bm) {
                const int32_t a4 = *reinterpret_cast<const int32_t*>(&sX[ty * BLOCK_M + bm][kk]);
                #pragma unroll
                for (int bn = 0; bn < BLOCK_N; ++bn) {
                    const int32_t b4 = *reinterpret_cast<const int32_t*>(&sW[tx * BLOCK_N + bn][kk]);
                    acc[bm][bn] = __dp4a(a4, b4, acc[bm][bn]);
                }
            }
        }

        __syncthreads();
    }

    // Write results with fused dequantization
    #pragma unroll
    for (int bm = 0; bm < BLOCK_M; ++bm) {
        int row = row_start + bm;
        #pragma unroll
        for (int bn = 0; bn < BLOCK_N; ++bn) {
            int col = col_start + bn;
            if (row < M && col < N) {
                const float deq = static_cast<float>(acc[bm][bn]) * x_scale[row] * w_scale[col];
                Y[row * (long long)N + col] = __float2half(deq);
            }
        }
    }
}

// Kernel launcher template
template<int TILE_M, int TILE_N, int TILE_K, int BLOCK_M, int BLOCK_N>
void qgemm_i8_dp4a_launch_tiled(
    const int8_t* X, const int8_t* W,
    const float* x_scale, const float* w_scale,
    void* Y_half_ptr, int M, int N, int K, cudaStream_t stream)
{
    const int threads_per_block_x = TILE_N / BLOCK_N;
    const int threads_per_block_y = TILE_M / BLOCK_M;
    dim3 block(threads_per_block_x, threads_per_block_y);
    dim3 grid((N + TILE_N - 1) / TILE_N, (M + TILE_M - 1) / TILE_M);

    qgemm_i8_dp4a_kernel_tiled<TILE_M, TILE_N, TILE_K, BLOCK_M, BLOCK_N>
        <<<grid, block, 0, stream>>>(
            X, W, x_scale, w_scale, reinterpret_cast<half*>(Y_half_ptr), M, N, K);
}

// Explicit instantiations for common configurations
// Format: qgemm_launch_Tm_Tn_Tk_Bm_Bn

// Small matrices (64x64)
extern "C" void qgemm_launch_64_64_32_1_1(
    const int8_t* X, const int8_t* W,
    const float* x_scale, const float* w_scale,
    void* Y_half_ptr, int M, int N, int K, cudaStream_t stream)
{
    qgemm_i8_dp4a_launch_tiled<64, 64, 32, 1, 1>(
        X, W, x_scale, w_scale, Y_half_ptr, M, N, K, stream);
}

// Medium matrices (128x64) - baseline
extern "C" void qgemm_launch_128_64_32_1_1(
    const int8_t* X, const int8_t* W,
    const float* x_scale, const float* w_scale,
    void* Y_half_ptr, int M, int N, int K, cudaStream_t stream)
{
    qgemm_i8_dp4a_launch_tiled<128, 64, 32, 1, 1>(
        X, W, x_scale, w_scale, Y_half_ptr, M, N, K, stream);
}

// Medium matrices with 2x2 register blocking
extern "C" void qgemm_launch_128_64_32_2_2(
    const int8_t* X, const int8_t* W,
    const float* x_scale, const float* w_scale,
    void* Y_half_ptr, int M, int N, int K, cudaStream_t stream)
{
    qgemm_i8_dp4a_launch_tiled<128, 64, 32, 2, 2>(
        X, W, x_scale, w_scale, Y_half_ptr, M, N, K, stream);
}

// Large matrices (256x64)
extern "C" void qgemm_launch_256_64_32_1_1(
    const int8_t* X, const int8_t* W,
    const float* x_scale, const float* w_scale,
    void* Y_half_ptr, int M, int N, int K, cudaStream_t stream)
{
    qgemm_i8_dp4a_launch_tiled<256, 64, 32, 1, 1>(
        X, W, x_scale, w_scale, Y_half_ptr, M, N, K, stream);
}

// Large matrices with 2x2 register blocking
extern "C" void qgemm_launch_256_64_32_2_2(
    const int8_t* X, const int8_t* W,
    const float* x_scale, const float* w_scale,
    void* Y_half_ptr, int M, int N, int K, cudaStream_t stream)
{
    qgemm_i8_dp4a_launch_tiled<256, 64, 32, 2, 2>(
        X, W, x_scale, w_scale, Y_half_ptr, M, N, K, stream);
}

// Very large matrices (128x128)
extern "C" void qgemm_launch_128_128_32_1_1(
    const int8_t* X, const int8_t* W,
    const float* x_scale, const float* w_scale,
    void* Y_half_ptr, int M, int N, int K, cudaStream_t stream)
{
    qgemm_i8_dp4a_launch_tiled<128, 128, 32, 1, 1>(
        X, W, x_scale, w_scale, Y_half_ptr, M, N, K, stream);
}

// Tall-and-skinny (256x32)
extern "C" void qgemm_launch_256_32_32_1_1(
    const int8_t* X, const int8_t* W,
    const float* x_scale, const float* w_scale,
    void* Y_half_ptr, int M, int N, int K, cudaStream_t stream)
{
    qgemm_i8_dp4a_launch_tiled<256, 32, 32, 1, 1>(
        X, W, x_scale, w_scale, Y_half_ptr, M, N, K, stream);
}

// Wide-and-short (64x256)
extern "C" void qgemm_launch_64_256_32_1_1(
    const int8_t* X, const int8_t* W,
    const float* x_scale, const float* w_scale,
    void* Y_half_ptr, int M, int N, int K, cudaStream_t stream)
{
    qgemm_i8_dp4a_launch_tiled<64, 256, 32, 1, 1>(
        X, W, x_scale, w_scale, Y_half_ptr, M, N, K, stream);
}
