// sm61_qgemm.cu
//
// Custom INT8 x INT8 GEMM kernel targeting NVIDIA Pascal (compute capability 6.1,
// e.g. GTX 1050 / GP107) using the native DP4A instruction.
//
//   Y[M,N] = dequant( X_int8[M,K] @ W_int8[K,N]^T )
//
// DP4A (__dp4a) is a real Pascal-and-later instruction: it computes a 4-wide
// int8 dot product plus accumulate in a single instruction. It's the correct
// primitive for quantized inference on this hardware -- Pascal has no tensor
// cores (those start at Volta, sm_70), so DP4A is the fastest integer path
// available on sm_61.
//
// Layout convention:
//   X : [M, K] row-major, int8, per-ROW dynamic quantization (activations)
//   W : [N, K] row-major, int8, per-COLUMN (i.e. per output channel) static
//       quantization -- this is W pre-transposed relative to a normal
//       [K, N] weight so that both X and W are read contiguously along K
//       inside the kernel (coalesced loads, no on-the-fly transpose).
//   Y : [M, N] fp16 output
//
// This is a correctness-first tiled shared-memory reference kernel, not a
// max-tuned production kernel. It has no double-buffering, no register
// blocking beyond 1 output/thread, and no bank-conflict-optimized shared
// memory swizzling. Profile it against cuBLAS-int8 / cutlass before trusting
// it as your fast path in anything latency-critical.

#include <cstdint>
#include <cuda_fp16.h>
#include <sm_61_intrinsics.h>   // declares __dp4a on sm_61+

#define TILE_M 32
#define TILE_N 32
#define TILE_K 32   // must be a multiple of 4 (dp4a packs 4 int8 lanes per int32 word)

extern "C" __global__
void qgemm_i8_dp4a_kernel(
    const int8_t* __restrict__ X,       // [M, K]
    const int8_t* __restrict__ W,       // [N, K]  (pre-transposed weight)
    const float*  __restrict__ x_scale, // [M]  per-row activation scale
    const float*  __restrict__ w_scale, // [N]  per-column weight scale
    half*         __restrict__ Y,       // [M, N]
    int M, int N, int K)
{
    __align__(4) __shared__ int8_t sX[TILE_M][TILE_K];
    __align__(4) __shared__ int8_t sW[TILE_N][TILE_K];

    const int tx = threadIdx.x;   // 0..TILE_N-1
    const int ty = threadIdx.y;   // 0..TILE_M-1

    const int row = blockIdx.y * TILE_M + ty;  // index into M
    const int col = blockIdx.x * TILE_N + tx;  // index into N

    int32_t acc = 0;
    const int kTiles = (K + TILE_K - 1) / TILE_K;

    for (int t = 0; t < kTiles; ++t) {
        const int kBase = t * TILE_K;

        // Cooperative load of the X tile (one element per thread column-stride).
        if (row < M) {
            for (int kk = tx; kk < TILE_K; kk += TILE_N) {
                const int k = kBase + kk;
                sX[ty][kk] = (k < K) ? X[row * (long long)K + k] : 0;
            }
        } else {
            for (int kk = tx; kk < TILE_K; kk += TILE_N) sX[ty][kk] = 0;
        }

        // Cooperative load of the W tile.
        if (col < N) {
            for (int kk = ty; kk < TILE_K; kk += TILE_M) {
                const int k = kBase + kk;
                sW[tx][kk] = (k < K) ? W[col * (long long)K + k] : 0;
            }
        } else {
            for (int kk = ty; kk < TILE_K; kk += TILE_M) sW[tx][kk] = 0;
        }

        __syncthreads();

        #pragma unroll
        for (int kk = 0; kk < TILE_K; kk += 4) {
            const int32_t a4 = *reinterpret_cast<const int32_t*>(&sX[ty][kk]);
            const int32_t b4 = *reinterpret_cast<const int32_t*>(&sW[tx][kk]);
            acc = __dp4a(a4, b4, acc);
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        const float deq = static_cast<float>(acc) * x_scale[row] * w_scale[col];
        Y[row * (long long)N + col] = __float2half(deq);
    }
}

// Host-side launcher. Kept CUDA-only (no ATen types) so this file only needs
// the CUDA toolkit to compile; the ATen <-> raw-pointer bridging happens in
// qgemm_binding.cpp.
extern "C" void qgemm_i8_dp4a_launch(
    const int8_t* X, const int8_t* W,
    const float* x_scale, const float* w_scale,
    void* Y_half_ptr, int M, int N, int K, cudaStream_t stream)
{
    dim3 block(TILE_N, TILE_M);
    dim3 grid((N + TILE_N - 1) / TILE_N, (M + TILE_M - 1) / TILE_M);
    qgemm_i8_dp4a_kernel<<<grid, block, 0, stream>>>(
        X, W, x_scale, w_scale, reinterpret_cast<half*>(Y_half_ptr), M, N, K);
}
