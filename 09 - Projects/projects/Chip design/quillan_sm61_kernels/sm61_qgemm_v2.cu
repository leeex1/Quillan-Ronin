// sm61_qgemm_v2.cu
//
// UPGRADED INT8 x INT8 GEMM for NVIDIA Pascal sm_61 (GTX 1050 GP107).
//
// Upgrade over v1 (sm61_qgemm.cu):
//   1. int4 (16B) VECTORIZED global->smem loads: 16 int8s per load instead of 1.
//   2. REGISTER-PREFETCH DOUBLE BUFFERING: while computing tile t, prefetch
//      tile t+1 global->smem. Two smem stages. Eliminates global-load stalls.
//   3. FP32 EPILOGUE (was fp16): Pascal FP16 math is ~1/64 the FP32 rate.
//   4. BANK-CONFLICT PADDING: smem leading dim padded (BK+4).
//   5. __launch_bounds__ for occupancy hinting on 5-SM GP107.
//
// Layout (unchanged from v1):
//   X : [M, K] row-major int8, per-row scale
//   W : [N, K] row-major int8, per-column scale (pre-transposed)
//   Y : [M, N] fp32 output (v1 wrote fp16!)

#include <cstdint>
#include <sm_61_intrinsics.h>   // __dp4a

#define TILE_M 128
#define TILE_N 64
#define TILE_K 32
#define BLOCK_M 8
#define BLOCK_N 4
#define SMEM_K_PAD 48   // TILE_K(32) + pad to 16B alignment (48 % 16 == 0)

extern "C" __global__ __launch_bounds__(256)
void qgemm_i8_dp4a_kernel_v2(
    const int8_t* __restrict__ X,
    const int8_t* __restrict__ W,
    const float*  __restrict__ x_scale,
    const float*  __restrict__ w_scale,
    float*        __restrict__ Y,
    int M, int N, int K)
{
    __align__(16) __shared__ int8_t sX[2][TILE_M][SMEM_K_PAD];
    __align__(16) __shared__ int8_t sW[2][TILE_N][SMEM_K_PAD];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * 16 + tx;

    const int row_start = blockIdx.y * TILE_M + ty * BLOCK_M;
    const int col_start = blockIdx.x * TILE_N + tx * BLOCK_N;

    int32_t acc[BLOCK_M][BLOCK_N];
    #pragma unroll
    for (int bm = 0; bm < BLOCK_M; ++bm)
        #pragma unroll
        for (int bn = 0; bn < BLOCK_N; ++bn)
            acc[bm][bn] = 0;

    const int kTiles = (K + TILE_K - 1) / TILE_K;
    const int mBase = blockIdx.y * TILE_M;
    const int nBase = blockIdx.x * TILE_N;

    // Prologue: load tile 0 (vectorized)
    {
        const int idx = tid;               // 0..255 -> 256 int4s for X (128 rows x 2)
        const int row = idx >> 1;          // 0..127
        const int c4  = (idx & 1) * 16;
        const int g_row = mBase + row;
        const int g_k = c4;
        int4 v = make_int4(0, 0, 0, 0);
        if (g_row < M && g_k + 16 <= K) {
            v = *reinterpret_cast<const int4*>(X + (long long)g_row * K + g_k);
        } else if (g_row < M && g_k < K) {
            int8_t* dst = reinterpret_cast<int8_t*>(&v);
            for (int i = 0; i < 16; ++i) {
                int gv = g_k + i;
                dst[i] = (gv < K) ? X[(long long)g_row * K + gv] : 0;
            }
        }
        *reinterpret_cast<int4*>(&sX[0][row][c4]) = v;
        if (tid < 128) {
            const int wrow = tid >> 1;
            const int wc4  = (tid & 1) * 16;
            const int g_col = nBase + wrow;
            int4 wv = make_int4(0, 0, 0, 0);
            if (g_col < N && wc4 + 16 <= K) {
                wv = *reinterpret_cast<const int4*>(W + (long long)g_col * K + wc4);
            } else if (g_col < N && wc4 < K) {
                int8_t* dst = reinterpret_cast<int8_t*>(&wv);
                for (int i = 0; i < 16; ++i) {
                    int gv = wc4 + i;
                    dst[i] = (gv < K) ? W[(long long)g_col * K + gv] : 0;
                }
            }
            *reinterpret_cast<int4*>(&sW[0][wrow][wc4]) = wv;
        }
    }
    __syncthreads();

    for (int t = 0; t < kTiles; ++t) {
        const int stage = t & 1;
        const int kBase = t * TILE_K;

        // Prefetch next tile into other stage (overlaps with compute below)
        if (t + 1 < kTiles) {
            const int nkBase = kBase + TILE_K;
            const int idx = tid;
            const int row = idx >> 1;
            const int c4  = (idx & 1) * 16;
            const int g_row = mBase + row;
            const int g_k = nkBase + c4;
            int4 v = make_int4(0, 0, 0, 0);
            if (g_row < M && g_k + 16 <= K) {
                v = *reinterpret_cast<const int4*>(X + (long long)g_row * K + g_k);
            } else if (g_row < M && g_k < K) {
                int8_t* dst = reinterpret_cast<int8_t*>(&v);
                for (int i = 0; i < 16; ++i) {
                    int gv = g_k + i;
                    dst[i] = (gv < K) ? X[(long long)g_row * K + gv] : 0;
                }
            }
            *reinterpret_cast<int4*>(&sX[stage ^ 1][row][c4]) = v;
            if (tid < 128) {
                const int wrow = tid >> 1;
                const int wc4  = (tid & 1) * 16;
                const int g_col = nBase + wrow;
                int4 wv = make_int4(0, 0, 0, 0);
                if (g_col < N && g_k + 16 <= K) {
                    wv = *reinterpret_cast<const int4*>(W + (long long)g_col * K + g_k);
                } else if (g_col < N && g_k < K) {
                    int8_t* dst = reinterpret_cast<int8_t*>(&wv);
                    for (int i = 0; i < 16; ++i) {
                        int gv = g_k + i;
                        dst[i] = (gv < K) ? W[(long long)g_col * K + gv] : 0;
                    }
                }
                *reinterpret_cast<int4*>(&sW[stage ^ 1][wrow][wc4]) = wv;
            }
        }

        // Compute on current stage
        #pragma unroll
        for (int kk = 0; kk < TILE_K; kk += 4) {
            int32_t a4[BLOCK_M];
            int32_t b4[BLOCK_N];
            #pragma unroll
            for (int bm = 0; bm < BLOCK_M; ++bm) {
                a4[bm] = *reinterpret_cast<const int32_t*>(&sX[stage][ty * BLOCK_M + bm][kk]);
            }
            #pragma unroll
            for (int bn = 0; bn < BLOCK_N; ++bn) {
                b4[bn] = *reinterpret_cast<const int32_t*>(&sW[stage][tx * BLOCK_N + bn][kk]);
            }
            #pragma unroll
            for (int bm = 0; bm < BLOCK_M; ++bm) {
                #pragma unroll
                for (int bn = 0; bn < BLOCK_N; ++bn) {
                    acc[bm][bn] = __dp4a(a4[bm], b4[bn], acc[bm][bn]);
                }
            }
        }

        if (t + 1 < kTiles) {
            __syncthreads();
        }
    }

    // FP32 epilogue (NOT fp16 - Pascal fp16 is 1/64 rate!)
    #pragma unroll
    for (int bm = 0; bm < BLOCK_M; ++bm) {
        const int row = row_start + bm;
        if (row < M) {
            const float xs = x_scale[row];
            #pragma unroll
            for (int bn = 0; bn < BLOCK_N; ++bn) {
                const int col = col_start + bn;
                if (col < N) {
                    Y[(long long)row * N + col] = (float)acc[bm][bn] * xs * w_scale[col];
                }
            }
        }
    }
}

// Host-side launcher.
extern "C" void qgemm_i8_dp4a_launch_v2(
    const int8_t* X, const int8_t* W,
    const float* x_scale, const float* w_scale,
    void* Y_ptr, int M, int N, int K, cudaStream_t stream)
{
    dim3 block(16, 16);
    dim3 grid((N + TILE_N - 1) / TILE_N, (M + TILE_M - 1) / TILE_M);
    qgemm_i8_dp4a_kernel_v2<<<grid, block, 0, stream>>>(
        X, W, x_scale, w_scale, reinterpret_cast<float*>(Y_ptr), M, N, K);
}
