// sm61_gemm_sk.cu — SPLIT-K INT8 GEMM for skinny M (decode/prefill small batch)
// Target: GTX 1050 GP107 (sm_61), 5 SMs. Persistent grid = 5 CTAs.
//
// Why this exists (measured gap, bench_kernel_baseline):
//   qgemm v1/v2 use TILE_M=128 blocks. At M=1..32 the tile is 75-99% wasted
//   and GPU LOSES to CPU (0.46ms vs 0.08ms at M=1). For decode/prefill-small:
//   - Persistent: 5 CTAs (one per SM), loop over (M-tile x N-tile) work items
//   - Split-K: each K slice computed by a different CTA -> parallelize the
//     reduction dimension when N-tiles x M-tiles < SMs
//   - Partial sums to workspace [splits, M, N] fp32, final reduce by last CTA
//     per output tile (atomic turnstile via counter, all CTAs resident).
//
// Layout: X [M,K] int8, W [N,K] int8 (pre-transposed), Y [M,N] fp32.
// Requires K % 32 == 0 (TILE_K=32), M <= 512 (one M-tile of 32 covers decode).

#include <cstdint>
#include <sm_61_intrinsics.h>

#define TILE_M 32
#define TILE_N 64
#define TILE_K 32
#define BLOCK_M 4    // 32/4 = 8 threads in Y
#define BLOCK_N 4    // 64/4 = 16 threads in X -> 128 threads
#define SMEM_K_PAD 48

extern "C" __global__ __launch_bounds__(128)
void qgemm_sk_kernel(
    const int8_t* __restrict__ X,
    const int8_t* __restrict__ W,
    const float*  __restrict__ x_scale,
    const float*  __restrict__ w_scale,
    float*        __restrict__ Y,
    float*        __restrict__ WS,     // [nsplits, M, N] partials + counters
    int*          __restrict__ CNTRS,  // [mtiles * ntiles] arrival counters
    int M, int N, int K, int nsplits)
{
    __align__(16) __shared__ int8_t sX[2][TILE_M][SMEM_K_PAD];
    __align__(16) __shared__ int8_t sW[2][TILE_N][SMEM_K_PAD];

    const int tx = threadIdx.x;   // 0..15
    const int ty = threadIdx.y;   // 0..7
    const int tid = ty * 16 + tx;
    constexpr int NTHREADS = 128;

    const int nMt = (M + TILE_M - 1) / TILE_M;
    const int nNt = (N + TILE_N - 1) / TILE_N;
    const int nTiles = nMt * nNt;
    const int kTilesTotal = K / TILE_K;
    const int kTilesPerSplit = (kTilesTotal + nsplits - 1) / nsplits;

    // Persistent: strided tile loop (grid = 5 CTAs, exact SM count).
    for (int tile = blockIdx.x; tile < nTiles; tile += gridDim.x) {
        const int tileM = tile / nNt;
        const int tileN = tile - tileM * nNt;
        const int mBase = tileM * TILE_M;
        const int nBase = tileN * TILE_N;

        int32_t acc[BLOCK_M][BLOCK_N] = {};
        const int sk = blockIdx.x;  // this CTA's K split index... (see below)

        // Each CTA takes a K-slice: k0 = split * kTilesPerSplit
        // NOTE: grid is 5 CTAs but nsplits can exceed... use work-item index:
        // worker id wid spans [0, nTiles*nsplits). Persistent over wid with
        // stride gridDim.x. worker -> (tile, split).
        // (Recomputed per wid for simplicity.)
        for (int wid = blockIdx.x; wid < nTiles * nsplits; wid += gridDim.x) {
            const int t = wid / nsplits;
            const int s = wid - t * nsplits;
            const int tm = t / nNt;
            const int tn = t - tm * nNt;
            const int mb = tm * TILE_M;
            const int nb = tn * TILE_N;
            const int k0 = s * kTilesPerSplit * TILE_K;
            const int k1 = min(k0 + kTilesPerSplit * TILE_K, K);

            for (int bm = 0; bm < BLOCK_M; ++bm)
                for (int bn = 0; bn < BLOCK_N; ++bn)
                    acc[bm][bn] = 0;

            bool first = true;
            for (int kBase = k0; kBase < k1; kBase += TILE_K) {
                const int stage = first ? 0 : (((kBase / TILE_K) - (k0 / TILE_K)) & 1);
                // Load X tile (TILE_M x TILE_K = 1024B = 64 int4s; 128 threads, half idle)
                if (tid < 64) {
                    const int row = tid >> 1;
                    const int c4  = (tid & 1) * 16;
                    const int g_row = mb + row;
                    int4 v = make_int4(0, 0, 0, 0);
                    if (g_row < M && kBase + c4 + 16 <= K) {
                        v = *reinterpret_cast<const int4*>(X + (long long)g_row * K + kBase + c4);
                    } else if (g_row < M && kBase + c4 < K) {
                        int8_t* d = reinterpret_cast<int8_t*>(&v);
                        for (int i = 0; i < 16; ++i) {
                            int gv = kBase + c4 + i;
                            d[i] = (gv < K) ? X[(long long)g_row * K + gv] : 0;
                        }
                    }
                    *reinterpret_cast<int4*>(&sX[stage][row][c4]) = v;
                }
                // Load W tile (TILE_N x TILE_K = 2048B = 128 int4s; all threads)
                {
                    const int row = tid >> 1;
                    const int c4  = (tid & 1) * 16;
                    const int g_col = nb + row;
                    int4 v = make_int4(0, 0, 0, 0);
                    if (g_col < N && kBase + c4 + 16 <= K) {
                        v = *reinterpret_cast<const int4*>(W + (long long)g_col * K + kBase + c4);
                    } else if (g_col < N && kBase + c4 < K) {
                        int8_t* d = reinterpret_cast<int8_t*>(&v);
                        for (int i = 0; i < 16; ++i) {
                            int gv = kBase + c4 + i;
                            d[i] = (gv < K) ? W[(long long)g_col * K + gv] : 0;
                        }
                    }
                    *reinterpret_cast<int4*>(&sW[stage][row][c4]) = v;
                }
                __syncthreads();

                #pragma unroll
                for (int kk = 0; kk < TILE_K; kk += 4) {
                    int32_t a4[BLOCK_M], b4[BLOCK_N];
                    #pragma unroll
                    for (int bm = 0; bm < BLOCK_M; ++bm)
                        a4[bm] = *reinterpret_cast<const int32_t*>(&sX[stage][ty * BLOCK_M + bm][kk]);
                    #pragma unroll
                    for (int bn = 0; bn < BLOCK_N; ++bn)
                        b4[bn] = *reinterpret_cast<const int32_t*>(&sW[stage][tx * BLOCK_N + bn][kk]);
                    #pragma unroll
                    for (int bm = 0; bm < BLOCK_M; ++bm)
                        #pragma unroll
                        for (int bn = 0; bn < BLOCK_N; ++bn)
                            acc[bm][bn] = __dp4a(a4[bm], b4[bn], acc[bm][bn]);
                }
                __syncthreads();
                (void)first; first = false;
            }

            // Write partials to workspace [s, M, N]
            float* wsp = WS + ((long long)s * M + mb) * N + nb;
            const int r0 = ty * BLOCK_M, c0 = tx * BLOCK_N;
            #pragma unroll
            for (int bm = 0; bm < BLOCK_M; ++bm) {
                const int row = mb + r0 + bm;
                if (row < M)
                    #pragma unroll
                    for (int bn = 0; bn < BLOCK_N; ++bn) {
                        const int col = nb + c0 + bn;
                        if (col < N) wsp[(long long)(r0 + bm) * N + (c0 + bn)] = (float)acc[bm][bn];
                    }
            }
            __threadfence();
            __syncthreads();

            // EVERY split worker arrives (reduction is once-only, below).
            if (tid == 0) atomicAdd(&CNTRS[t], 1);
            __syncthreads();

            // Only the split-0 worker reduces (exactly one reducer per tile).
            if (s != 0) continue;

            // Spin until all splits arrived (all CTAs resident: grid=5).
            // NOTE: nanosleep.u32 is sm_70+ only — plain atomic spin on sm_61.
            if (tid == 0) {
                while (atomicAdd(&CNTRS[t], 0) < nsplits) { asm volatile("" ::: "memory"); }
            }
            __syncthreads();

            // Reduce nsplits partials (first nsplits workers' threads reduce slice)
            const int rows = TILE_M, cols = TILE_N;
            for (int idx = tid; idx < rows * cols; idx += NTHREADS) {
                const int rr = idx / cols, cc = idx - rr * cols;
                const int grow = mb + rr, gcol = nb + cc;
                float sum = 0.f;
                for (int s2 = 0; s2 < nsplits; ++s2)
                    sum += WS[((long long)s2 * M + grow) * N + gcol];
                if (grow < M && gcol < N)
                    Y[(long long)grow * N + gcol] =
                        sum * x_scale[grow] * w_scale[gcol];
            }
            // Reset counter for next call (counters allocated per call, zeroed host-side).
        }
    }
}

// Host-side launcher.
extern "C" void qgemm_sk_launch(
    const int8_t* X, const int8_t* W,
    const float* x_scale, const float* w_scale,
    void* Y_ptr, void* WS_ptr, void* CNTRS_ptr,
    int M, int N, int K, int nsplits, cudaStream_t stream)
{
    dim3 block(16, 8);  // 128 threads
    dim3 grid(5);       // persistent: one CTA per SM (GP107)
    qgemm_sk_kernel<<<grid, block, 0, stream>>>(
        X, W, x_scale, w_scale, reinterpret_cast<float*>(Y_ptr),
        reinterpret_cast<float*>(WS_ptr), reinterpret_cast<int*>(CNTRS_ptr),
        M, N, K, nsplits);
}
