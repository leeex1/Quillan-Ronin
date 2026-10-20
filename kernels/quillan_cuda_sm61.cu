// ==============================================================================
// QUILLAN-RONIN v5.4.0-ONI: PASCAL sm_61 NATIVE __dp4a TERNARY CUDA KERNEL
// Target Architecture: NVIDIA Pascal GP107 (GTX 1050 4GB, sm_61)
// Instruction: __dp4a(int32_t srcA, int32_t srcB, int32_t c)
// ==============================================================================

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdint.h>
#include <stdio.h>

#define WARP_SIZE 32
#define FULL_MASK 0xffffffff

// Native hardware __dp4a intrinsic on sm_61+ (included via sm_61_intrinsics.h)
// Computes c + sum_{i=0..3} (srcA[i] * srcB[i]) in a single GPU cycle.
#include <sm_61_intrinsics.h>

// ==============================================================================
// 1. BitNet 1.58b GEMV Kernel using __dp4a
// Computes y = (W_ternary * x_quant) * (row_scale * x_scale)
// Weights are packed 4 per 32-bit int: 4 signed int8 values in {-1, 0, +1}
// ==============================================================================
__global__ void bitnet_gemv_dp4a_kernel(
    const int32_t* __restrict__ W_packed,  // [M, K/4]
    const int32_t* __restrict__ X_packed,  // [K/4]
    float*         __restrict__ Y,         // [M]
    const float*   __restrict__ row_scales,// [M]
    float                       x_scale,   // Scalar activation scale
    int                         M,         // Output dimension
    int                         K_div4     // In_dim / 4
) {
    int row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= M) return;

    int tid_x = threadIdx.x; // 0..31 inside warp
    int acc = 0;

    const int32_t* w_row = W_packed + row * K_div4;

    // Warp-stride loop over K/4
    for (int k = tid_x; k < K_div4; k += WARP_SIZE) {
        int32_t w4 = w_row[k];
        int32_t x4 = X_packed[k];
        acc = __dp4a(w4, x4, acc);
    }

    // Warp reduction
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        acc += __shfl_down_sync(FULL_MASK, acc, offset);
    }

    // Write final output
    if (tid_x == 0) {
        float r_scale = row_scales ? row_scales[row] : 1.0f;
        Y[row] = (float)acc * r_scale * x_scale;
    }
}

// ==============================================================================
// 2. Fused 34-Expert Dense-Pull LoRA + Swarm Rank-8 Kernel
// Computes: out = sum_{e=0..33} pull[e] * (x * A_e * B_e)
// x: [1, in_dim], A: [34, in_dim, 8], B: [34, 8, out_dim], pull: [34]
// Avoids launching 34 separate kernels and ping-ponging activation buffers.
// ==============================================================================
__global__ void fused_34expert_rank8_kernel(
    const float* __restrict__ X,          // [in_dim]
    const float* __restrict__ A_all,      // [34, in_dim, 8]
    const float* __restrict__ B_all,      // [34, 8, out_dim]
    const float* __restrict__ pull_gate,  // [34]
    float*       __restrict__ Out,        // [out_dim]
    int                       in_dim,
    int                       out_dim
) {
    int o = blockIdx.x * blockDim.x + threadIdx.x;
    if (o >= out_dim) return;

    float total_out = 0.0f;

    for (int e = 0; e < 34; ++e) {
        float p = pull_gate[e];
        if (p < 1e-6f) continue; // Skip negligible expert contributions

        const float* A_e = A_all + e * (in_dim * 8);
        const float* B_e = B_all + e * (8 * out_dim);

        // Compute 8-element intermediate bottleneck: h[r] = sum_k (X[k] * A_e[k, r])
        float h[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        for (int k = 0; k < in_dim; ++k) {
            float x_k = X[k];
            #pragma unroll
            for (int r = 0; r < 8; ++r) {
                h[r] += x_k * A_e[k * 8 + r];
            }
        }

        // Project intermediate bottleneck through B_e to output o
        float exp_contrib = 0.0f;
        #pragma unroll
        for (int r = 0; r < 8; ++r) {
            exp_contrib += h[r] * B_e[r * out_dim + o];
        }

        total_out += p * exp_contrib;
    }

    Out[o] = total_out;
}

// ==============================================================================
// 3. C-Exportable Function Wrappers for DLL / Ctypes / Torch Dispatch
// ==============================================================================
extern "C" {

#ifdef _WIN32
__declspec(dllexport)
#endif
void launch_bitnet_gemv_dp4a(
    const int32_t* W_packed,
    const int32_t* X_packed,
    float*         Y,
    const float*   row_scales,
    float          x_scale,
    int            M,
    int            K_div4,
    cudaStream_t   stream
) {
    dim3 block(WARP_SIZE, 4); // 4 rows per block, 32 threads per row
    dim3 grid((M + 3) / 4);

    bitnet_gemv_dp4a_kernel<<<grid, block, 0, stream>>>(
        W_packed, X_packed, Y, row_scales, x_scale, M, K_div4
    );
}

#ifdef _WIN32
__declspec(dllexport)
#endif
void launch_fused_34expert_rank8(
    const float* X,
    const float* A_all,
    const float* B_all,
    const float* pull_gate,
    float*       Out,
    int          in_dim,
    int          out_dim,
    cudaStream_t stream
) {
    int threads = 128;
    int blocks = (out_dim + threads - 1) / threads;

    fused_34expert_rank8_kernel<<<blocks, threads, 0, stream>>>(
        X, A_all, B_all, pull_gate, Out, in_dim, out_dim
    );
}

#ifdef _WIN32
__declspec(dllexport)
#endif
int query_sm61_support() {
    int device = 0;
    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, device) == cudaSuccess) {
        if (prop.major == 6 && prop.minor == 1) {
            return 61; // Pascal sm_61 native support confirmed
        }
        return prop.major * 10 + prop.minor;
    }
    return 0;
}

} // extern "C"
