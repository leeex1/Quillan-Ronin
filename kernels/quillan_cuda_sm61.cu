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
#define NUM_EXPERTS 34
#define LORA_RANK 8

// Native hardware __dp4a intrinsic on sm_61+ (included via sm_61_intrinsics.h)
// Computes c + sum_{i=0..3} (srcA[i] * srcB[i]) in a single GPU cycle.
#include <sm_61_intrinsics.h>

// ==============================================================================
// 1. BitNet 1.58b GEMV Kernel using __dp4a
// Computes y = (W_ternary * x_quant) * (row_scale * x_scale)
// Weights are packed 4 per 32-bit int: 4 signed int8 values in {-1, 0, +1}
// ==============================================================================
__global__ void bitnet_gemv_dp4a_kernel(
    const int32_t* __restrict__ W_packed,  // [M, K_div4]
    const int32_t* __restrict__ X_packed,  // [K_div4]
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
    #pragma unroll
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
// 2. Fused 34-Expert Dense-Pull LoRA + Swarm Rank-8 Kernel (CORRECTED)
// Fixes:
//   1. Eliminates O(out_dim * in_dim * 8) redundant computation by computing
//      h[8] bottleneck once in __shared__ memory cooperatively across the block.
//   2. Corrects memory stride ordering to match standard row-major PyTorch parameters.
//   3. Adds batch dimension support [B, in_dim] -> [B, out_dim].
// ==============================================================================
__global__ void fused_34expert_rank8_kernel(
    const float* __restrict__ X,          // [B, in_dim]
    const float* __restrict__ A_all,      // [34, in_dim, 8]
    const float* __restrict__ B_all,      // [34, 8, out_dim]
    const float* __restrict__ pull_gate,  // [34]
    float*       __restrict__ Out,        // [B, out_dim]
    int                       batch_size,
    int                       in_dim,
    int                       out_dim
) {
    int b = blockIdx.y;
    if (b >= batch_size) return;

    int o = blockIdx.x * blockDim.x + threadIdx.x;

    const float* x_ptr = X + b * in_dim;
    float* out_ptr = Out + b * out_dim;

    __shared__ float s_h[NUM_EXPERTS][LORA_RANK];

    // Cooperative computation of intermediate bottlenecks s_h[34][8]
    int tid = threadIdx.x;
    int total_threads = blockDim.x;

    for (int idx = tid; idx < NUM_EXPERTS * LORA_RANK; idx += total_threads) {
        int e = idx / LORA_RANK;
        int r = idx % LORA_RANK;

        float p = pull_gate[e];
        if (p < 1e-6f) {
            s_h[e][r] = 0.0f;
            continue;
        }

        const float* A_e = A_all + e * (in_dim * LORA_RANK);
        float sum = 0.0f;
        for (int k = 0; k < in_dim; ++k) {
            sum += x_ptr[k] * A_e[k * LORA_RANK + r];
        }
        s_h[e][r] = sum * p; // Pre-weight intermediate bottleneck by expert pull gate
    }

    __syncthreads();

    if (o >= out_dim) return;

    float total_out = 0.0f;

    // Output projection: each thread accumulates across 34 experts for output channel o
    for (int e = 0; e < NUM_EXPERTS; ++e) {
        if (pull_gate[e] < 1e-6f) continue;

        const float* B_e = B_all + e * (LORA_RANK * out_dim);
        float exp_contrib = 0.0f;

        #pragma unroll
        for (int r = 0; r < LORA_RANK; ++r) {
            exp_contrib += s_h[e][r] * B_e[r * out_dim + o];
        }

        total_out += exp_contrib;
    }

    out_ptr[o] = total_out;
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
    int          batch_size,
    int          in_dim,
    int          out_dim,
    cudaStream_t stream
) {
    int threads = 128;
    dim3 block(threads);
    dim3 grid((out_dim + threads - 1) / threads, batch_size);

    fused_34expert_rank8_kernel<<<grid, block, 0, stream>>>(
        X, A_all, B_all, pull_gate, Out, batch_size, in_dim, out_dim
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
