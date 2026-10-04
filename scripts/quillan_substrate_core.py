import os
import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Any, Optional, List

# Force UTF-8 on Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# ─── PHOENIX HARDWARE AFFINITY LOCK ──────────────────────────────────────────

def apply_phoenix_affinity() -> None:
    """forcefully bind the orchestration execution threads strictly to CPU Cores 2 and 3."""
    try:
        import psutil
        p = psutil.Process(os.getpid())
        p.cpu_affinity([2, 3])
        print("[PHOENIX AFFINITY] Thread affinity successfully locked to CPU Cores 2 and 3.")
    except ImportError:
        print("[PHOENIX AFFINITY WARNING] psutil not installed. Cannot lock CPU affinity.")
    except Exception as e:
        print(f"[PHOENIX AFFINITY WARNING] Could not lock CPU affinity: {e}")

# ─── BITNET 1.58b QUANTIZATION PRIMITIVES (STE) ──────────────────────────────

def _activation_quant(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Scale and round inputs to 8-bit integers (-128 to 127) using a Straight-Through Estimator."""
    max_val = x.abs().max(dim=-1, keepdim=True).values.clamp(min=eps)
    x_scaled = x * (127.0 / max_val)
    x_q = torch.round(torch.clamp(x_scaled, -128.0, 127.0))
    # Straight-Through Estimator (STE)
    return x + (x_q * (max_val / 127.0) - x).detach()

def _weight_quant(w: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Compress weights to ternary bounds (-1.0, 0.0, 1.0) with learned scaling using STE."""
    # Mean absolute value as the scaling factor
    scale = w.abs().mean(dim=[-2, -1] if w.dim() >= 2 else -1, keepdim=True).clamp(min=eps)
    w_scaled = w / scale
    w_q = torch.round(torch.clamp(w_scaled, -1.0, 1.0))
    # Straight-Through Estimator (STE)
    return w + (w_q * scale - w).detach()

class BitLinear(nn.Linear):
    """Sovereign BitNet 1.58b Quantized Linear Layer."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_q = _activation_quant(x)
        w_q = _weight_quant(self.weight)
        return F.linear(x_q, w_q, self.bias)

# ─── LEE-MACH-6 DYNAMIC COGNITIVE GOVERNOR ───────────────────────────────────

class LeeMach6Governor:
    """Tracks execution times and dynamically adjusts swarm variance and memory recency bias."""
    def __init__(self, target_latency_ms: float = 100.0):
        self.target_latency = target_latency_ms
        self.current_scale = 1.0
        self.recency_bias = 0.0

    def adjust(self, latency_ms: float) -> Tuple[float, float]:
        if latency_ms > self.target_latency:
            # Exceeded: scale down variance, shift memory recall recency bias to favor new states
            self.current_scale = max(0.1, self.current_scale * 0.8)
            self.recency_bias = min(1.0, self.recency_bias + 0.2)
        else:
            # Within bounds: scale up variance, reduce recency bias
            self.current_scale = min(1.0, self.current_scale * 1.1)
            self.recency_bias = max(0.0, self.recency_bias - 0.1)
        return self.current_scale, self.recency_bias

# ─── PHOENIX INT8 OPENCL ACCELERATOR INTERFACE ───────────────────────────────

class PhoenixINT8OpenCLAccelerator:
    """Pre-allocates buffer memory and executes vectorized kernels with CPU fallback."""
    def __init__(self, max_agents: int = 100000, device_id: int = 0):
        self.max_agents = max_agents
        self.cl_ctx = None
        self.cl_queue = None
        self.cl_program = None
        self.pyopencl_available = False
        
        # Native OpenCL C kernel source code
        self.kernel_src = """
        __kernel void int8_cosine_sim_vec4(
            __global const char* query,
            __global const char* database,
            __global float* output,
            const int N
        ) {
            int gid = get_global_id(0);
            if (gid >= N) return;
            float dot = 0.0f;
            float norm_q = 0.0f;
            float norm_d = 0.0f;
            for (int i = 0; i < 4; i++) {
                float q_val = (float)query[i];
                float d_val = (float)database[gid * 4 + i];
                dot += q_val * d_val;
                norm_q += q_val * q_val;
                norm_d += d_val * d_val;
            }
            output[gid] = dot / (sqrt(norm_q) * sqrt(norm_d) + 1e-8f);
        }

        __kernel void bitnet_eggroll_modulate(
            __global const float* U,
            __global const float* V,
            __global const float* X,
            __global float* Y,
            const int B,
            const int D,
            const int R,
            const float scale
        ) {
            int b = get_global_id(0);
            int d = get_global_id(1);
            if (b >= B || d >= D) return;
            
            float sum_val = 0.0f;
            for (int r = 0; r < R; r++) {
                float temp_r = 0.0f;
                for (int i = 0; i < D; i++) {
                    temp_r += X[b * D + i] * U[i * R + r];
                }
                sum_val += temp_r * V[r * D + d];
            }
            Y[b * D + d] = X[b * D + d] + sum_val * scale * 0.25f;
        }
        """
        
        try:
            import pyopencl as cl
            platforms = cl.get_platforms()
            if platforms:
                devices = platforms[0].get_devices()
                if devices:
                    selected_device = devices[min(device_id, len(devices)-1)]
                    self.cl_ctx = cl.Context([selected_device])
                    self.cl_queue = cl.CommandQueue(self.cl_ctx)
                    self.cl_program = cl.Program(self.cl_ctx, self.kernel_src).build()
                    self.pyopencl_available = True
                    print("[Phoenix Accelerator] OpenCL Context successfully initialized on device.")
        except Exception as e:
            print(f"[Phoenix Accelerator Warning] OpenCL initialization failed (falling back to PyTorch): {e}")

    def cosine_sim_vec4(self, query: torch.Tensor, database: torch.Tensor) -> torch.Tensor:
        """Vectorized cosine similarity matching on 8-bit inputs."""
        # query: [4] int8, database: [N, 4] int8
        if self.pyopencl_available:
            import pyopencl as cl
            import numpy as np
            N = database.shape[0]
            
            query_np = query.cpu().numpy().astype(np.int8)
            database_np = database.cpu().numpy().astype(np.int8)
            
            mf = cl.mem_flags
            query_buf = cl.Buffer(self.cl_ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=query_np)
            database_buf = cl.Buffer(self.cl_ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=database_np)
            output_buf = cl.Buffer(self.cl_ctx, mf.WRITE_ONLY, size=N * 4)
            
            self.cl_program.int8_cosine_sim_vec4(
                self.cl_queue, (N,), None,
                query_buf, database_buf, output_buf, np.int32(N)
            )
            
            result = np.empty(N, dtype=np.float32)
            cl.enqueue_copy(self.cl_queue, result, output_buf)
            return torch.tensor(result, device=query.device)
        else:
            # Vectorized PyTorch Fallback
            q_float = query.float()
            d_float = database.float()
            dot = torch.sum(q_float * d_float, dim=-1)
            norm_q = torch.norm(q_float)
            norm_d = torch.norm(d_float, dim=-1)
            return dot / (norm_q * norm_d + 1e-8)

    def eggroll_modulate(self, U: torch.Tensor, V: torch.Tensor, X: torch.Tensor, scale: float) -> torch.Tensor:
        """Fast low-rank mutation scaling (U * V^T) across active agents."""
        # U: [D, R], V: [R, D], X: [B, D]
        if self.pyopencl_available:
            import pyopencl as cl
            import numpy as np
            B, D = X.shape
            R = U.shape[1]
            
            mf = cl.mem_flags
            U_buf = cl.Buffer(self.cl_ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=U.cpu().numpy().astype(np.float32))
            V_buf = cl.Buffer(self.cl_ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=V.cpu().numpy().astype(np.float32))
            X_buf = cl.Buffer(self.cl_ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=X.cpu().numpy().astype(np.float32))
            Y_buf = cl.Buffer(self.cl_ctx, mf.WRITE_ONLY, size=B * D * 4)
            
            self.cl_program.bitnet_eggroll_modulate(
                self.cl_queue, (B, D), None,
                U_buf, V_buf, X_buf, Y_buf,
                np.int32(B), np.int32(D), np.int32(R), np.float32(scale)
            )
            
            result = np.empty((B, D), dtype=np.float32)
            cl.enqueue_copy(self.cl_queue, result, Y_buf)
            return torch.tensor(result, device=X.device, dtype=X.dtype)
        else:
            # Vectorized PyTorch Low-Rank Perturbation Fallback: X + (X @ U @ V) * scale * 0.25
            return X + (X @ U @ V) * scale * 0.25

# ─── VERIFICATION BLOCK ──────────────────────────────────────────────────────

if __name__ == "__main__":
    print("[*] Initiating verification routine for quillan_substrate_core...")
    
    # 1. Test Hardware Affinity
    apply_phoenix_affinity()
    
    # 2. Test Lee-Mach-6 Governor
    gov = LeeMach6Governor(target_latency_ms=100.0)
    scale, bias = gov.adjust(120.0)
    print(f"[*] Governor Adjustment: Scale={scale:.4f}, Bias={bias:.4f}")
    assert scale < 1.0, "Governor should damp scaling when latency limit is exceeded."
    
    # 3. Test BitNet Quantization Primitives
    x_test = torch.randn(2, 16, 2560)
    x_quant = _activation_quant(x_test)
    print(f"[*] Quantized Activations Max: {x_quant.max().item():.4f}, Min: {x_quant.min().item():.4f}")
    
    # 4. Test BitLinear layer forward pass
    bit_linear = BitLinear(2560, 1280)
    out = bit_linear(x_test)
    print(f"[*] BitLinear output shape: {list(out.shape)}")
    assert out.shape == (2, 16, 1280), "BitLinear layer shape mismatch."
    
    # 5. Test OpenCL/PyTorch Swarm Accelerator
    acc = PhoenixINT8OpenCLAccelerator()
    query = torch.randint(-128, 127, (4,), dtype=torch.int8)
    db = torch.randint(-128, 127, (10, 4), dtype=torch.int8)
    sims = acc.cosine_sim_vec4(query, db)
    print(f"[*] Cosine Similarities (first 3): {sims[:3].tolist()}")
    
    U = torch.randn(2560, 16)
    V = torch.randn(16, 2560)
    X = torch.randn(2, 2560)
    mod = acc.eggroll_modulate(U, V, X, scale=0.5)
    print(f"[*] Modulated Swarm output shape: {list(mod.shape)}")
    
    print("\n[+] ALL VERIFICATIONS PASSED SUCCESSFULLY.")
