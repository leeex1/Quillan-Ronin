# Quillan CUDA Kernel Upgrades — GTX 1050 Ti (sm_61) Inference Speed
## Date: 2026-11-06 | Author: Quillan (C10-CODEWEAVER + C13-WARDEN + C26-TECHNE)

## Context
PyTorch 2.12 + CUDA 13.0 dropped sm_61 support entirely. The GTX 1050 **cannot
run any PyTorch CUDA kernel** ("no kernel image available"). The custom sm_61
DLLs are the ONLY GPU compute path on this machine.

## Measured Baseline (before upgrades)
| Path | Result |
|------|--------|
| Model decode (CPU, unpatched) | 1187.6 ms/tok (0.84 tok/s) |
| Model prefill T=32 (CPU, unpatched) | 1522.4 ms |
| GEMM M=1 (GPU v1) | 0.462 ms (5.9x SLOWER than CPU!) |
| GEMV decode (GPU v1) | 15-20x vs CPU, but DEAD CODE (never called) |

## Upgrades Landed

### 1. Python quick wins (quillan_v5_4_oni.py)
- `compute_aux=False` at both decode call sites (generate prefill + loop):
  skips ~100 paper aux modules per forward (agent_evolution, mamba, moR, MoD...)
- `BatchedCouncilCompute` eval short-circuit: swarm variance + 3-step world
  recurrence computed then discarded in eval → return `delta` immediately.

**Measured: prefill 1.77x (1522→860ms), decode 1.73x (1188→685ms/tok).**
Numerics unchanged (aux feeds aux outputs only; swarm path discarded in eval).

### 2. GEMM v2 kernel (sm61_qgemm_v2.cu) — NEW
File: `09 - Projects/projects/Chip design/quillan_sm61_kernels/sm61_qgemm_v2.cu`
- `int4` (16B) vectorized global→smem loads (was scalar byte loads)
- Register-prefetch double buffering (2 smem stages, overlap load/compute)
- FP32 epilogue (v1 wrote FP16 — Pascal FP16 is ~1/64 rate!)
- Bank-conflict padded smem (48B stride, 16B aligned)
- `__launch_bounds__(256)` occupancy hint

**Measured A/B vs v1 (parity <0.001):**
| Shape | v1 | v2 | Speedup |
|-------|----|----|---------|
| 32×1024×1024 | 0.544 | 0.168 | **3.23x** |
| 128×1024×1024 | 0.786 | 0.158 | **4.98x** |
| 128×4096×4096 | 9.503 | 2.078 | **4.57x** |
| 512×1024×4096 | 7.559 | 1.839 | **4.11x** |
| 512×4096×4096 | 34.54 | 7.14 | **4.84x** |
| 1024×4096×4096 | 75.57 | 15.29 | **4.94x** |

Build: `ptx_build_v2.py` (NVRTC, compute_61). Binding: `quillan_sm61_qgemm_v2.py`.

### 3. GEMV — v2 ATTEMPTED, REJECTED (measured slower)
File: `kernels/quillan_cuda_sm61_v2.cu` (kept for reference, NOT production)
- Split-K + smem X staging + unrolled loop: **0.59-1.00x vs v1** (slower!)
- Reason: GEMV is bandwidth-bound; X fits in L2; extra syncs hurt.
- **Production GEMV = v1** (`quillan_cuda_sm61.dll`), already 15-20x vs CPU.

### 4. DLL dispatcher (quillan_cuda_dispatcher.py) — NEW
Monkey-patches `BitLinear.forward`: prefill (M≥64, ≥1M elems) → GEMM v2;
decode (M<64) → CPU STE fallback (measured: 110 sequential PCIe round-trips
per token cost ~1.5ms/module, exceeding sub-ms GEMV time). Quant on CPU
(no PyTorch CUDA kernels exist on sm_61), weights pre-packed + cached per
module (key insight: per-call requant was 16ms vs 0.1ms kernel). Parity 0.0000.

**Measured at BitLinear level (1024→2048):**
- Decode M=1: 0.407ms vs 15.788ms CPU = **38.8x**
- Prefill M=128: 2.596ms vs 9.968ms CPU = **3.8x**

**Measured end-to-end (full 593M model, T=128 prefill): 0.98x — NO WIN.**
110 sequential round-trips cancel the kernel gains. The kernels are proven
fast; the integration is transfer-bound because the model must stay on CPU
(PyTorch GPU broken on sm_61). Dispatcher kept for portability (wins if the
model ever lives on GPU), but default production path = Python wins only.

**END-TO-END PRODUCTION NUMBERS (Python wins only):**
- Prefill: 1522.4 → 860.2ms (**1.77x**)
- Decode: 1187.6 → 684.6 ms/tok (**1.73x**, 0.84 → 1.46 tok/s)

## How to use
```python
import sys
sys.path.insert(0, r"C:\02_QUILLAN\03 - Training & Model\scripts")
from quillan_v5_4_oni import QuillanRoninOni, QuillanOniConfig
import quillan_cuda_dispatcher as D
model = QuillanRoninOni(QuillanOniConfig()).eval()
D.install(model)  # routes BitLinear through sm_61 DLLs
tokens = model.generate([42, 7, 1337], max_tokens=64)  # compute_aux already False
D.uninstall()
```

## Backup
Original model file: `quillan_v5_4_oni.py.bak_speedup` (toggle by copy).

## Future work (not done)
- 2-bit ternary GEMV packing (4x less weight traffic; needs repack + decode kernel)
- Persistent grid (5 CTAs) for GEMM—only helps if launch overhead dominates
- KV cache preallocation (avoid torch.cat per token)
- Dropdown: install an sm_61-capable PyTorch (≤1.13) to enable full GPU inference
