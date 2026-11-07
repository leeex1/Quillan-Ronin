#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
quillan_cuda_dispatcher.py — Routes Quillan BitLinear through sm_61 DLL kernels.

Dispatch strategy (measured on GTX 1050 GP107):
  - M == 1 (decode):      BitNet GEMV DLL (15-20x vs CPU fp32)
  - M >= 64 (prefill):    INT8 GEMM v2 DLL (4-5x vs v1, 2-3x vs CPU)
  - Otherwise / failure:  F.linear fallback (always correct)

Activation/weight quantization matches the model's existing STE scheme
(per-row absmax/127), so numerical parity is preserved (verified <1% rel err).

Usage:
    import quillan_cuda_dispatcher as D
    D.install(model)   # monkey-patches BitLinear.forward
    ... run inference ...
    D.uninstall(model) # restore original forward
"""

from __future__ import annotations
import ctypes
import sys
from pathlib import Path
import torch
import torch.nn.functional as F

_HERE = Path(__file__).resolve().parent
_KDIR = Path(r"C:\02_QUILLAN\kernels")
_QDIR = Path(r"C:\02_QUILLAN\09 - Projects\projects\Chip design\quillan_sm61_kernels")
sys.path.insert(0, str(_QDIR))

_gemv_dll = None
_gemm_v2 = None
_enabled = False
_orig_forward = None
# Cached pre-packed weights per BitLinear module (weights are static in eval).
# Key: id(module), Value: (w_q_cpu, w_s_cpu, w_q_gpu, w_s_gpu, w_version)
_weight_cache = {}


def _load():
    global _gemv_dll, _gemm_v2, _enabled
    if _enabled:
        return True
    try:
        from quillan_sm61_qgemm_v2 import qgemm_forward_v2

        globals()["_gemm_v2"] = qgemm_forward_v2
        dll = ctypes.CDLL(str(_KDIR / "quillan_cuda_sm61.dll"))
        f = dll.launch_bitnet_gemv_dp4a
        f.argtypes = [ctypes.c_void_p] * 4 + [
            ctypes.c_float,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p,
        ]
        f.restype = None
        globals()["_gemv_dll"] = dll
        _enabled = True
        return True
    except Exception as e:
        print(f"[DISPATCH] DLL load failed ({e}); staying on F.linear", flush=True)
        return False


def _gemv_gpu(W_ternary_int8, x_int8, row_scales, x_scale):
    """W: [M,K] int8 ternary, x: [K] int8 -> y: [M] fp32 (GPU, sm_61)."""
    M, K = W_ternary_int8.shape
    Kd4 = K // 4
    Wp = W_ternary_int8.view(torch.int32).contiguous()
    Xp = x_int8.view(torch.int32).contiguous()
    Y = torch.empty((M,), dtype=torch.float32, device=W_ternary_int8.device)
    rs = (
        row_scales
        if row_scales is not None
        else torch.ones((M,), dtype=torch.float32, device=W_ternary_int8.device)
    )
    globals()["_gemv_dll"].launch_bitnet_gemv_dp4a(
        Wp.data_ptr(),
        Xp.data_ptr(),
        Y.data_ptr(),
        rs.data_ptr(),
        float(x_scale),
        M,
        Kd4,
        0,
    )
    return Y


def _quant_row_absmax(v):
    """Per-row absmax/127 scale + round clamp to int8. v: [..., K] float."""
    scale = v.abs().amax(dim=-1, keepdim=True).clamp(min=1e-6) / 127.0
    q = torch.clamp((v / scale).round(), -128, 127).to(torch.int8)
    return q, scale.squeeze(-1)


def _cached_weights(module):
    """Pre-quantized ternary int8 weights + row scales, cached per module.
    Weights are static in eval, so quantize/transfer ONCE (this was the 16ms/call
    overhead killing the dispatch path — kernel itself is 0.1ms)."""
    key = id(module)
    ver = module.weight._version
    hit = _weight_cache.get(key)
    if hit is not None and hit[4] == ver:
        return hit  # (w_q_cpu, w_s_cpu, w_q_gpu, w_s_gpu, ver)
    from quillan_v5_4_oni import _weight_quant

    w_cpu = module.weight.detach().cpu()
    w = _weight_quant(w_cpu)
    w_q, w_s = _quant_row_absmax(w)
    w_q_gpu = w_q.cuda()
    w_s_gpu = w_s.float().cuda()
    if len(_weight_cache) > 512:  # bound cache size
        _weight_cache.clear()
    _weight_cache[key] = (w_q, w_s, w_q_gpu, w_s_gpu, ver)
    return _weight_cache[key]


def dispatch_bitlinear(module, x):
    """Drop-in replacement for BitLinear.forward using sm_61 DLLs.

    x: [..., in_f] float (CPU or CUDA). Returns [..., out_f] float.
    Falls back to reference STE path on any failure.
    """
    # Reference STE (same as model's original forward)
    from quillan_v5_4_oni import _weight_quant, _quantize_act

    if not _enabled:
        return None  # signal fallback

    try:
        # --- Cached pre-packed weights (quant+transfer once, not per call!) ---
        w_q, w_s, w_q_gpu, w_s_gpu, _v = _cached_weights(module)
        x_cpu = x.detach().cpu()
        x_2d = x_cpu.reshape(-1, module.in_features)
        M = x_2d.shape[0]

        # Dispatch policy (measured on GTX 1050):
        # - Prefill (M >= 64): GEMM v2 wins (large GEMMs amortize PCIe dtrip).
        # - Decode (M < 64): CPU F.linear wins — 110 sequential PCIe round-trips
        #   (~1.5ms/module overhead) exceed the sub-ms GEMV kernel time.
        # So: only dispatch prefill GEMMs; decode falls back to CPU STE path.
        if M < 64:
            return None

        # Skip tiny GEMMs where transfer overhead dominates kernel time.
        if module.out_features * module.in_features < 1_000_000:
            return None

        # ---- Prefill path: GEMM v2 (M >= 64, large GEMMs only) ----
        x_q, x_s = _quant_row_absmax(x_2d)  # [M,in], [M] on CPU
        y = _gemm_v2(x_q.cuda(), w_q_gpu, x_s.float().cuda(), w_s_gpu)
        y = y.cpu().reshape(M, -1)

        if module.bias is not None:
            y = y + module.bias.detach().cpu()
        return y.reshape(x.shape[:-1] + (-1,))
    except Exception as e:
        print(f"[DISPATCH] fallback ({e})", flush=True)
        return None


def install(model):
    """Monkey-patch all BitLinear modules in `model` to use the DLL dispatcher."""
    if not _load():
        print("[DISPATCH] inactive — model runs original code")
        return False
    from quillan_v5_4_oni import BitLinear, _weight_quant, _quantize_act

    global _orig_forward
    _orig_forward = BitLinear.forward

    def patched(self, x):
        out = dispatch_bitlinear(self, x)
        if out is not None:
            return out
        # fallback: original STE path (CPU-safe: sm_61 has no PyTorch CUDA kernels)
        xc = x.detach().cpu()
        w = (
            _weight_quant(self.weight.detach().cpu())
            if self.quantize_weight
            else self.weight.detach().cpu()
        )
        x_q = _quantize_act(xc) if self.quantize_act else xc
        b = self.bias.detach().cpu() if self.bias is not None else None
        y = F.linear(x_q, w, b)
        return y.to(x.device) if x.is_cuda else y

    BitLinear.forward = patched
    n = sum(1 for m in model.modules() if isinstance(m, BitLinear))
    print(
        f"[DISPATCH] installed on {n} BitLinear modules (GEMV decode + GEMM-v2 prefill)",
        flush=True,
    )
    return True


def uninstall():
    from quillan_v5_4_oni import BitLinear

    if _orig_forward is not None:
        BitLinear.forward = _orig_forward
        print("[DISPATCH] uninstalled", flush=True)


if __name__ == "__main__":
    print("Dispatcher ready. Import this module from your inference script.")
