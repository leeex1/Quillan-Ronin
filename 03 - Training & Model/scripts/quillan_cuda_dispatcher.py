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
_orig_linear_forward = None
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
    """Pre-quantized int8 weights + row scales, cached per module.
    Weights are static in eval, so quantize/transfer ONCE (this was the 16ms/call
    overhead killing the dispatch path — kernel itself is 0.1ms).

    BitLinear modules: ternary-quantize first (matches model's STE semantics).
    Plain nn.Linear modules: quantize raw fp32 per-row to int8 (matches F.linear
    up to int8 error — same as BitLinear's own STE error budget)."""
    key = id(module)
    ver = module.weight._version
    hit = _weight_cache.get(key)
    if hit is not None and hit[4] == ver:
        return hit  # (w_q_cpu, w_s_cpu, w_q_gpu, w_s_gpu, ver)
    w_cpu = module.weight.detach().cpu()
    try:
        from quillan_v5_4_oni import BitLinear, _weight_quant

        is_bit = isinstance(module, BitLinear)
    except Exception:
        is_bit = False
    w = _weight_quant(w_cpu) if is_bit else w_cpu
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

        # Dispatch policy (measured on GTX 1050, Windows WDDM):
        # - Prefill (M >= 64): GEMM v2 wins for large GEMMs (amortizes PCIe dtrip).
        # - Decode (M < 64): per-dispatch round-trip (launch+sync from idle) is
        #   ~12-40ms on WDDM (isolated back-to-back benches hide this!). Only
        #   dispatch when CPU time dwarfs it: lm_head [50262x1024] 22ms -> ~2ms.
        #   Threshold: decode needs out*in >= 16M; prefill needs >= 1M.
        # (c_fc 4M: CPU 0.8ms < overhead -> CPU. prism 1M: CPU -> CPU.)
        n_elems = module.out_features * module.in_features
        if M < 64:
            if n_elems < 16_000_000:
                return None
        else:
            if n_elems < 1_000_000:
                return None

        if M < 64:
            # ---- Decode path: GEMV (big modules only, bandwidth-bound) ----
            x_q, x_s = _quant_row_absmax(x_2d)  # [M,in], [M] on CPU
            if M == 1:
                y = _gemv_gpu(
                    w_q_gpu, x_q.squeeze(0).cuda(), w_s_gpu, float(x_s.item())
                )
                y = y.cpu().reshape(1, -1)
            else:
                # M in [2,64): GEMM v2 handles it (or GEMV per-row loop).
                # Use GEMM v2 — correct for all M, fast enough at M>=2.
                y = _gemm_v2(x_q.cuda(), w_q_gpu, x_s.float().cuda(), w_s_gpu)
                y = y.cpu().reshape(M, -1)
        else:
            # ---- Prefill path: GEMM v2 ----
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
    """Monkey-patch BitLinear AND large nn.Linear modules to use the DLL dispatcher.

    BitLinear: STE path (ternary weights, per-call weight quant in original).
    nn.Linear (large only, out*in >= 1M): fp32 weights quantized per-row to
    int8 once + cached. Big targets: lm_head [50262,1024], FFN c_fc/c_proj,
    c_attn/c_proj QKV — previously 60%+ of FLOPs ran pure CPU F.linear.
    Small modules (routers, classifiers) stay on CPU: PCIe overhead > gain.
    """
    if not _load():
        print("[DISPATCH] inactive — model runs original code")
        return False
    import torch.nn as nn2
    from quillan_v5_4_oni import BitLinear, _weight_quant, _quantize_act

    global _orig_forward, _orig_linear_forward
    _orig_forward = BitLinear.forward

    def patched_bit(self, x):
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

    BitLinear.forward = patched_bit

    # --- Large nn.Linear modules: same dispatch, fp32->int8 per-row quant ---
    def patched_linear(self, x):
        out = dispatch_bitlinear(self, x)
        if out is not None:
            return out
        xc = x.detach().cpu()
        y = F.linear(
            xc,
            self.weight.detach().cpu(),
            self.bias.detach().cpu() if self.bias is not None else None,
        )
        return y.to(x.device) if x.is_cuda else y

    # Only patch nn.Linear instances whose weights exceed 1M elements.
    # torch.nn.Linear patching is class-level; guard with size check inside.
    _orig_linear_forward = nn2.Linear.forward

    def patched_lin(self, x):
        if self.weight.numel() >= _LINEAR_MIN_ELEMS:
            out = dispatch_bitlinear(self, x)
            if out is not None:
                return out
        return _orig_linear_forward(self, x)

    nn2.Linear.forward = patched_lin
    n_bit = sum(1 for m in model.modules() if isinstance(m, BitLinear))
    n_lin = sum(
        1
        for m in model.modules()
        if type(m) is nn2.Linear and m.weight.numel() >= _LINEAR_MIN_ELEMS
    )
    print(
        f"[DISPATCH] installed on {n_bit} BitLinear + {n_lin} large nn.Linear "
        f"(>={_LINEAR_MIN_ELEMS // 1000000}M weights): DLL GEMV/GEMM-v2 + CPU fallback",
        flush=True,
    )
    return True


_LINEAR_MIN_ELEMS = 1_000_000  # only dispatch nn.Linear with >= 1M weights


def uninstall():
    from quillan_v5_4_oni import BitLinear
    import torch.nn as nn2

    if _orig_forward is not None:
        BitLinear.forward = _orig_forward
    if _orig_linear_forward is not None:
        nn2.Linear.forward = _orig_linear_forward
    print("[DISPATCH] uninstalled", flush=True)


if __name__ == "__main__":
    print("Dispatcher ready. Import this module from your inference script.")
