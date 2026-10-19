# -*- coding: utf-8 -*-
"""
Quillan-Ronin Native Kernel Bridge (C++ / CUDA / AVX2)
Automatically compiles and binds high-performance native kernels into PyTorch.
"""

import os
import sys
from pathlib import Path
import torch
import torch.nn.functional as F

_CSRC_DIR = Path(__file__).resolve().parent
_NATIVE_KERNEL_MODULE = None

def load_native_kernels():
    """Load or JIT-compile native C++/CUDA BitNet & Prism kernels."""
    global _NATIVE_KERNEL_MODULE
    if _NATIVE_KERNEL_MODULE is not None:
        return _NATIVE_KERNEL_MODULE

    sources = [_CSRC_DIR / "bitnet_cpu_avx2.cpp"]
    extra_cflags = ["/O2", "/openmp", "/arch:AVX2"] if sys.platform == "win32" else ["-O3", "-fopenmp", "-mavx2"]
    extra_cuda_cflags = []

    has_cuda = torch.cuda.is_available()
    if has_cuda:
        sources.append(_CSRC_DIR / "bitnet_cuda_kernel.cu")
        extra_cuda_cflags = [
            "--use_fast_math",
            "-gencode=arch=compute_61,code=sm_61", # Pascal GTX 1050
            "-gencode=arch=compute_75,code=sm_75", # Turing
            "-gencode=arch=compute_86,code=sm_86", # Ampere
            "-gencode=arch=compute_89,code=sm_89", # Ada
        ]

    try:
        from torch.utils.cpp_extension import load
        _NATIVE_KERNEL_MODULE = load(
            name="quillan_native_ops",
            sources=[str(s) for s in sources],
            extra_cflags=extra_cflags,
            extra_cuda_cflags=extra_cuda_cflags if has_cuda else None,
            verbose=False
        )
        print("[KERNELS] Native C++/AVX2 kernels compiled and loaded successfully.")
        return _NATIVE_KERNEL_MODULE
    except Exception as e:
        # Fallback to JIT PyTorch tensor operations gracefully
        return None

def native_weight_quant(w: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    mod = load_native_kernels()
    if mod is not None and not w.is_cuda:
        try:
            return mod.bitnet_weight_quant_cpu(w, eps)
        except Exception:
            pass
    # JIT fallback
    scale = 1.0 / w.abs().mean(dim=-1, keepdim=True).clamp(min=eps)
    w_scaled = w * scale
    w_q = torch.round(torch.clamp(w_scaled, -1.0, 1.0))
    return (w_scaled + (w_q - w_scaled).detach()) / scale

def native_nine_vector_prism(x: torch.Tensor, w_stacked: torch.Tensor, w_gate: torch.Tensor) -> torch.Tensor:
    mod = load_native_kernels()
    if mod is not None and not x.is_cuda:
        try:
            return mod.nine_vector_prism_forward_cpu(x, w_stacked, w_gate)
        except Exception:
            pass
    prism = torch.einsum('bld,ned->ble', x, w_stacked) / 9.0
    return F.linear(prism, w_gate)
