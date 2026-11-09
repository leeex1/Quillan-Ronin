#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""quillan_sm61_gemm_sk.py — binding for split-K INT8 GEMM (skinny M, persistent grid).

For M < 64 (decode/prefill-small) where the 128-tall tile wastes 75-99% of threads.
Persistent grid = 5 CTAs (one per SM on GP107); K split across CTAs; workspace
partial reduction by the split-0 worker per tile.
"""

from __future__ import annotations
import ctypes
from pathlib import Path
import torch

HERE = Path(__file__).resolve().parent
PTX_PATH = HERE / "sm61_gemm_sk_sm61.ptx"

_nvcuda = None
_cu_mod = ctypes.c_void_p(None)
_cu_fn = ctypes.c_void_p(None)


def _init_driver():
    global _nvcuda, _cu_mod, _cu_fn
    if _cu_fn.value is not None:
        return
    if not PTX_PATH.exists():
        import subprocess, sys

        subprocess.run([sys.executable, str(HERE / "ptx_build_sk.py")], check=True)
    _nvcuda = ctypes.WinDLL("nvcuda.dll")
    _nvcuda.cuModuleLoadDataEx.restype = int
    _nvcuda.cuModuleLoadDataEx.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_char_p,
        ctypes.c_uint,
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    _nvcuda.cuModuleGetFunction.restype = int
    _nvcuda.cuModuleGetFunction.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_void_p,
        ctypes.c_char_p,
    ]
    _nvcuda.cuLaunchKernel.restype = int
    _nvcuda.cuLaunchKernel.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint,
        ctypes.c_uint,
        ctypes.c_uint,
        ctypes.c_uint,
        ctypes.c_uint,
        ctypes.c_uint,
        ctypes.c_uint,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_void_p,
    ]
    torch.cuda.init()
    _ = torch.empty((1,), device="cuda")
    ptx_bytes = PTX_PATH.read_bytes() + b"\x00"
    rc = _nvcuda.cuModuleLoadDataEx(ctypes.byref(_cu_mod), ptx_bytes, 0, None, None)
    if rc != 0:
        raise RuntimeError(f"cuModuleLoadDataEx rc={rc}")
    rc = _nvcuda.cuModuleGetFunction(ctypes.byref(_cu_fn), _cu_mod, b"qgemm_sk_kernel")
    if rc != 0:
        raise RuntimeError(f"cuModuleGetFunction rc={rc}")


def qgemm_forward_sk(x_int8, w_int8, x_scale, w_scale, nsplits=5) -> torch.Tensor:
    """Y[M,N] fp32 = dequant(X_int8[M,K] @ W_int8[N,K]^T), split-K persistent."""
    if not (x_int8.is_cuda and w_int8.is_cuda):
        raise ValueError("inputs must be CUDA")
    x_int8 = x_int8.contiguous()
    w_int8 = w_int8.contiguous()
    x_scale = x_scale.contiguous()
    w_scale = w_scale.contiguous()
    M, K = x_int8.shape
    N, _ = w_int8.shape
    _init_driver()
    y = torch.empty((M, N), dtype=torch.float32, device=x_int8.device)
    ws = torch.empty((nsplits, M, N), dtype=torch.float32, device=x_int8.device)
    # NOTE: torch.zeros(..., device=cuda) launches a fill kernel (absent on sm_61).
    # Create zeros on CPU and memcpy instead.
    cntrs = torch.zeros(((M + 31) // 32) * ((N + 63) // 64), dtype=torch.int32).cuda()
    c_x = ctypes.c_uint64(x_int8.data_ptr())
    c_w = ctypes.c_uint64(w_int8.data_ptr())
    c_xs = ctypes.c_uint64(x_scale.data_ptr())
    c_ws = ctypes.c_uint64(w_scale.data_ptr())
    c_y = ctypes.c_uint64(y.data_ptr())
    c_wsx = ctypes.c_uint64(ws.data_ptr())
    c_cn = ctypes.c_uint64(cntrs.data_ptr())
    c_M = ctypes.c_int(M)
    c_N = ctypes.c_int(N)
    c_K = ctypes.c_int(K)
    c_ns = ctypes.c_int(nsplits)
    kernel_params = (ctypes.c_void_p * 11)(
        ctypes.cast(ctypes.byref(c_x), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(c_w), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(c_xs), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(c_ws), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(c_y), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(c_wsx), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(c_cn), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(c_M), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(c_N), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(c_K), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(c_ns), ctypes.c_void_p),
    )
    stream_ptr = ctypes.c_void_p(torch.cuda.current_stream(x_int8.device).cuda_stream)
    rc = _nvcuda.cuLaunchKernel(
        _cu_fn, 5, 1, 1, 16, 8, 1, 0, stream_ptr, kernel_params, None
    )
    if rc != 0:
        raise RuntimeError(f"cuLaunchKernel(sk) rc={rc}")
    return y
