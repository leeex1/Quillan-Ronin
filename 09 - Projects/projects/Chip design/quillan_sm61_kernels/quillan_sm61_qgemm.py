#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
quillan_sm61_qgemm.py — Version-agnostic CUDA Driver API (`nvcuda.dll`) binding
for the NVRTC-compiled Pascal sm_61 DP4A quantized INT8 GEMM kernel (`sm61_qgemm_sm61.ptx`).

Works across Python 3.11, 3.12, and 3.14 without CPython ABI lock by loading
`sm61_qgemm_sm61.ptx` via `cuModuleLoadDataEx` and launching `qgemm_i8_dp4a_kernel`
via `cuLaunchKernel` with block(16, 16, 1) = 256 threads/block (8x4 register blocking).
"""
from __future__ import annotations

import ctypes
from pathlib import Path
import torch

HERE = Path(__file__).resolve().parent
PTX_PATH = HERE / "sm61_qgemm_sm61.ptx"

_nvcuda = None
_cu_mod = ctypes.c_void_p(None)
_cu_fn = ctypes.c_void_p(None)


def _init_driver():
    global _nvcuda, _cu_mod, _cu_fn
    if _cu_fn.value is not None:
        return
    if not PTX_PATH.exists():
        import subprocess
        import sys
        subprocess.run([sys.executable, str(HERE / "ptx_build.py")], check=True)

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
        ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,
        ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,
        ctypes.c_uint,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_void_p,
    ]

    # Ensure primary CUDA context is active via PyTorch
    torch.cuda.init()
    _ = torch.empty((1,), device="cuda")

    ptx_bytes = PTX_PATH.read_bytes() + b"\x00"
    rc = _nvcuda.cuModuleLoadDataEx(ctypes.byref(_cu_mod), ptx_bytes, 0, None, None)
    if rc != 0:
        raise RuntimeError(f"cuModuleLoadDataEx failed with CUresult={rc}")

    rc = _nvcuda.cuModuleGetFunction(ctypes.byref(_cu_fn), _cu_mod, b"qgemm_i8_dp4a_kernel")
    if rc != 0:
        raise RuntimeError(f"cuModuleGetFunction failed with CUresult={rc}")


def qgemm_forward(
    x_int8: torch.Tensor,
    w_int8: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
) -> torch.Tensor:
    """
    Executes Pascal sm_61 DP4A quantized INT8 GEMM -> FP16 on CUDA:
      Y[M, N] = dequant( X_int8[M, K] @ W_int8[N, K]^T )
    """
    if not (x_int8.is_cuda and w_int8.is_cuda and x_scale.is_cuda and w_scale.is_cuda):
        raise ValueError("All inputs must be CUDA tensors")
    if x_int8.dtype != torch.int8 or w_int8.dtype != torch.int8:
        raise ValueError("x_int8 and w_int8 must have dtype torch.int8")
    if x_scale.dtype != torch.float32 or w_scale.dtype != torch.float32:
        raise ValueError("x_scale and w_scale must have dtype torch.float32")

    x_int8 = x_int8.contiguous()
    w_int8 = w_int8.contiguous()
    x_scale = x_scale.contiguous()
    w_scale = w_scale.contiguous()

    M, K = x_int8.shape
    N, Kw = w_int8.shape
    if K != Kw:
        raise ValueError(f"K mismatch: x is [{M}, {K}], w is [{N}, {Kw}]")

    _init_driver()

    y = torch.empty((M, N), dtype=torch.float16, device=x_int8.device)

    grid_x = (N + 63) // 64
    grid_y = (M + 127) // 128

    c_x = ctypes.c_uint64(x_int8.data_ptr())
    c_w = ctypes.c_uint64(w_int8.data_ptr())
    c_xs = ctypes.c_uint64(x_scale.data_ptr())
    c_ws = ctypes.c_uint64(w_scale.data_ptr())
    c_y = ctypes.c_uint64(y.data_ptr())
    c_M = ctypes.c_int(M)
    c_N = ctypes.c_int(N)
    c_K = ctypes.c_int(K)

    kernel_params = (ctypes.c_void_p * 8)(
        ctypes.cast(ctypes.byref(c_x), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(c_w), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(c_xs), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(c_ws), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(c_y), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(c_M), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(c_N), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(c_K), ctypes.c_void_p),
    )

    stream_ptr = ctypes.c_void_p(torch.cuda.current_stream(x_int8.device).cuda_stream)
    rc = _nvcuda.cuLaunchKernel(
        _cu_fn,
        grid_x, grid_y, 1,
        16, 16, 1,
        0,
        stream_ptr,
        kernel_params,
        None,
    )
    if rc != 0:
        raise RuntimeError(f"cuLaunchKernel failed with CUresult={rc}")

    return y

