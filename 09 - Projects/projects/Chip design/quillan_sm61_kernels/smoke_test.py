#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
smoke_test.py — correctness check vs the fp32 reference. Run from this folder:

    python smoke_test.py

Auto-detects the compiled DP4A extension: exercises the real GPU kernel
when present, CPU fallback otherwise. Trust procedure: diff a few outputs
against torch matmul on dequantized fp32 before using this for anything real.
"""
from __future__ import annotations

import sys

import torch
import torch.nn as nn

from quantize import quantize_per_output, quantize_per_row, quantize_roundtrip_error

try:
    import quillan_sm61_qgemm

    EXT = True
except Exception as e:
    print(f"[smoke] extension not importable ({e}); GPU-kernel tests will skip.")
    EXT = False

from sm61_linear import SM61Linear


def check(name: str, cond: bool, detail: str = ""):
    print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name} {detail}")
    if not cond:
        raise SystemExit(f"smoke test failed at: {name}")


def main() -> None:
    torch.manual_seed(0)
    M, K, N = 32, 128, 64

    # 1. quantize roundtrip stays sane (int8 rounds sub-LSB elements to
    # zero by design, so score only elements large enough to be
    # representable — otherwise the metric measures zeros, not the quant).
    x = torch.randn(M, K)
    q, s = quantize_per_row(x)
    ref = q.float() * s.view(-1, 1)
    mask = x.abs() > 0.05
    rel = (ref[mask] - x[mask]).abs() / (x[mask].abs() + 1e-8)
    # int8 rounds small-in-row elements coarsely: median must be tight,
    # max only bounded loosely (small magnitudes, few LSBs).
    check("quantize roundtrip",
          rel.median().item() < 0.02 and rel.max().item() < 0.35,
          f"median_rel_err={rel.median().item():.5f} max={rel.max().item():.4f}")

    # 2. int8 GEMM numerics vs fp32 reference (CPU path).
    w = torch.randn(N, K)
    x_q, x_s = quantize_per_row(x)
    w_q, w_s = quantize_per_output(w)
    from vgpu_backend import vgpu_qgemm_cpu

    y_q = vgpu_qgemm_cpu(x_q, w_q, x_s, w_s).float()
    y_ref = (x_q.float() * x_s.view(-1, 1)) @ (w_q.float() * w_s.view(-1, 1)).T
    big = y_ref.abs() > 0.05
    rel = ((y_q[big] - y_ref[big]).abs() / (y_ref[big].abs() + 1e-8))
    check("vgpu cpu numerics", rel.max().item() < 1e-2, f"max_rel_err={rel.max().item():.6f}")

    # 3. SM61Linear vs nn.Linear fp32 reference (end-to-end module).
    lin = nn.Linear(K, N, bias=True)
    sm = SM61Linear.from_linear(lin, split_ratio=0.0 if not EXT else 0.9)
    with torch.no_grad():
        y_sm = sm(x).float()
        y_fp = lin(x.float()).float()
    # int8 is lossy: tolerance is percent-level, NOT fp32-exact.
    big = y_fp.abs() > 0.1
    rel = ((y_sm[big] - y_fp[big]).abs() / (y_fp[big].abs() + 1e-6))
    check("SM61Linear vs fp32", rel.median().item() < 0.05 and rel.max().item() < 0.25,
          f"median_rel_err={rel.median().item():.5f} max={rel.max().item():.4f}")

    # 4. Real DP4A kernel, if the extension imported.
    if EXT and torch.cuda.is_available():
        try:
            y_gpu = quillan_sm61_qgemm.qgemm_forward(
                x_q.cuda(), w_q.cuda(), x_s.cuda(), w_s.cuda()
            )
            yg = y_gpu.float().cpu()
            big = y_ref.abs() > 0.05
            rel = ((yg[big] - y_ref[big]).abs() / (y_ref[big].abs() + 1e-8))
            check("dp4a kernel vs ref", rel.max().item() < 2e-2, f"max_rel_err={rel.max().item():.6f}")
        except Exception as e:
            check("dp4a kernel vs ref", False, f"launch failed: {e}")
    else:
        print("[smoke] SKIP: dp4a kernel (no extension or no CUDA device)")

    print("[smoke] all exercised checks passed.")


if __name__ == "__main__":
    sys.exit(main())
