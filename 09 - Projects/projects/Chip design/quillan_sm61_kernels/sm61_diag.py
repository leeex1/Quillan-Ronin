#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sm61_diag.py — diagnose your torch/card combo for the sm_61 (Pascal) trap.

Compares your card's compute capability against torch.cuda.get_arch_list()
and prints a straight verdict + fix. Run with the python you train/serve with:

    python sm61_diag.py
"""
from __future__ import annotations

import sys


def main() -> int:
    try:
        import torch
    except ImportError:
        print("FAIL: torch is not installed in this interpreter.")
        return 2

    print(f"torch: {torch.__version__}  (CUDA runtime {torch.version.cuda})")
    if not torch.cuda.is_available():
        print("FAIL: torch.cuda.is_available() is False — no CUDA device visible.")
        print("Fix: install an NVIDIA driver + a CUDA torch wheel.")
        return 1

    archs = torch.cuda.get_arch_list()
    major, minor = torch.cuda.get_device_capability(0)
    sm = f"sm_{major}{minor}"
    name = torch.cuda.get_device_name(0)
    print(f"card: {name}  ->  {sm}")
    print(f"torch arch_list: {archs}")

    if any(sm in a for a in archs):
        print(f"PASS: {sm} is covered — torch kernels will launch on this card.")
        # Prove it with a real kernel launch, not just metadata.
        try:
            x = torch.randn(64, 64, device="cuda")
            y = torch.randn(64, 64, device="cuda")
            (x @ y).sum().item()
            print("PASS: real CUDA matmul kernel launched and completed.")
        except Exception as e:
            print(f"FAIL: metadata claims {sm} support but kernel launch died: {e}")
            return 1
        return 0

    print(f"TRAP: torch.cuda.is_available() is True but {sm} is NOT in arch_list.")
    print("The first torch kernel launch will die with:")
    print("  CUDA error: no kernel image is available for execution on the device.")
    print("Real fixes (no shim exists — the kernels were never compiled in):")
    print("  Path A (training): pip install torch --index-url "
          "https://download.pytorch.org/whl/cu126   # 2.14, last Pascal wheel")
    print("  Path B (newer torch): build from source with TORCH_CUDA_ARCH_LIST=6.1")
    print("  Path C (inference):   int8 DP4A extension in this folder "
          "(carries its own sm_61 SASS+PTX; only needs torch's allocator).")
    return 1


if __name__ == "__main__":
    sys.exit(main())
