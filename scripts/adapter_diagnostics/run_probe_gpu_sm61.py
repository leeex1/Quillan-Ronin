#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_probe_gpu_sm61.py — generic GPU wrapper for any wsp_math_eval-based probe.

Usage (from cu118 venv):
  python run_probe_gpu_sm61.py <probe.py> [probe args...]

Patches load_model -> CUDA and torch.tensor -> CUDA, then runs the target probe
as __main__. No edits to model-agent scripts. Disable GPU routing with env
QUILLAN_SM61_DIRECT=1.
"""

from __future__ import annotations
import sys, os, runpy
from pathlib import Path
import torch

DIAG = Path(r"C:\02_QUILLAN\scripts\adapter_diagnostics")
sys.path.insert(0, str(DIAG))
sys.path.insert(0, r"C:\02_QUILLAN\scripts")
sys.path.insert(0, r"C:\02_QUILLAN\03 - Training & Model")

if os.environ.get("QUILLAN_SM61_DIRECT") != "1" and torch.cuda.is_available():
    print(
        f"[sm61] generic GPU wrapper | torch={torch.__version__} | {torch.cuda.get_device_name(0)}",
        flush=True,
    )

    # Patch load_model -> CUDA
    import wsp_math_eval

    _orig = wsp_math_eval.load_model

    def _cuda_load(path):
        return _orig(Path(path)).to("cuda").eval()

    wsp_math_eval.load_model = _cuda_load

    # Patch tensor factories -> CUDA default (probe-side intermediate tensors).
    # Injects device only when the caller did not specify one explicitly.
    def _patch_factory(name):
        orig = getattr(torch, name)

        def wrapped(*a, **kw):
            # An explicit generator fixes the device; don't inject a conflicting one.
            if "device" not in kw and "generator" not in kw:
                kw["device"] = "cuda"
            return orig(*a, **kw)

        setattr(torch, name, wrapped)

    for _fn in (
        "tensor",
        "eye",
        "zeros",
        "ones",
        "randn",
        "rand",
        "arange",
        "full",
        "empty",
        "zeros_like",
        "ones_like",
    ):
        try:
            _patch_factory(_fn)
        except Exception:
            pass

if len(sys.argv) < 2:
    print("usage: run_probe_gpu_sm61.py <probe.py> [args...]")
    sys.exit(2)
target = Path(sys.argv[1])
if not target.exists():
    target = DIAG / sys.argv[1]
sys.argv = [str(target)] + sys.argv[2:]
print(f"[sm61] running {target.name} with args {sys.argv[1:]}", flush=True)
runpy.run_path(str(target), run_name="__main__")
