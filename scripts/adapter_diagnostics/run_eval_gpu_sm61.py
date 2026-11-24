#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Attempt GPU routing for eval_quillan_trio_bitnet.py via loader override.
Tests whether the eval's explicit device="cpu" can be overridden non-invasively.
Usage: venv_sm61_test\python.exe run_eval_gpu_sm61.py <eval.py> [args...]
"""

from __future__ import annotations
import sys, os, runpy
from pathlib import Path
import torch

sys.path.insert(0, r"C:\02_QUILLAN\scripts")
sys.path.insert(0, r"C:\02_QUILLAN\03 - Training & Model")

if torch.cuda.is_available() and os.environ.get("QUILLAN_SM61_DIRECT") != "1":
    print(
        f"[sm61] eval GPU attempt | {torch.__version__} | {torch.cuda.get_device_name(0)}",
        flush=True,
    )
    import quillan_sovereign_bitnet as qsb

    _orig = qsb.QuillanSovereignBitNetLM.load_packed_safetensors

    @classmethod
    def _forced_cuda(cls, *a, **kw):
        kw["device"] = "cuda"  # override the eval's device="cpu"
        print("[sm61] forced loader device=cuda", flush=True)
        return _orig.__func__(cls, *a, **kw)

    qsb.QuillanSovereignBitNetLM.load_packed_safetensors = _forced_cuda

if len(sys.argv) < 2:
    print("usage: run_eval_gpu_sm61.py <eval.py> [args...]")
    sys.exit(2)
target = Path(sys.argv[1])
sys.argv = [str(target)] + sys.argv[2:]
print(f"[sm61] running {target.name} {sys.argv[1:]}", flush=True)
runpy.run_path(str(target), run_name="__main__")
