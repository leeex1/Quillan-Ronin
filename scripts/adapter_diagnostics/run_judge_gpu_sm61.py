#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GPU wrapper for judge_sovereign_bitnet.py on sm_61. No edits to the judge itself.
Usage: venv_sm61_test\python.exe run_judge_gpu_sm61.py <judge.py> [args...]
Disable with env QUILLAN_SM61_DIRECT=1.
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
        f"[sm61] judge GPU wrapper | {torch.__version__} | {torch.cuda.get_device_name(0)}",
        flush=True,
    )

    # judge has its OWN load_model (not wsp_math_eval's). Patch by injecting a
    # CUDA-moving wrapper into the module namespace when it is run via runpy.
    # We patch torch.tensor factories + the classmethod at import time via a
    # meta-path hook is overkill; instead patch torch.set_num_threads no-op and
    # rely on load_model being called after our tensor factory patches.
    def _patch(name):
        orig = getattr(torch, name)

        def wrapped(*a, **kw):
            if "device" not in kw and "generator" not in kw:
                kw["device"] = "cuda"
            return orig(*a, **kw)

        setattr(torch, name, wrapped)

    for fn in ("tensor", "eye", "zeros", "ones", "randn", "rand", "arange"):
        try:
            _patch(fn)
        except Exception:
            pass
    # Patch the native loader's classmethod so it returns a CUDA model.
    try:
        import quillan_sovereign_bitnet as qsb

        _orig_load = qsb.QuillanSovereignBitNetLM.load_packed_safetensors

        @classmethod
        def _cuda_load(cls, *a, **kw):
            m = _orig_load.__func__(cls, *a, **kw)
            return m.to("cuda")

        qsb.QuillanSovereignBitNetLM.load_packed_safetensors = _cuda_load
        print("[sm61] judge loader patched to CUDA", flush=True)
    except Exception as e:
        print(f"[sm61] loader patch skipped: {e}", flush=True)

if len(sys.argv) < 2:
    print("usage: run_judge_gpu_sm61.py <judge.py> [args...]")
    sys.exit(2)
target = Path(sys.argv[1])
if not target.exists():
    target = DIAG / sys.argv[1]
sys.argv = [str(target)] + sys.argv[2:]
print(f"[sm61] running {target.name} {sys.argv[1:]}", flush=True)
runpy.run_path(str(target), run_name="__main__")
