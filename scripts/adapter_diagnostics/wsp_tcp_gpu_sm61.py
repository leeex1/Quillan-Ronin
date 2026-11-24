#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
wsp_tcp_gpu_sm61.py — GPU-enabled runner for the WSP probe on sm_61 (GTX 1050).

Runs the EXISTING wsp_tcp_gated_v2 probe logic with the model on CUDA, using the
isolated cu118 venv (Python 3.11 + torch 2.7.1+cu118). NO model-agent files are
edited. This file is a thin wrapper that monkeypatches device placement before
importing the probe's helper modules.

Usage:
  C:\02_QUILLAN\venv_sm61_test\Scripts\python.exe \
    C:\02_QUILLAN\scripts\adapter_diagnostics\wsp_tcp_gpu_sm61.py [probe args...]

Verify GPU is engaged:
  nvidia-smi --query-compute-apps=pid,used_memory --format=csv
"""

from __future__ import annotations
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import torch

if not torch.cuda.is_available():
    print("CUDA unavailable — run with venv_sm61_test (torch 2.7.1+cu118)")
    sys.exit(2)

print(
    f"[sm61] torch={torch.__version__} device={torch.cuda.get_device_name(0)}",
    flush=True,
)

# Patch load_model so every model it returns moves to CUDA.
import wsp_math_eval

_orig_load = wsp_math_eval.load_model


def _cuda_load(path):
    m = _orig_load(Path(path))
    return m.to("cuda").eval()


wsp_math_eval.load_model = _cuda_load

# Patch the helpers that create input tensors without a device.
_orig_last_logits = wsp_math_eval.last_logits


def _cuda_last_logits(model, ids):
    x = torch.tensor([ids], dtype=torch.long, device="cuda")
    with torch.inference_mode():
        logits, _ = model(x, use_cache=False)
    return logits[0, -1, :].float()


wsp_math_eval.last_logits = _cuda_last_logits

# Patch torch.tensor so probe-side implicit CPU tensors follow the model device.
# Scope: only tensors created AFTER this point; limits collateral to int/long ids.
_orig_tensor = torch.tensor


def _dev_tensor(data, *a, **kw):
    if "device" not in kw and torch.cuda.is_available():
        kw["device"] = "cuda"
    return _orig_tensor(data, *a, **kw)


torch.tensor = _dev_tensor

print(
    "[sm61] device patches installed; launching probe with args:",
    sys.argv[1:],
    flush=True,
)

# Run the existing probe's main() in-process with our patched helpers.
import wsp_tcp_gated_v2 as probe

probe.main()
