#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_swappable_model.py — load whatever model is named in model_launcher_config.json
and run a short generation on the GPU. Swap the model by editing ONE config field.

Uses the cu118 venv (torch 2.7.1+cu118), clang-cl shim for BitNet conversion,
and KMP_DUPLICATE_LIB_OK for the OpenMP runtime conflict. All env setup is done
here so the caller doesn't need to remember it.

Usage:
  C:\02_QUILLAN\venv_sm61_test\Scripts\python.exe C:\02_QUILLAN\scripts\run_swappable_model.py
  C:\02_QUILLAN\venv_sm61_test\Scripts\python.exe ... "custom prompt here"
"""

from __future__ import annotations
import os, sys, json, time
from pathlib import Path

CFG = Path(r"C:\02_QUILLAN\daemons\model_launcher_config.json")
# Env setup (must precede torch import)
_SHIM = r"C:\Users\Admin\AppData\Local\Temp\opencode\cl_shim"
os.environ["PATH"] = (
    _SHIM
    + os.pathsep
    + r"C:\Program Files\LLVM\bin"
    + os.pathsep
    + os.environ.get("PATH", "")
)
os.environ["TORCHINDUCTOR_CXX_COMPILER"] = r"C:\Program Files\LLVM\bin\clang-cl.exe"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

cfg = json.loads(CFG.read_text(encoding="utf-8"))
model_path = cfg["model_path"]
prompt = (
    sys.argv[1] if len(sys.argv) > 1 else cfg.get("prompt", "The capital of France is")
)
max_new = int(cfg.get("max_new_tokens", 24))
dtype = getattr(torch, cfg.get("dtype", "bfloat16"))
device = cfg.get("device", "cuda")

if not torch.cuda.is_available() and device == "cuda":
    print("[swappable] CUDA unavailable; falling back to CPU")
    device = "cpu"

print(f"[swappable] model={model_path}", flush=True)
print(f"[swappable] device={device} dtype={dtype} prompt={prompt!r}", flush=True)
t0 = time.time()
tok = AutoTokenizer.from_pretrained(model_path)
m = AutoModelForCausalLM.from_pretrained(model_path, dtype=dtype).to(device).eval()
print(f"[swappable] model loaded in {time.time() - t0:.1f}s", flush=True)
ids = tok.encode(prompt, return_tensors="pt").to(device)
with torch.no_grad():
    out = m.generate(ids, max_new_tokens=max_new, do_sample=False)
text = tok.decode(out[0], skip_special_tokens=True)
print(f"[swappable] OUTPUT: {text}", flush=True)
print(f"[swappable] OK — swap by editing model_path in {CFG}", flush=True)
