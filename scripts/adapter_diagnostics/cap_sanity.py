#!/usr/bin/env python3
"""Sanity: does batched capture produce distinct, deterministic hidden states per prompt?"""

from __future__ import annotations
import sys
from pathlib import Path
import torch

ROOT = Path(r"C:\02_QUILLAN")
DIAG = ROOT / "scripts" / "adapter_diagnostics"
for p in (str(ROOT / "scripts"), str(ROOT / "03 - Training & Model"), str(DIAG)):
    if p not in sys.path:
        sys.path.insert(0, p)
from wsp_math_eval import tok_for, load_model
from wsp_math_tune_v3 import enable_cache
from latent_repro import capture_down_batched, fmt

m = load_model(Path(r"C:\02_QUILLAN\exports\quillan-mini-rank64.safetensors"))
enable_cache(m)
tok = tok_for(151936)
prompts = [fmt(a, b, None)[0] for a, b in [(3, 17), (11, 9), (25, 15), (1, 19)]]
X = capture_down_batched(m, tok, prompts)
X2 = capture_down_batched(m, tok, prompts)
print("shape", tuple(X.shape), flush=True)
print("deterministic maxdiff", float((X - X2).abs().max()), flush=True)
d = torch.cdist(X, X)
print("pairwise dist diag", [round(float(d[i, i]), 3) for i in range(4)], flush=True)
print(
    "pairwise dist offdiag",
    [round(float(d[i, j]), 3) for i in range(4) for j in range(4) if i != j],
    flush=True,
)
