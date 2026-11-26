#!/usr/bin/env python3
"""Falsify the harness: are captured layer-8 MoE-input states distinct & deterministic?"""

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
from stage_a_feasibility import prompt, capture

m = load_model(Path(r"C:\02_QUILLAN\exports\quillan-mini-carved-gate010.safetensors"))
enable_cache(m)
tok = tok_for(151936)
ps = [prompt(a, b) for a, b in [(0, 0), (1, 1), (2, 3), (7, 8), (9, 9)]]
X = capture(m, tok, ps)
X2 = capture(m, tok, ps)
print("shape", tuple(X.shape), flush=True)
print("det_maxdiff", float((X - X2).abs().max()), flush=True)
d = torch.cdist(X, X)
print(
    "offdiag_min",
    float(d[d > 0].min()),
    "offdiag_mean",
    float(d[~torch.eye(len(ps), dtype=torch.bool)].mean()),
    flush=True,
)
print(
    "per-state norm", [round(float(X[i].norm()), 3) for i in range(len(ps))], flush=True
)
