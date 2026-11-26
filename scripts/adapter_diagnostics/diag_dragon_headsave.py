#!/usr/bin/env python3
"""Compare lm_head packed weights: base vs headrow-saved vs in-memory-applied."""

import sys
from pathlib import Path
import torch

ROOT = Path(r"C:\02_QUILLAN")
for p in (
    str(ROOT / "03 - Training & Model"),
    str(ROOT / "scripts"),
    str(ROOT / "scripts" / "adapter_diagnostics"),
):
    if p not in sys.path:
        sys.path.insert(0, p)
from wsp_math_eval import load_model

base = load_model(ROOT / "exports" / "quillan-dragon-packed.safetensors")
saved = load_model(ROOT / "exports" / "quillan-dragon-math40-headrow.safetensors")
for tag, m in [("base", base), ("saved", saved)]:
    h = m.lm_head
    print(
        f"{tag}: lm_head q_packed shape={tuple(h.q_packed.shape)} gamma_absmax={h.gamma.abs().max().item():.4f}"
    )
    W = h.unpack_effective_weight().float()
    print(
        f"  W[2356(A)] absmax={W[2356].abs().max().item():.4f} norm={W[2356].norm().item():.4f}"
    )
# diff in head rows
Wb = base.lm_head.unpack_effective_weight().float()
Ws = saved.lm_head.unpack_effective_weight().float()
for lid, nm in zip([2356, 2380, 2350, 2407], "ABCD"):
    d = (Ws[lid] - Wb[lid]).norm().item()
    print(f"row {nm}({lid}) delta norm base->saved = {d:.6f}")
print("all-head max diff:", float((Ws - Wb).abs().max()))
