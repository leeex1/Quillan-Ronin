#!/usr/bin/env python3
"""Verify the council MoE path is a free (zero-gated) slot on real checkpoints."""

from __future__ import annotations
import sys
from pathlib import Path
import torch

ROOT = Path(r"C:\02_QUILLAN")
for p in (str(ROOT / "scripts"), str(ROOT / "03 - Training & Model")):
    if p not in sys.path:
        sys.path.insert(0, p)
from wsp_math_eval import load_model

for name in [
    "quillan-mini-rank64.safetensors",
    "quillan-mini-math40-rank64.safetensors",
]:
    m = load_model(Path(r"C:\02_QUILLAN\exports") / name)
    slots = [i for i, l in enumerate(m.layers) if l.council_moe is not None]
    info = []
    for i in slots:
        moe = m.layers[i].council_moe
        info.append(
            f"L{i}:gate_alpha={float(moe.gate_alpha.detach()):.4f} ffn={moe.moe_ffn_dim} experts={moe.num_experts} w1_gamma_max={float(moe.w1_gamma.abs().max()):.4f}"
        )
    print(f"[{name}] moe_layers={slots}", flush=True)
    for s in info:
        print("  " + s, flush=True)
