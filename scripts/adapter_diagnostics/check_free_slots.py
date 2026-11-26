#!/usr/bin/env python3
"""Precondition check for C8 PTP: which layers have zero-init (free) LoRA slots?"""

from __future__ import annotations
import sys
from pathlib import Path
import torch

ROOT = Path(r"C:\02_QUILLAN")
DIAG = ROOT / "scripts" / "adapter_diagnostics"
for p in (str(ROOT / "scripts"), str(ROOT / "03 - Training & Model"), str(DIAG)):
    if p not in sys.path:
        sys.path.insert(0, p)
from wsp_math_eval import load_model

CK = ROOT / "exports" / "quillan-mini-rank64.safetensors"
m = load_model(CK)
print(f"layers={len(m.layers)}", flush=True)
for i, l in enumerate(m.layers):
    mods = {
        "q": l.q_proj,
        "k": l.k_proj,
        "v": l.v_proj,
        "o": l.o_proj,
        "gate": l.gate_proj,
        "up": l.up_proj,
        "down": l.down_proj,
    }
    row = []
    for nm, md in mods.items():
        if md.lora_B is None:
            row.append(f"{nm}:NA")
            continue
        mx = float(md.lora_B.abs().max())
        row.append(f"{nm}:{mx:.4f}")
    moe = "moe" if l.council_moe is not None else "-"
    print(f"L{i:02d} [{moe:3s}] " + " ".join(row), flush=True)
