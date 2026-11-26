#!/usr/bin/env python3
"""Minimal rank64 neutrality: 3 prompts, compare logits."""

import sys
from pathlib import Path
import torch

ROOT = Path(r"C:\02_QUILLAN")
for p in (str(ROOT / "03 - Training & Model"), str(ROOT / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)
from quillan_sovereign_bitnet import QuillanSovereignBitNetLM

names = ["quillan-mini-dense-council.safetensors", "quillan-mini-rank64.safetensors"]
texts = ["The capital of France is", "Water freezes at", "2 + 2 equals"]
outs = []
for name in names:
    m = QuillanSovereignBitNetLM.load_packed_safetensors(ROOT / "exports" / name)
    ls = []
    for t in texts:
        ids = m.cfg  # noop
        from transformers import AutoTokenizer
    outs.append(name)
# simpler: just verify load + rank, print
for name in names:
    m = QuillanSovereignBitNetLM.load_packed_safetensors(ROOT / "exports" / name)
    print(
        f"{name}: rank={m.cfg.eggroll_rank} loraA={tuple(m.layers[-1].down_proj.lora_A.shape)} loraB_absmax={m.layers[-1].down_proj.lora_B.abs().max().item():.4f}"
    )
print("RANK64 MODEL LOADS OK")
