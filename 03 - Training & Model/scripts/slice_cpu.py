#!/usr/bin/env python3
"""Slice a CPU-sized subset (additive only, source untouched).
Takes first N shuffled-with-seed sequences for a stable subset."""
import sys
from pathlib import Path
import torch

REPO = Path(r"C:\02_QUILLAN")
SRC = REPO / "training_data" / "canonical_standardized" / "quillan_clean_v62_v1.pt"
OUT = REPO / "training_data" / "canonical_standardized" / "quillan_clean_v62_cpu20k.pt"
N = 20000

d = torch.load(SRC, map_location="cpu", weights_only=True)
t = d["input_ids"]
g = torch.Generator().manual_seed(7)
idx = torch.randperm(t.shape[0], generator=g)[:N]
sub = t[idx]
torch.save({"input_ids": sub, "labels": sub.clone()}, OUT)
print(f"slice {tuple(sub.shape)} -> {OUT.name} ({OUT.stat().st_size/1024**2:.0f}MB), source untouched",
      flush=True)
