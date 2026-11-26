#!/usr/bin/env python3
"""Verify param counts using the REAL owner_config presets with dim overrides."""

import sys
from dataclasses import replace
import torch

sys.path.insert(0, r"F:\02_QUILLAN\03 - Training & Model\scripts")
from oni_v2_config import owner_config
from quillan_v5_4_oni import QuillanRoninOni

# candidates from the bare-config sweep
MINI = [(1024, 2048, 8), (1152, 2304, 8), (1152, 3072, 6), (1280, 2560, 6)]
MAIN = [(1536, 3072, 12), (1280, 2560, 12), (1152, 2304, 12), (1024, 2048, 12)]
for base, cands in (("mini", MINI), ("main", MAIN)):
    for H, F, L in cands:
        c = replace(
            owner_config(base),
            hidden_dim=H,
            ffn_dim=F,
            n_layer=L,
            n_head=H // 64,
            head_dim=64,
        )
        m = QuillanRoninOni(c)
        p = sum(x.numel() for x in m.parameters())
        print(f"{base} H={H} F={F} L={L} -> {p / 1e9:.3f}B", flush=True)
        del m
