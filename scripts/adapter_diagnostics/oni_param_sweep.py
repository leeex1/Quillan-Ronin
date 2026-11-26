#!/usr/bin/env python3
"""Find ONI body dimensions matching owner size targets (~0.5B mini, ~1B main).
Uses tiny vocab for fast non-embedding count, then adds locked embedding rows.
"""

import sys
import torch

sys.path.insert(0, r"F:\02_QUILLAN\03 - Training & Model\scripts")
from quillan_v5_4_oni import QuillanOniConfig, QuillanRoninOni

V = 151648
for H in (768, 896, 1024, 1152, 1280, 1536):
    for L in (4, 6, 8, 12):
        F = H * 2
        cfg = QuillanOniConfig(
            vocab_size=1000,
            hidden_dim=H,
            ffn_dim=F,
            n_layer=L,
            n_head=max(4, H // 64),
            head_dim=64,
            num_experts=34,
            max_seq_len=512,
            device="cpu",
        )
        m = QuillanRoninOni(cfg)
        non_emb = sum(p.numel() for p in m.parameters()) - 1000 * H
        total = non_emb + V * H
        print(f"H={H:4d} F={F:4d} L={L:2d} -> {total / 1e9:.3f}B", flush=True)
        del m
