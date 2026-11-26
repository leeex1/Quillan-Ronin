#!/usr/bin/env python3
import sys
from dataclasses import replace

sys.path.insert(0, r"F:\02_QUILLAN\03 - Training & Model\scripts")
from oni_v2_config import owner_config
from quillan_v5_4_oni import QuillanRoninOni

base, H, F, L = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
c = replace(
    owner_config(base), hidden_dim=H, ffn_dim=F, n_layer=L, n_head=H // 64, head_dim=64
)
m = QuillanRoninOni(c)
p = sum(x.numel() for x in m.parameters())
print(f"{base} H={H} F={F} L={L} -> {p / 1e9:.4f}B ({p})", flush=True)
