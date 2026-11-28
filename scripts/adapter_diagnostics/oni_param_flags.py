#!/usr/bin/env python3
import sys
from dataclasses import replace

sys.path.insert(0, r"F:\02_QUILLAN\03 - Training & Model\scripts")
from oni_v2_config import owner_config
from quillan_v5_4_oni import QuillanRoninOni

H, F, L = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
cfg = replace(
    owner_config("mini"),
    hidden_dim=H,
    ffn_dim=F,
    n_layer=L,
    n_head=H // 64,
    head_dim=64,
    use_evo_moe=False,
    use_mod=False,
    use_agent_evolution=False,
    use_real_swarm=False,
    use_world_model=False,
    couil_sparse_heads=False,
    use_memory_attention=False,
    use_fa3=False,
    use_moba=False,
)
m = QuillanRoninOni(cfg)
p = sum(x.numel() for x in m.parameters())
print(f"H={H} F={F} L={L} flags_off -> {p} ({p / 1e9:.4f}B)")
