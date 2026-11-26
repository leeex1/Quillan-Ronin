#!/usr/bin/env python3
from __future__ import annotations
import json
from collections import Counter, defaultdict
from pathlib import Path
from safetensors import safe_open

PATH = Path(
    r"C:\02_QUILLAN\exports\quillan-mini-sovereign-bitnet158-socket-v2.safetensors"
)
with safe_open(str(PATH), framework="pt", device="cpu") as f:
    meta = {
        k: (tuple(f.get_slice(k).get_shape()), f.get_slice(k).get_dtype())
        for k in f.keys()
    }

base = {k: v for k, v in meta.items() if not k.endswith(".bitnet_scale")}
print("BASE KEYS:", len(base))
print("DTYPES:", dict(Counter(d for _, d in base.values())))

prefix = Counter(k.split(".")[0] for k in base)
print("BASE PREFIX COUNTS:", json.dumps(dict(sorted(prefix.items())), indent=2))

# Scale keys
print("SCALE KEYS:", [k for k in meta if k.endswith(".bitnet_scale")])

# Spec shape claims vs actual
claims = {
    "ingestion.txt_emb.weight": [151936, 896],
    "txt_dec.weight": [151936, 896],
    "txt_dec.lora_A": [896, 256],
    "txt_dec.lora_B": [256, 151936],
    "moe.w1": [34, 896, 4864],
    "moe.wgate": [34, 896, 4864],
    "moe.w2": [34, 4864, 896],
    "moe.w1_lora_A": [34, 896, 16],
    "moe.w1_lora_B": [34, 16, 4864],
    "moe.w2_lora_A": [34, 896, 16],  # spec lumps w1/wgate/w2 as [34,896,16]
    "moe.w2_lora_B": [34, 16, 4864],  # spec lumps
    "diffusion_core.couil_attn.q_proj.weight": [896, 896],
    "diffusion_core.couil_attn.k_proj.weight": [128, 896],
    "diffusion_core.couil_attn.v_proj.weight": [128, 896],
    "diffusion_core.couil_attn.o_proj.weight": [896, 896],
    "diffusion_core.ffn.0.weight": [3584, 896],
    "diffusion_core.ffn.2.weight": [896, 3584],
    "ingestion.pos_embed": [1, 1024, 896],
    "quillan_gate.weight": [896, 1792],
}
print("\n=== SPEC SHAPE CLAIMS vs ACTUAL ===")
for k, claim in claims.items():
    actual = list(base[k][0]) if k in base else None
    ok = "OK" if actual == claim else "MISMATCH"
    print(
        f"{ok:9} | {k:52} | spec={claim} actual={actual} dtype={base.get(k, (None, None))[1]}"
    )
