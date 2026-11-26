#!/usr/bin/env python3
"""Do the dense bank and packed council experts have compatible dimensions?"""

from pathlib import Path
from safetensors import safe_open

ROOT = Path(r"C:\02_QUILLAN\exports")

with safe_open(
    str(ROOT / "donor_experts_aligned.safetensors"), framework="pt", device="cpu"
) as f:
    print(
        "BANK dense: w1",
        tuple(f.get_tensor("moe.w1").shape),
        "wgate",
        tuple(f.get_tensor("moe.wgate").shape),
        "w2",
        tuple(f.get_tensor("moe.w2").shape),
    )

with safe_open(
    str(ROOT / "quillan-mini-sovereign-bitnet.safetensors"),
    framework="pt",
    device="cpu",
) as f:
    for k in [
        "layers.11.council_moe.w1_q",
        "layers.11.council_moe.w1_gamma",
        "layers.11.council_moe.w2_q",
        "layers.11.council_moe.w2_gamma",
    ]:
        t = f.get_tensor(k)
        print(f"CKPT {k.split('.')[-1]}: shape={tuple(t.shape)} dtype={t.dtype}")
    # infer ffn from gamma: w1_gamma [E, ffn, hidden/group]
    g = f.get_tensor("layers.11.council_moe.w1_gamma")
    q = f.get_tensor("layers.11.council_moe.w1_q")
    print(
        f"  -> w1_gamma[1]={g.shape[1]} (moe_ffn_dim), w1_gamma[2]={g.shape[2]} (hidden/group)"
    )
    print(f"  -> w1_q packed_len={q.shape[1]} -> logical elements ~ {q.shape[1] * 4}")
