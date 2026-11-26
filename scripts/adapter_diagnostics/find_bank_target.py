#!/usr/bin/env python3
"""Which checkpoint schema does the 4-donor bank (moe.w1 F16) actually fit?"""

from pathlib import Path
from safetensors import safe_open

ROOT = Path(r"C:\02_QUILLAN\exports")
for p in sorted(ROOT.glob("*.safetensors")):
    try:
        with safe_open(str(p), framework="pt", device="cpu") as f:
            ks = list(f.keys())
        has_dense = "moe.w1" in ks
        has_packed = any("council_moe.w1_q" in k for k in ks)
        if has_dense or has_packed:
            print(
                f"{p.name}: dense_moe_w1={has_dense} packed_council_moe={has_packed} total_keys={len(ks)}"
            )
    except Exception as e:
        print(f"{p.name}: ERR {type(e).__name__}")
