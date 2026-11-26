#!/usr/bin/env python3
"""Compare packed MoE buffers in checkpoint vs built donor fusion (correct key names)."""

from pathlib import Path
import torch
from safetensors import safe_open

ROOT = Path(r"C:\02_QUILLAN")


def keys(p):
    with safe_open(str(p), framework="pt", device="cpu") as f:
        return [k for k in f.keys() if "moe" in k.lower() or "w1" in k.lower()]


don_p = ROOT / "exports" / "donor_experts_aligned.safetensors"
ck_p = ROOT / "exports" / "quillan-mini-sovereign-bitnet.safetensors"
print("DONOR keys:", keys(don_p))
ck_keys = keys(ck_p)
print("CKPT  keys (first 20):", ck_keys[:20])
print("CKPT  moe-ish key count:", len(ck_keys))

# Find a council_moe gamma buffer and compare to donor w1 gamma derived from F16
with safe_open(str(ck_p), framework="pt", device="cpu") as f:
    gkeys = [k for k in f.keys() if "council_moe" in k and "gamma" in k]
    print("council gamma buffers:", gkeys[:6], "... total", len(gkeys))
    if gkeys:
        g = f.get_tensor(gkeys[0]).float()
        print(
            f"  {gkeys[0]} shape={tuple(g.shape)} max={float(g.abs().max()):.3f} nonzero={(g != 0).sum().item()}"
        )

with safe_open(str(don_p), framework="pt", device="cpu") as f:
    w1 = f.get_tensor("moe.w1").float()  # [34,896,4864] ternary*gamma F16
    print(
        f"donor moe.w1 norm={float(w1.norm()):.1f} max={float(w1.abs().max()):.3f} nonzero={(w1 != 0).sum().item()}/{w1.numel()}"
    )
