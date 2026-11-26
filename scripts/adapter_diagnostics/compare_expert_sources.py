#!/usr/bin/env python3
"""Is the built 4-donor fusion actually IN the sovereign checkpoints?
Compare moe.w1/wgate/w2 tensors in donor_experts_aligned.safetensors vs the checkpoints."""

from __future__ import annotations
from pathlib import Path
import torch
from safetensors import safe_open

ROOT = Path(r"C:\02_QUILLAN")


def load_moe(p):
    out = {}
    with safe_open(str(p), framework="pt", device="cpu") as f:
        for k in f.keys():
            if k.startswith("moe."):
                out[k] = f.get_tensor(k)
    return out


don = load_moe(ROOT / "exports" / "donor_experts_aligned.safetensors")
print("donor file keys:", {k: list(v.shape) for k, v in don.items()})
for name in [
    "quillan-mini-sovereign-bitnet.safetensors",
    "quillan-flagship-sovereign-bitnet.safetensors",
]:
    p = ROOT / "exports" / name
    ck = load_moe(p)
    print(f"\n=== {name} ===")
    print("ckpt moe keys:", {k: list(v.shape) for k, v in ck.items()})
    for key in ["moe.w1", "moe.wgate", "moe.w2"]:
        if key in don and key in ck:
            a = don[key].float()
            b = ck[key].float()
            if a.shape != b.shape:
                print(
                    f"  {key}: SHAPE MISMATCH donor={tuple(a.shape)} ckpt={tuple(b.shape)}"
                )
                continue
            exact = bool(torch.equal(a, b))
            maxd = float((a - b).abs().max())
            rel = float((a - b).norm() / a.norm().clamp_min(1e-9))
            print(
                f"  {key}: exact={exact} max_abs_diff={maxd:.4f} rel_diff={rel:.4f} "
                f"donor_norm={float(a.norm()):.1f} ckpt_norm={float(b.norm()):.1f}"
            )
        else:
            print(
                f"  {key}: missing in donor={'no ' if key not in don else ''}ckpt={'no' if key not in ck else ''}"
            )
