#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Patch a QuillanSovereignBitNet checkpoint's MoE tensors with the corrected
all-four-donor expert bank (donor_experts_aligned.safetensors).

Why: build_quillan_trio_bitnet.py seeds MoE from a stale internal bank
(dragon_1b sparse) — it does NOT load the corrected verified bank. This patch
copies the corrected moe.w1/wgate/w2 into the checkpoint so all 4 donors
(qwen/llama/falcon REAL BF16/bitnet) are genuinely present.

Reads checkpoint + bank, writes to a NEW file (never overwrites input).
Zero backprop. No gate/canonical changes. Verifies shapes + finite after patch.
"""

from __future__ import annotations
import sys, json
from pathlib import Path
import torch
from safetensors.torch import safe_open, save_file

CKPT = (
    Path(sys.argv[1])
    if len(sys.argv) > 1
    else Path(r"C:\02_QUILLAN\exports\quillan-mini-dense-council.safetensors")
)
BANK = Path(r"C:\02_QUILLAN\exports\donor_experts_aligned.safetensors")
OUT = CKPT.with_name(CKPT.stem + "-moe4.safetensors")
KEYS = ["moe.w1", "moe.wgate", "moe.w2"]


def main():
    print(f"[patch] checkpoint: {CKPT}")
    print(f"[patch] bank:       {BANK}")
    with safe_open(str(BANK), framework="pt", device="cpu") as f:
        bank_keys = set(f.keys())
        patches = {k: f.get_tensor(k) for k in KEYS if k in bank_keys}
    print(f"[patch] bank MoE keys found: {list(patches)}")
    assert set(patches) == set(KEYS), (
        f"bank missing MoE keys: {set(KEYS) - set(patches)}"
    )

    with safe_open(str(CKPT), framework="pt", device="cpu") as f:
        keys = list(f.keys())
        meta = dict(f.metadata() or {})
        tensors = {}
        for k in keys:
            t = f.get_tensor(k)
            if k in patches:
                assert tuple(t.shape) == tuple(patches[k].shape), (
                    f"shape mismatch {k}: ckpt {t.shape} vs bank {patches[k].shape}"
                )
                print(f"  patched {k}: {tuple(t.shape)}")
                t = patches[k]
            tensors[k] = t.contiguous()
    save_file(tensors, str(OUT), metadata=meta)
    # verify finite
    with safe_open(str(OUT), framework="pt", device="cpu") as f:
        for k in KEYS:
            t = f.get_tensor(k).float()
            print(
                f"[verify] {k}: finite={bool(torch.isfinite(t).all())} nonzero_frac={(t != 0).float().mean().item():.3f} absm={t.abs().mean().item():.5f}"
            )
    print(f"[done] {OUT} ({OUT.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
