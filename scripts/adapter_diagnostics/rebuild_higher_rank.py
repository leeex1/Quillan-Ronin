#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Rebuild a checkpoint with eggroll_rank 16 -> R (64) via the model class.

Raw tensor surgery would fail load_state_dict (shape mismatch). Correct path:
construct a NEW QuillanSovereignBitNetLM with cfg.eggroll_rank=R, copy every
compatible tensor from the old checkpoint, and ZERO-PAD the wider LoRA A/B
(output-neutral: lora_B=0 => LoRA branch contributes 0). Save as new checkpoint.
Closed-form, zero backprop, canonical untouched.
"""

from __future__ import annotations
import sys, json
from dataclasses import asdict
from pathlib import Path
import torch

ROOT = Path(r"C:\02_QUILLAN")
for p in (str(ROOT / "03 - Training & Model"), str(ROOT / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)
from quillan_sovereign_bitnet import QuillanSovereignBitNetLM, QuillanSovereignConfig


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--rank", type=int, default=64)
    a = ap.parse_args()
    old = QuillanSovereignBitNetLM.load_packed_safetensors(a.ckpt)
    cfg = QuillanSovereignConfig(**{**asdict(old.cfg)})
    old_rank = cfg.eggroll_rank
    cfg.eggroll_rank = a.rank
    print(f"[rank] {old_rank} -> {a.rank}; building new model", flush=True)
    new = QuillanSovereignBitNetLM(cfg)
    old_sd = old.state_dict()
    new_sd = new.state_dict()
    copied = 0
    padded = 0
    missing = []
    for k, v in new_sd.items():
        if k not in old_sd:
            missing.append(k)
            continue
        ov = old_sd[k]
        if tuple(v.shape) == tuple(ov.shape):
            v.copy_(ov)
            copied += 1
        elif "lora_A" in k or "lora_B" in k:
            # zero-pad from old rank to new rank (keep learned part, new dims zero)
            if "lora_A" in k:  # [rank, in]
                v[:old_rank].copy_(ov[:old_rank])
                padded += 1
            else:  # [out, rank]
                v[:, :old_rank].copy_(ov[:, :old_rank])
                padded += 1
        else:
            missing.append(f"{k} SHAPE {tuple(ov.shape)}!={tuple(v.shape)}")
    new.save_packed_safetensors(a.out)
    print(f"[done] copied={copied} padded={padded} issues={missing[:5]}", flush=True)
    print(f"[verify] reload -> ", end="", flush=True)
    chk = QuillanSovereignBitNetLM.load_packed_safetensors(a.out)
    print(
        f"rank={chk.cfg.eggroll_rank} lora_A={tuple(chk.layers[-1].down_proj.lora_A.shape)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
