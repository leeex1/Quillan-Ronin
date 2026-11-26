#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Layer-fill audit for quillan-mini-sovereign-bitnet.safetensors.

Decisive test of the open fork: are the transformer layers actually filled with
donor (Qwen) weights, or left at the zero-initialization sentinel?

Sentinel: q_packed == 0x55 (all bytes) unpacks to ternary ZERO; gamma == 0.
A layer with sentinel q_packed AND zero gamma contributes exactly zero output.

This script reads ONLY the packed checkpoint (no donor forward, no GPU), reports:
  - per-layer gamma magnitude (L2 mean) and q_packed distinct-value count
  - fraction of layers at the zero sentinel
  - byte-level identity of norm weights vs the intended all-ones init
"""

from __future__ import annotations

import sys
from pathlib import Path

import safetensors.torch
import torch

CKPT = Path(
    sys.argv[1]
    if len(sys.argv) > 1
    else r"C:\02_QUILLAN\exports\quillan-mini-sovereign-bitnet.safetensors"
)


def main() -> None:
    with safetensors.torch.safe_open(str(CKPT), framework="pt", device="cpu") as f:
        keys = list(f.keys())
        print(f"[audit] checkpoint: {CKPT.name}  tensors={len(keys)}")
        meta = f.metadata() or {}
        print(f"[audit] config meta present: {'quillan_sovereign_config' in meta}")

    with safetensors.torch.safe_open(str(CKPT), framework="pt", device="cpu") as f:
        # Probe one projection per layer across all 7 projection families.
        families = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
        zero_layers: list[str] = []
        print("\n[per-layer gamma L2 mean / q_packed unique bytes]")
        print(f"{'layer':>6} " + " ".join(f"{fam[:6]:>9}" for fam in families))
        for i in range(24):
            cells = []
            layer_zero = True
            for fam in families:
                gkey = f"layers.{i}.{fam}.gamma"
                qkey = f"layers.{i}.{fam}.q_packed"
                if gkey not in keys:
                    cells.append("   MISS  ")
                    continue
                g = f.get_tensor(gkey).float()
                q = f.get_tensor(qkey)
                gmean = g.abs().mean().item()
                # 0x55 (85) means every 2-bit code == 1 -> ternary zero.
                sentinel = int((q == 0x55).all().item())
                uniq = int(torch.unique(q).numel())
                cells.append(f"{gmean:8.5f}{'*' if sentinel else ' '}")
                if not sentinel and gmean > 1e-6:
                    layer_zero = False
            print(f"{i:>6} " + " ".join(f"{c:>9}" for c in cells))
            if layer_zero:
                zero_layers.append(str(i))

        print("\n[norm weights: deviation from all-ones init]")
        for i in (0, 12, 23):
            for norm in ("input_layernorm", "post_attention_layernorm"):
                k = f"layers.{i}.{norm}.weight"
                if k in keys:
                    w = f.get_tensor(k).float()
                    print(
                        f"  {k}: mean={w.mean().item():.6f} "
                        f"dev_from_1={(w - 1).abs().mean().item():.6f}"
                    )
        if "norm.weight" in keys:
            w = f.get_tensor("norm.weight").float()
            print(
                f"  norm.weight: mean={w.mean().item():.6f} "
                f"dev_from_1={(w - 1).abs().mean().item():.6f}"
            )

    print(
        f"\n[VERDICT] zero-sentinel layers: {len(zero_layers)}/24 -> {','.join(zero_layers) if zero_layers else 'NONE'}"
    )
    if zero_layers:
        print("  => those layers contribute ZERO output (near-identity residual).")
    else:
        print(
            "  => all layers carry nonzero packed weights; failure is NOT a zero-fill bug."
        )


if __name__ == "__main__":
    main()
