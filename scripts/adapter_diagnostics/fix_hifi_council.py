#!/usr/bin/env python3
"""Fix the silently-zeroed Council bank in hifi checkpoints.

Predator audit: build_hifidelity_mini.py skipped keys with '.gamma' (dot) but
MoE buffers use 'w1_gamma' (underscore); their shape changed with group_size so
the shape guard dropped them -> expert scales all zero -> dead Council.

Fix: for each hifi checkpoint, unpack the ORIGINAL base council expert matrices
(group32) and repack them at the checkpoint's group size, writing q_packed +
gamma into the correct buffers. Preserves gate_alpha and router from base.
"""

from __future__ import annotations
import sys
from pathlib import Path
import torch

ROOT = Path(r"C:\02_QUILLAN")
for p in (
    str(ROOT / "scripts"),
    str(ROOT / "03 - Training & Model"),
    str(ROOT / "scripts" / "adapter_diagnostics"),
):
    if p not in sys.path:
        sys.path.insert(0, p)
from quillan_sovereign_bitnet import (
    quantize_group_ternary_ls,
    pack_ternary_2bit,
    unpack_ternary_2bit,
)
from safetensors.torch import safe_open, save_file

torch.set_num_threads(2)
BASE = ROOT / "exports" / "quillan-mini-sovereign-bitnet.safetensors"
LAYERS = (11, 22)
SUBS = (("w1", 512, 896), ("wgate", 512, 896), ("w2", 896, 512))


def unpack_expert(q, g, out_dim, in_dim, group):
    gc = in_dim // group
    t = unpack_ternary_2bit(q, (out_dim, gc, group), out_dim * in_dim)
    return (t * g.float().unsqueeze(-1)).view(out_dim, in_dim)


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--ck", required=True)
    ap.add_argument("--group", type=int, required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    with safe_open(str(BASE), framework="pt", device="cpu") as f:
        base = {
            f"layers.{li}.council_moe.{n}_{suffix}": f.get_tensor(
                f"layers.{li}.council_moe.{n}_{suffix}"
            )
            for li in LAYERS
            for n, _, _ in SUBS
            for suffix in ("q", "gamma")
        }
    path = ROOT / "exports" / a.ck
    with safe_open(str(path), framework="pt", device="cpu") as f:
        keys = list(f.keys())
        meta = dict(f.metadata() or {})
        tens = {k: f.get_tensor(k).clone() for k in keys}
    for li in LAYERS:
        for name, out_dim, in_dim in SUBS:
            bq = base[f"layers.{li}.council_moe.{name}_q"].byte()
            bg = base[f"layers.{li}.council_moe.{name}_gamma"].float()
            qs = []
            gs = []
            # base gamma layout: (34,out,g_in_base) with g_in_base=in_dim//32
            for e in range(34):
                W = unpack_expert(bq[e], bg[e], out_dim, in_dim, 32)
                t, g = quantize_group_ternary_ls(W, group_size=a.group)
                pk, _ = pack_ternary_2bit(t)
                qs.append(pk)
                gs.append(g.to(torch.float16))
            tens[f"layers.{li}.council_moe.{name}_q"] = torch.stack(qs).contiguous()
            tens[f"layers.{li}.council_moe.{name}_gamma"] = torch.stack(gs).contiguous()
    out = ROOT / "exports" / a.out
    save_file(tens, str(out), metadata=meta)
    print(f"FIXED {a.ck} -> {a.out} group={a.group}", flush=True)
    with safe_open(str(out), framework="pt", device="cpu") as f:
        g = f.get_tensor("layers.11.council_moe.w1_gamma").float()
        print(
            f"  w1_gamma max={g.abs().max().item():.4f} nonzero={(g != 0).sum().item()}/{g.numel()}",
            flush=True,
        )


if __name__ == "__main__":
    main()
