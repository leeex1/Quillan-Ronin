#!/usr/bin/env python3
"""SHAM control: replace donor expert weights with RANDOM ternary of matched
magnitude, keep the same random router + gate. If random experts also improve
GK, the +8 from the LS bank is generic perturbation, not donor knowledge.
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
from quillan_sovereign_bitnet import quantize_group_ternary_ls, pack_ternary_2bit
from safetensors.torch import safe_open, save_file

torch.manual_seed(20261124)
SRC = ROOT / "exports" / "quillan-mini-council-ls.safetensors"
OUT = ROOT / "exports" / "quillan-mini-council-sham.safetensors"
GROUP = 32
LAYERS = (11, 22)


def main():
    with safe_open(str(SRC), framework="pt", device="cpu") as f:
        keys = list(f.keys())
        meta = dict(f.metadata() or {})
        tensors = {k: f.get_tensor(k).clone() for k in keys}
    for li in LAYERS:
        # read donor gamma to match magnitude stats
        g1 = tensors[f"layers.{li}.council_moe.w1_gamma"].float()
        scale = float(g1.abs().mean())
        for name, out_dim, in_dim in (
            ("w1", 512, 896),
            ("wgate", 512, 896),
            ("w2", 896, 512),
        ):
            qkey = f"layers.{li}.council_moe.{name}_q"
            gkey = f"layers.{li}.council_moe.{name}_gamma"
            # random weights at matched scale
            W = torch.randn(34, out_dim, in_dim) * scale
            qs, gs = [], []
            for e in range(34):
                t, g = quantize_group_ternary_ls(W[e], group_size=GROUP)
                pk, _ = pack_ternary_2bit(t)
                qs.append(pk)
                gs.append(g.to(torch.float16))
            tensors[qkey] = torch.stack(qs).contiguous()
            tensors[gkey] = torch.stack(gs).contiguous()
    save_file(tensors, str(OUT), metadata=meta)
    print(f"SAVED {OUT}", flush=True)


if __name__ == "__main__":
    main()
