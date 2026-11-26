#!/usr/bin/env python3
"""Does the new checkpoint actually load the 4-donor council buffers, and do they differ from original?"""

from pathlib import Path
import torch
from safetensors.torch import safe_open

ROOT = Path(r"C:\02_QUILLAN\exports")
A = ROOT / "quillan-mini-sovereign-bitnet.safetensors"
B = ROOT / "quillan-mini-council4.safetensors"


def get(p, k):
    with safe_open(str(p), framework="pt", device="cpu") as f:
        return f.get_tensor(k)


for li in (11, 22):
    for suf in (
        "w1_q",
        "w1_gamma",
        "wgate_q",
        "wgate_gamma",
        "w2_q",
        "w2_gamma",
        "gate_alpha",
    ):
        k = f"layers.{li}.council_moe.{suf}"
        a = get(A, k)
        b = get(B, k)
        if suf == "gate_alpha":
            print(f"L{li} {suf}: orig={float(a)} ported={float(b)}")
        else:
            eq = bool(torch.equal(a, b))
            d = float((a.float() - b.float()).abs().mean())
            print(
                f"L{li} {suf}: identical={eq} mean_abs_diff={d:.4f} orig_absm={float(a.float().abs().mean()):.4f} ported_absm={float(b.float().abs().mean()):.4f}"
            )

# Verify the ported w1 actually unpacks to something finite and non-trivial
import sys

for p in (str(ROOT.parent / "03 - Training & Model"), str(ROOT.parent / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)
from quillan_sovereign_bitnet import unpack_ternary_2bit

with safe_open(str(B), framework="pt", device="cpu") as f:
    q = f.get_tensor("layers.11.council_moe.w1_q")
    g = f.get_tensor("layers.11.council_moe.w1_gamma")
    orig_numel = 512 * 896
    t = unpack_ternary_2bit(q[0], (512, 28, 32), orig_numel)
    w = t * g[0].float().unsqueeze(-1)
    print(
        f"unpacked expert0 w1: shape={tuple(w.shape)} absm={float(w.abs().mean()):.4f} max={float(w.abs().max()):.3f} nonzero={(w != 0).sum().item()}/{w.numel()}"
    )
