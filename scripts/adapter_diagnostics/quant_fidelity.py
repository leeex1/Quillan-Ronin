#!/usr/bin/env python3
"""Measure per-matrix reconstruction error of the 2-bit ternary packing.

0.99 global cosine can hide large relative error if element magnitudes are
small. Report relative Frobenius error and cosine for each projection type.
Also test how many planes are needed to reach a target fidelity.
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
from wsp_math_eval import load_model

torch.set_num_threads(1)


def rel_err(w, w2):
    return (w - w2).norm().item() / w.norm().clamp_min(1e-12).item()


def cos(w, w2):
    return torch.nn.functional.cosine_similarity(
        w.flatten().unsqueeze(0), w2.flatten().unsqueeze(0)
    ).item()


def main():
    m = load_model(ROOT / "exports" / "quillan-mini-sovereign-bitnet.safetensors")
    print("Per-matrix unpack fidelity (stored round-trip, no re-pack):", flush=True)
    for li in (0, 11, 23):
        for sub in (
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ):
            mod = getattr(m.layers[li], sub, None)
            if mod is None or not hasattr(mod, "unpack_effective_weight"):
                continue
            W = mod.unpack_effective_weight()
            g = mod.gamma.float()
            print(
                f"  L{li}.{sub:10s} shape={tuple(W.shape)} "
                f"gamma_mean={g.mean().item():.5f} gamma_std={g.std().item():.5f} "
                f"W_absmax={W.abs().max().item():.4f} "
                f"W_zerofrac={(W == 0).float().mean().item():.3f}",
                flush=True,
            )

    # test repack fidelity on a real matrix: unpack -> repack -> unpack
    mod = m.layers[11].o_proj
    W = mod.unpack_effective_weight().float()
    W2 = mod.unpack_effective_weight().float()  # already stored
    # repack into a fresh layer with 3 planes and compare
    from quillan_sovereign_bitnet import PackedBitLinear

    fresh = PackedBitLinear(
        mod.in_features,
        mod.out_features,
        group_size=mod.group_size,
        num_planes=3,
        eggroll_rank=0,
        use_rht=mod.use_rht,
    )
    fresh.set_weights_from_float(W2)
    W3 = fresh.unpack_effective_weight().float()
    print(
        f"\nRepack round-trip on L11.o_proj: rel_err={rel_err(W3, W2):.4f} "
        f"cos={cos(W3, W2):.4f} planes=3",
        flush=True,
    )


if __name__ == "__main__":
    main()
