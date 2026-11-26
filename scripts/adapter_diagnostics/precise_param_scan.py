#!/usr/bin/env python3
"""PRECISION SCAN: find the specific broken parameters, not whole matrices.
For each packed linear, unpack to fp32 and report per-output-row norm outliers
and per-input-feature magnitude outliers. These are the surgical targets.
No edits, no training.
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

torch.set_num_threads(2)


def robust_stats(t, name):
    a = t.reshape(t.shape[0], -1)
    rn = a.float().norm(dim=1)
    med = rn.median()
    mad = (rn - med).abs().median().clamp_min(1e-9)
    z = (rn - med) / (1.4826 * mad)
    hi = (z > 8).sum().item()
    lo = (z < -8).sum().item()
    return med.item(), z.max().item(), z.min().item(), hi, lo


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--ck", required=True)
    ap.add_argument("--top", type=int, default=12)
    a = ap.parse_args()
    m = load_model(ROOT / "exports" / a.ck)
    rows = []
    for li, layer in enumerate(m.layers):
        for sub in (
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ):
            mod = getattr(layer, sub, None)
            if mod is None or not hasattr(mod, "unpack_effective_weight"):
                continue
            W = mod.unpack_effective_weight().float()
            med, zmax, zmin, hi, lo = robust_stats(W, f"L{li}.{sub}")
            rows.append(
                (
                    max(abs(zmax), abs(zmin)),
                    li,
                    sub,
                    med,
                    zmax,
                    zmin,
                    hi,
                    lo,
                    tuple(W.shape),
                )
            )
    rows.sort(reverse=True)
    print(
        f"{'tensor':22s} {'shape':16s} {'med':>8s} {'zmax':>8s} {'zmin':>8s} {'hi':>5s} {'lo':>5s}"
    )
    for z, li, sub, med, zmax, zmin, hi, lo, shape in rows[: a.top]:
        print(
            f"L{li:02d}.{sub:16s} {str(shape):16s} {med:8.4f} {zmax:8.1f} {zmin:8.1f} {hi:5d} {lo:5d}"
        )
    print(f"\ntotal tensors scanned={len(rows)}")


if __name__ == "__main__":
    main()
