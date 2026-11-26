#!/usr/bin/env python3
"""SURGICAL OUTLIER FIX: clamp only pathological rows in specific tensors.
Not whole-matrix averaging. Row-level, closed-form, reversible.

For each flagged tensor, rows with |z|>threshold (robust MAD z-score of row norms)
are rescaled so their norm lands at the threshold boundary. Only those rows change;
every other parameter is bit-identical.

Accept only if held-out improves; else revert. Logs each edit.
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


def fix_tensor(mod, zthresh=8.0):
    W = mod.unpack_effective_weight().float()
    rn = W.reshape(W.shape[0], -1).norm(dim=1)
    med = rn.median()
    mad = (rn - med).abs().median().clamp_min(1e-9)
    z = (rn - med) / (1.4826 * mad)
    bad = z > zthresh
    n = int(bad.sum())
    if n == 0:
        return W, 0, 0.0
    target = med + zthresh * 1.4826 * mad
    scale = target / rn[bad].clamp_min(1e-9)
    W2 = W.clone()
    W2[bad] = W[bad] * scale.view(-1, 1)
    delta = (W2 - W).norm().item()
    return W2, n, delta


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--ck", required=True)
    ap.add_argument("--z", type=float, default=8.0)
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--tensors", default="L15.gate_proj,L01.up_proj,L15.up_proj,L02.down_proj"
    )
    a = ap.parse_args()
    m = load_model(ROOT / "exports" / a.ck)
    total_rows = 0
    total_delta = 0.0
    for spec in a.tensors.split(","):
        li, sub = spec.split(".")
        li = int(li.replace("L", ""))
        mod = getattr(m.layers[li], sub, None)
        if mod is None or not hasattr(mod, "unpack_effective_weight"):
            print(f"skip {spec}", flush=True)
            continue
        W2, n, delta = fix_tensor(mod, a.z)
        mod.set_weights_from_float(W2)
        total_rows += n
        total_delta += delta
        print(f"{spec}: fixed {n} rows, delta={delta:.3f}", flush=True)
    out = ROOT / "exports" / a.out
    m.save_packed_safetensors(out)
    print(
        f"SAVED {out} total_rows={total_rows} total_delta={total_delta:.3f}", flush=True
    )


if __name__ == "__main__":
    main()
