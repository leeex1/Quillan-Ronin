#!/usr/bin/env python3
"""Exact-criterion gamma surgery: clip ONLY the outlier rows found by
weight_pathology_scan.py's own criterion.

Scan criterion was: rownorm = gamma.amax(dim=1) over groups; flag rows with
rownorm > 5x median(rownorm). My earlier gamma_surgery.py used median over ALL
gamma entries, which selected 28 unrelated entries. This script reproduces the
scan exactly and touches only the flagged rows, at a chosen clip multiple.
"""

from __future__ import annotations
import sys, argparse
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
BASE = ROOT / "exports" / "quillan-mini-sovereign-bitnet.safetensors"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", type=float, default=5.0)
    ap.add_argument(
        "--factor",
        type=float,
        default=0.8,
        help="scale flagged rows by this factor (0.8 = -20%%)",
    )
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    m = load_model(BASE)
    edits = []
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
            if mod is None or not hasattr(mod, "gamma"):
                continue
            g = mod.gamma.float()
            # gamma is (planes, out_features, groups); the scan flattened
            # shape[0]=planes, so "hi5x=1" means one PLANE is >5x the median.
            plane_stat = g.reshape(g.shape[0], -1).abs().amax(dim=1)
            med = plane_stat.median().clamp_min(1e-12)
            flagged = (plane_stat > a.clip * med).nonzero().flatten()
            if flagged.numel() == 0:
                continue
            with torch.no_grad():
                for r in flagged.tolist():
                    old = mod.gamma[r].float().clone()
                    new = (old * a.factor).to(torch.float16)
                    mod.gamma[r].copy_(new)
                    edits.append(
                        f"L{li}.{sub} plane={r} stat={plane_stat[r].item():.4g} "
                        f"x{a.factor} max {old.max().item():.4g}->{new.max().item():.4g}"
                    )

    print(f"clip={a.clip}x factor={a.factor} rows_edited={len(edits)}", flush=True)
    for e in edits:
        print("  " + e, flush=True)
    out = ROOT / "exports" / a.out
    m.save_packed_safetensors(out)
    print(f"SAVED {out}", flush=True)


if __name__ == "__main__":
    main()
