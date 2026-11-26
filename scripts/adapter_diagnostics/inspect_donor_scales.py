#!/usr/bin/env python3
"""Inspect donor tensor structure + per-tensor scales. This is the user's point:
each donor has a different numeric scale (e.g. Qwen ~3.2 vs Falcon ~4.3);
raw merging makes them clash. Measure it before aligning/fusing.
"""

from __future__ import annotations
from pathlib import Path
import torch

D = Path(r"C:\02_QUILLAN\exports\donor_cache_v23")
FILES = [
    "qwen_0.5b_donors.pt",
    "llama_1b_donors.pt",
    "falcon3_1b_donors.pt",
    "bitnet_2b4t_donors.pt",
]


def load(p):
    try:
        return torch.load(p, map_location="cpu", weights_only=True)
    except Exception:
        return torch.load(p, map_location="cpu")


def main():
    stores = {}
    for f in FILES:
        p = D / f
        if not p.exists():
            print(f"MISSING {f}", flush=True)
            continue
        d = load(p)
        stores[f] = d
        if isinstance(d, dict):
            keys = list(d.keys())
            print(f"\n{f}: {len(keys)} keys; sample={keys[:5]}", flush=True)
            # tensors only
            tk = [k for k in keys if torch.is_tensor(d[k])]
            print(f"  tensor entries={len(tk)}", flush=True)
        else:
            print(f"\n{f}: type={type(d)}", flush=True)

    # For common keys, report norms/means (the "3.2 vs 4.3" scale signature)
    dicts = {f: d for f, d in stores.items() if isinstance(d, dict)}
    if not dicts:
        return
    common = set.intersection(*[set(d.keys()) for d in dicts.values()])
    common = [k for k in common if all(torch.is_tensor(d[k]) for d in dicts.values())]
    print(f"\ncommon tensor keys across all donors: {len(common)}", flush=True)
    for k in sorted(common)[:12]:
        parts = []
        for f, d in dicts.items():
            t = d[k].float()
            parts.append(
                f"{f.split('_')[0]}:n={t.norm().item():.3f},mean={t.mean().item():.4f},max={t.abs().max().item():.3f}"
            )
        print(f"  {k}: " + " | ".join(parts), flush=True)


if __name__ == "__main__":
    main()
