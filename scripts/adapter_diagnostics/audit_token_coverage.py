#!/usr/bin/env python3
"""Exact audit: how much of every donor's vocabulary actually survived the compile?"""

from __future__ import annotations
import sys, json
from pathlib import Path
import torch

ROOT = Path(r"C:\02_QUILLAN")
OUT = ROOT / "exports"
for p in (
    str(ROOT / "scripts"),
    str(ROOT / "03 - Training & Model"),
    str(ROOT / "scripts" / "adapter_diagnostics"),
):
    if p not in sys.path:
        sys.path.insert(0, p)


def main():
    meta = json.loads((OUT / "quillan_token_maps.json").read_text(encoding="utf-8"))
    d2q = meta["donor_to_quillan"]
    Q = meta["meta"]["canonical_vocab_size"]
    print(f"canonical Quillan slots = {Q}", flush=True)

    # coverage from the COMPILED embedding (authoritative)
    ck = torch.load(OUT / "quillan_embedding_compiled.pt", weights_only=True)
    cov = ck["coverage"]
    E = ck["embedding"]
    dist = torch.bincount(cov, minlength=5)
    print("\n=== compiled-embedding row coverage ===", flush=True)
    for k in range(5):
        print(f"  rows with {k} donor(s): {int(dist[k])}", flush=True)
    zero_rows = int((E.norm(dim=-1) == 0).sum())
    print(f"  ZERO/empty rows in compiled embedding: {zero_rows}", flush=True)

    print("\n=== per-donor native-token fate ===", flush=True)
    for name, m in d2q.items():
        ids = [int(v) for v in m.values()]
        distinct = len(set(ids))
        collisions = len(ids) - distinct
        print(
            f"  {name}: native_ids={len(ids)}  distinct_canonical_slots={distinct}  "
            f"collapsed/overwritten={collisions}",
            flush=True,
        )

    print("\n=== slots covered by how many donors (cross-check) ===", flush=True)
    q2d = torch.load(OUT / "quillan_token_maps_q2d.pt", weights_only=True)
    names = list(q2d.keys())
    cnt = sum((q2d[n] >= 0).int() for n in names)
    for k in range(5):
        print(f"  {k} donor(s): {int((cnt == k).sum())}", flush=True)


if __name__ == "__main__":
    main()
