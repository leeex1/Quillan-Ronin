#!/usr/bin/env python3
"""Single-model per-letter confusion matrix. Base already captured in perletter.txt."""

from __future__ import annotations
import sys
from pathlib import Path
import torch

ROOT = Path(r"C:\02_QUILLAN")
DIAG = ROOT / "scripts" / "adapter_diagnostics"
for p in (str(ROOT / "scripts"), str(ROOT / "03 - Training & Model"), str(DIAG)):
    if p not in sys.path:
        sys.path.insert(0, p)
from wsp_math_eval import tok_for, letter_ids, load_model
from heldout_fresh40 import prompts as p1, KEY as K1
from heldout_fresh40b import prompts as p2, KEY as K2

torch.set_num_threads(1)


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--ck", required=True)
    ap.add_argument("--label", default="MODEL")
    a = ap.parse_args()
    m = load_model(Path(a.ck))
    tok = tok_for(m.cfg.vocab_size)
    lids = letter_ids(tok, m.cfg.vocab_size)
    for bname, pf, KEY in (("fresh40", p1, K1), ("fresh40b", p2, K2)):
        ids = pf(tok)
        pred = []
        with torch.inference_mode():
            for seq in ids:
                lg = m(torch.tensor([seq]))[0][0, -1, lids].float()
                pred.append(int(lg.argmax()))
        print(f"\n[{a.label}] {bname}", flush=True)
        print("  true\\pred   A   B   C   D | recall", flush=True)
        for t in range(4):
            row = [
                sum(1 for p, k in zip(pred, KEY) if k == t and p == j) for j in range(4)
            ]
            tot = sum(row)
            print(
                f"   {'ABCD'[t]}       "
                + " ".join(f"{v:3d}" for v in row)
                + f" | {row[t]:2d}/{tot}",
                flush=True,
            )
        print(
            "  predicted: "
            + " ".join(f"{'ABCD'[j]}={pred.count(j)}" for j in range(4)),
            flush=True,
        )
        print(
            f"  ACCURACY={sum(1 for p, k in zip(pred, KEY) if p == k)}/{len(KEY)}",
            flush=True,
        )


if __name__ == "__main__":
    main()
