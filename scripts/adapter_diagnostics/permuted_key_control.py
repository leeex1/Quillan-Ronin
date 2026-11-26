#!/usr/bin/env python3
"""Permuted-KEY control: prove the repair gain is content-driven, not a
letter-prior drift. For each bank, score base and repaired under the TRUE key
and under several SHUFFLED keys. Real knowledge => true score >> shuffled score
for both models; a pure letter-bias model would score similarly on shuffled
keys (any fixed letter guess matches a shuffled key ~25%).
"""

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
from heldout_fresh40c import prompts as p3, KEY as K3

torch.set_num_threads(1)
BANKS = [("fresh40", p1, K1), ("fresh40b", p2, K2), ("fresh40c", p3, K3)]


def preds(m, tok, lids, pf):
    out = []
    with torch.inference_mode():
        for seq in pf(tok):
            lg = m(torch.tensor([seq]))[0][0, -1, lids].float()
            out.append(int(lg.argmax()))
    return out


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--ck", required=True)
    ap.add_argument("--label", default="MODEL")
    ap.add_argument("--seeds", type=int, default=3)
    a = ap.parse_args()
    m = load_model(Path(a.ck))
    tok = tok_for(m.cfg.vocab_size)
    lids = letter_ids(tok, m.cfg.vocab_size)
    for name, pf, KEY in BANKS:
        pr = preds(m, tok, lids, pf)
        true = sum(1 for p, k in zip(pr, KEY) if p == k)
        sham = []
        for s in range(a.seeds):
            g = torch.Generator().manual_seed(100 + s)
            perm = torch.randperm(len(KEY), generator=g).tolist()
            shuf = [KEY[i] for i in perm]
            sham.append(sum(1 for p, k in zip(pr, shuf) if p == k))
        print(
            f"[{a.label}] {name}: TRUE={true}/40 shuffled={sham} "
            f"(shuf_mean={sum(sham) / len(sham):.1f})",
            flush=True,
        )


if __name__ == "__main__":
    main()
