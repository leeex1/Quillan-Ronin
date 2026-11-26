#!/usr/bin/env python3
"""Controlled A/B on the larger pre-locked bank (80 q, 0 overlap with gates).
Pinned 1 thread, paired per-item. base vs frozen-attention, plus optional ck2."""

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
from wsp_math_eval import tok_for, letter_ids, load_model
from wsp_math_tune_v3 import enable_cache
from heldout_domain_large import BANKS, prompts

torch.set_num_threads(1)


def score(m, tok, lids, dom):
    Q, K = BANKS[dom]
    bits = []
    with torch.inference_mode():
        for seq, gold in zip(prompts(tok, dom), K):
            lg = m(torch.tensor([seq]))[0][0, -1, lids].float()
            bits.append(int(int(lg.argmax()) == gold))
    return bits


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    a = ap.parse_args()
    A = load_model(ROOT / "exports" / a.a)
    enable_cache(A)
    B = load_model(ROOT / "exports" / a.b)
    enable_cache(B)
    tok = tok_for(A.cfg.vocab_size)
    lids = letter_ids(tok, A.cfg.vocab_size)
    tf = tb = 0
    fixed = broke = 0
    for dom in BANKS:
        x = score(A, tok, lids, dom)
        y = score(B, tok, lids, dom)
        f = sum(1 for p, q in zip(x, y) if p == 0 and q == 1)
        br = sum(1 for p, q in zip(x, y) if p == 1 and q == 0)
        tf += sum(x)
        tb += sum(y)
        fixed += f
        broke += br
        print(
            f"{dom:10s} {sum(x)}->{sum(y)} gain={sum(y) - sum(x):+d} (fix={f},brk={br})",
            flush=True,
        )
    print(
        f"POOLED {tf}->{tb} gain={tb - tf:+d} fixed={fixed} broke={broke}", flush=True
    )


if __name__ == "__main__":
    main()
