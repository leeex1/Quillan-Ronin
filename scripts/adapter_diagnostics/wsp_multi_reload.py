#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fresh-reload validation: re-read checkpoint from disk N times in one process.

Each iteration constructs a NEW model from the on-disk checkpoint (genuine fresh
state), so this proves the saved weights reproduce the score. Avoids the ~5min
process startup that made 5-subprocess validation time out.
"""

from __future__ import annotations
import sys, gc
from pathlib import Path
import torch

ROOT = Path(r"C:\02_QUILLAN")
DIAG = ROOT / "scripts" / "adapter_diagnostics"
for p in (str(ROOT / "scripts"), str(ROOT / "03 - Training & Model"), str(DIAG)):
    if p not in sys.path:
        sys.path.insert(0, p)
from hard_math20 import KEY, format_problem, LETTERS
from hard_math40 import KEY as K40, PROBLEMS as P40
from hard_math20 import PROBLEMS as P20
from wsp_math_eval import tok_for, letter_ids, load_model


def score_once(ckpt, tok, lids, bench):
    m = load_model(ckpt)  # fresh from disk, new model object
    probs, gold = (P20, KEY) if bench == "20" else (P40, K40)
    ok = 0
    misses = []
    for i in range(len(gold)):
        q, opts = probs[i][0], probs[i][1]
        s = (
            f"Question {i + 1}: {q}\n"
            + "".join(f"{L}) {o}\n" for L, o in zip(LETTERS, opts))
            + "Answer:"
        )
        ids = tok.encode(s, add_special_tokens=False)
        with torch.inference_mode():
            lg = m(torch.tensor([ids]))[0][0, -1, :]
        p = max(range(4), key=lambda j: float(lg[lids[j]]))
        ok += int(p == gold[i])
        if p != gold[i]:
            misses.append(i + 1)
    del m
    gc.collect()
    return ok, misses


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--bench", default="20")
    a = ap.parse_args()
    tok = tok_for(151936)
    lids = letter_ids(tok, 151936)
    print(
        f"=== {a.runs} FRESH DISK RELOADS | {a.ckpt.name} | bench={a.bench} ===",
        flush=True,
    )
    results = []
    for i in range(a.runs):
        ok, miss = score_once(a.ckpt, tok, lids, a.bench)
        print(
            f"RELOAD #{i + 1}: {ok}/{len(KEY) if a.bench == '20' else len(K40)} misses={miss}",
            flush=True,
        )
        results.append(ok)
    target = 20 if a.bench == "20" else 40
    print(
        f"=== SUMMARY {results} | all_pass={all(r == target for r in results)} | min={min(results)} ===",
        flush=True,
    )


if __name__ == "__main__":
    main()
