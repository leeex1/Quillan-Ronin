#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Score a WSP checkpoint in ONE fresh process (no accumulation).

Called N times by wsp_multi_fresh.py, each invocation is an independent fresh
reload from disk — this is the '20/20 on multiple fresh runs' proof the user
requires. Prints one line: RUN result.
"""

from __future__ import annotations
import sys
from pathlib import Path
import torch

ROOT = Path(r"C:\02_QUILLAN")
DIAG = ROOT / "scripts" / "adapter_diagnostics"
for _p in (str(ROOT / "scripts"), str(ROOT / "03 - Training & Model"), str(DIAG)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from hard_math20 import KEY, PROBLEMS, format_problem, LETTERS
from hard_math40 import KEY as KEY40, PROBLEMS as PROBLEMS40
from wsp_math_eval import tok_for, letter_ids, load_model


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--bench", choices=["20", "40"], default="20")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    tok = tok_for(151936)
    lids = letter_ids(tok, 151936)
    m = load_model(a.ckpt)  # fresh from disk
    if a.bench == "20":
        probs, gold = PROBLEMS, KEY
    else:
        probs, gold = PROBLEMS40, KEY40
    ok = 0
    misses = []
    for i in range(len(gold)):
        s = f"Question {i + 1}: {probs[i][0]}\n"
        for L, o in zip(LETTERS, probs[i][1]):
            s += f"{L}) {o}\n"
        s += "Answer:"
        ids = tok.encode(s, add_special_tokens=False)
        with torch.inference_mode():
            lg = m(torch.tensor([ids]))[0][0, -1, :]
        p = max(range(4), key=lambda j: float(lg[lids[j]]))
        ok += int(p == gold[i])
        if p != gold[i]:
            misses.append(i + 1)
    print(f"RUN {a.tag} {a.bench}: {ok}/{len(gold)} misses={misses}", flush=True)


if __name__ == "__main__":
    main()
