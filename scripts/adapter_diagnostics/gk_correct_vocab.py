#!/usr/bin/env python3
"""Correct GK measurement: derive tokenizer+letter_ids from the MODEL's own vocab, not hardcoded 151936."""

from __future__ import annotations
import sys
from pathlib import Path
import torch

ROOT = Path(r"C:\02_QUILLAN")
DIAG = ROOT / "scripts" / "adapter_diagnostics"
for p in (str(ROOT / "scripts"), str(ROOT / "03 - Training & Model"), str(DIAG)):
    if p not in sys.path:
        sys.path.insert(0, p)
from general_knowledge import QUESTIONS as GQ, KEY as GK, LETTERS
from wsp_math_eval import tok_for, load_model
from wsp_math_tune_v3 import enable_cache


def letter_ids_for(tok):
    # ' A',' B',' C',' D' -> single token ids regardless of tokenizer
    out = []
    for L in LETTERS:
        for cand in (f" {L}", L):
            e = tok.encode(cand, add_special_tokens=False)
            if len(e) == 1:
                out.append(e[0])
                break
        else:
            out.append(-1)
    return out


for name in [
    "quillan-mini-sovereign-bitnet.safetensors",
    "quillan-flagship-sovereign-bitnet.safetensors",
]:
    ck = ROOT / "exports" / name
    m = load_model(ck)
    enable_cache(m)
    vocab = m.cfg.vocab_size
    tok = tok_for(vocab)
    lids = letter_ids_for(tok)
    print(f"\n=== {name} vocab={vocab} letter_ids={lids} ===", flush=True)
    ok = 0
    pred = [0] * 4
    for i, (q, opts, _) in enumerate(GQ):
        p = (
            f"Question {i + 1}: {q}\n"
            + "".join(f"{L}) {o}\n" for L, o in zip(LETTERS, opts))
            + "Answer:"
        )
        ids = tok.encode(p, add_special_tokens=False)
        with torch.inference_mode():
            lg = m(torch.tensor([ids]))[0][0, -1, lids].float()
        pi = int(lg.argmax())
        pred[pi] += 1
        ok += int(pi == GK[i])
    print(
        f"GK={ok}/25 pred_dist={pred} true_dist={[sum(1 for k in GK if k == j) for j in range(4)]}",
        flush=True,
    )
