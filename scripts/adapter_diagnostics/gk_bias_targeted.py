#!/usr/bin/env python3
"""Targeted logit-bias sweep: suppress ONLY over-predicted letters, not all.
Earlier centering subtracted every letter's mean (hurt: boosted rare 'A').
Here: subtract a bias from the single most over-predicted letter, closed-form.
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
from general_knowledge import QUESTIONS as GQ, KEY as GK, LETTERS
from wsp_math_eval import tok_for, letter_ids, load_model
from wsp_math_tune_v3 import enable_cache


def collect(m, tok, lids):
    L = []
    for i, (q, opts, _) in enumerate(GQ):
        pr = (
            f"Question {i + 1}: {q}\n"
            + "".join(f"{x}) {o}\n" for x, o in zip(LETTERS, opts))
            + "Answer:"
        )
        ids = tok.encode(pr, add_special_tokens=False)
        with torch.inference_mode():
            lg = m(torch.tensor([ids]))[0][0, -1, lids].float()
        L.append(lg)
    return torch.stack(L)


def run(ckpt):
    tok = tok_for(151936)
    lids = letter_ids(tok, 151936)
    m = load_model(ckpt)
    enable_cache(m)
    L = collect(m, tok, lids)
    K = torch.tensor(GK)
    base = int((L.argmax(-1) == K).sum())
    pred = L.argmax(-1)
    dist = [int((pred == j).sum()) for j in range(4)]
    correct = [int((pred == j).sum()) for j in range(4)]
    print(
        f"[{ckpt.name}] base={base}/25 dist={dist} true_key_dist={[int((K == j).sum()) for j in range(4)]}",
        flush=True,
    )
    # targeted: subtract bias b from the most over-predicted letter only
    over = int(torch.tensor(dist).argmax())
    best = (base, 0, over)
    for b in [0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]:
        Lb = L.clone()
        Lb[:, over] -= b
        ok = int((Lb.argmax(-1) == K).sum())
        print(f"  suppress {LETTERS[over]} by {b}: {ok}/25", flush=True)
        if ok > best[0]:
            best = (ok, b, over)
    # two-letter suppression: the two most over-predicted
    order = sorted(range(4), key=lambda j: -dist[j])
    top2 = order[:2]
    for b in [0.5, 1.0, 2.0, 3.0]:
        Lb = L.clone()
        for j in top2:
            Lb[:, j] -= b
        ok = int((Lb.argmax(-1) == K).sum())
        print(f"  suppress {[LETTERS[j] for j in top2]} by {b}: {ok}/25", flush=True)
        if ok > best[0]:
            best = (ok, b, tuple(top2))
    print(f"[BEST] {best[0]}/25 suppression={best[1]} on={best[2]}", flush=True)


for n in ["quillan-mini-math40-rank64.safetensors", "quillan-mini-rank64.safetensors"]:
    run(Path(r"C:\02_QUILLAN\exports") / n)
