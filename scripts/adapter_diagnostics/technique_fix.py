#!/usr/bin/env python3
"""'Better running technique, no training' test.

Hypothesis: the model has the factual knowledge but a systematic OUTPUT BIAS
(it over-predicts one answer letter - generation collapses to 'C D D D...').
Fixing that bias is a closed-form technique correction: no new knowledge,
no gradient, no label fitting (unsupervised - never reads the answer key).

Stage 1: measure letter logit bias + accuracy with/without centering (in logit space).
Stage 2: if it helps, express the correction as a tensor write (lm_head bias /
         rank-1 head edit) and verify on a fresh reload.
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


def prompts_lids(tok, lids):
    ps = []
    for i, (q, opts, _) in enumerate(GQ):
        ps.append(
            f"Question {i + 1}: {q}\n"
            + "".join(f"{L}) {o}\n" for L, o in zip(LETTERS, opts))
            + "Answer:"
        )
    ids = [tok.encode(p, add_special_tokens=False) for p in ps]
    return ids


def batch_logits(m, ids, lids):
    # group by length (no padding -> valid for this engine)
    g = {}
    for i, x in enumerate(ids):
        g.setdefault(len(x), []).append((i, x))
    L = [None] * len(ids)
    for _, items in g.items():
        with torch.inference_mode():
            out = m(torch.tensor([x for _, x in items]))[0]
        for j, (orig, _) in enumerate(items):
            L[orig] = out[j, -1, lids].float()
    return torch.stack(L)  # [n,4]


def report(tag, logits, gk):
    pred = logits.argmax(-1)
    ok = int((pred == torch.tensor(gk)).sum())
    dist = [int((pred == k).sum()) for k in range(4)]
    print(
        f"[{tag}] accuracy={ok}/{len(gk)} ({100 * ok / len(gk):.1f}%) pred_dist={dist}",
        flush=True,
    )
    return ok


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--ckpt",
        type=Path,
        default=Path(r"C:\02_QUILLAN\exports\quillan-mini-rank64.safetensors"),
    )
    ap.add_argument(
        "--train-ckpt",
        type=Path,
        default=Path(r"C:\02_QUILLAN\exports\quillan-mini-math40-rank64.safetensors"),
    )
    a = ap.parse_args()
    tok = tok_for(151936)
    lids = letter_ids(tok, 151936)
    ids = prompts_lids(tok, lids)
    for tag, ck in [("base", a.ckpt), ("math40", a.train_ckpt)]:
        m = load_model(ck)
        enable_cache(m)
        L = batch_logits(m, ids, lids)
        print(f"\n=== {ck.name} ===", flush=True)
        print(
            f"[raw logits] mean={[round(float(x), 2) for x in L.mean(0)]} std={[round(float(x), 2) for x in L.std(0)]}",
            flush=True,
        )
        report(f"{tag} BASE", L, GK)
        # UNSUPERVISED bias correction: subtract each letter's mean logit
        c = L.mean(0)
        for lam in [0.5, 1.0, 1.5]:
            report(f"{tag} centering x{lam}", L - lam * c, GK)
        # per-letter scale normalization (z-score) - also unsupervised
        z = (L - L.mean(0)) / (L.std(0) + 1e-6)
        report(f"{tag} z-score", z, GK)
        del m


if __name__ == "__main__":
    main()
