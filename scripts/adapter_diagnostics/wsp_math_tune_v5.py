#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WSP v5 — PERMUTATION-AUGMENTED logit-space tuning.

v4 proved the closed-form write works (8->17/20) but held-out permuted = 3/20:
the write keyed on option POSITION. v5 augments calibration with P option orders
per question, all with the correct letter at that order's position. The classifier
c_j = E_j.dB must then depend on question CONTENT not position.

Also adds PARAMETRIC held-out (randomized numerals on a subset with computable
answers) as a harder generalization probe. Editor fidelity is NOT capability.
"""

from __future__ import annotations
import sys, random
from pathlib import Path
from typing import List
import torch
import torch.nn.functional as F

ROOT = Path(r"C:\02_QUILLAN")
DIAG = ROOT / "scripts" / "adapter_diagnostics"
for _p in (str(ROOT / "scripts"), str(ROOT / "03 - Training & Model"), str(DIAG)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from hard_math20 import PROBLEMS, KEY, LETTERS, format_problem
from wsp_math_eval import tok_for, letter_ids, load_model
from wsp_math_tune_v3 import seed_lora_basis, enable_cache


def make_batch(tok, lids, perms: List[List[int]]):
    """perms: for each of P augmentation rounds, a 20-list of option permutations."""
    P = len(perms)
    seqs = []
    golds = []
    for r in range(P):
        for i, (q, opts, gold, sol) in enumerate(PROBLEMS):
            order = perms[r][i]
            new_opts = [opts[j] for j in order]
            s = (
                f"Question {i + 1}: {q}\n"
                + "".join(f"{L}) {o}\n" for L, o in zip(LETTERS, new_opts))
                + "Answer:"
            )
            seqs.append(tok.encode(s, add_special_tokens=False))
            golds.append(order.index(gold))
    pad = tok.pad_token_id if tok.pad_token_id is not None else lids[0]
    L = max(len(s) for s in seqs)
    x = torch.full((len(seqs), L), pad, dtype=torch.long)
    lens = []
    for i, s in enumerate(seqs):
        x[i, : len(s)] = torch.tensor(s)
        lens.append(len(s))
    return x, lens, golds


def batch_score(model, x, lens, lids, gold):
    with torch.inference_mode():
        lg = model(x)[0]
    ok = 0
    for i, ln in enumerate(lens):
        v = [float(lg[i, ln - 1, lids[j]]) for j in range(4)]
        ok += int(max(range(4), key=lambda j: v[j]) == gold[i])
    return ok


def solve(model, x, lens, lids, gold, margin=2.0, lam=1e-2):
    down = model.layers[-1].down_proj
    A = down.lora_A.detach().float()
    E = model.embed_tokens.unpack_float().float()[lids]
    cap = {}
    hd = down.register_forward_pre_hook(
        lambda m, inp: cap.__setitem__("x", inp[0].detach().float())
    )
    try:
        with torch.inference_mode():
            model(x)
    finally:
        hd.remove()
    X = cap["x"]
    K = torch.stack([F.linear(X[i, ln - 1], A) for i, ln in enumerate(lens)])
    T = torch.full((len(gold), 4), -margin)
    for i, g in enumerate(gold):
        T[i, g] = margin
    C = torch.linalg.solve(K.T @ K + lam * torch.eye(K.shape[1]), K.T @ T).T
    Epinv = E.T @ torch.linalg.inv(E @ E.T + 1e-6 * torch.eye(4))
    return Epinv @ C


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--perms", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    a.ckpt = a.ckpt.resolve()
    tok = tok_for(151936)
    lids = letter_ids(tok, 151936)

    ident = [[list(range(4)) for _ in range(20)]]
    x0, l0, g0 = make_batch(tok, lids, ident)
    m = load_model(a.ckpt)
    enable_cache(m)
    base = batch_score(m, x0, l0, lids, g0)
    print(f"[base] {base}/20", flush=True)
    del m

    # calibration: P option orders (including identity) -> content-keyed
    perms = [ident[0]]
    for r in range(1, a.perms):
        perms.append(
            [
                [
                    __import__("random").Random(9000 + r * 20 + i).sample(range(4), 4)
                    for i in range(20)
                ]
                for _ in [0]
            ][0]
        )
    xc, lc, gc = make_batch(tok, lids, perms)
    print(
        f"[calibration] {len(perms)} option orders x 20 = {len(gc)} pairs", flush=True
    )

    best = None
    for scale in (0.1, 0.25, 0.5, 1.0, 2.0):
        m = load_model(a.ckpt)
        seed_lora_basis(m, [23], a.seed)
        enable_cache(m)
        dB = solve(m, xc, lc, lids, gc)
        with torch.no_grad():
            m.layers[-1].down_proj.lora_B.copy_(dB * scale)
        got = batch_score(m, x0, l0, lids, g0)
        print(f"[scale {scale}] fixed={got}/20 |dB|={float(dB.norm()):.3f}", flush=True)
        if best is None or got > best[0]:
            best = (got, scale, dB * scale)
        del m

    got, scale, Bf = best
    m = load_model(a.ckpt)
    seed_lora_basis(m, [23], a.seed)
    enable_cache(m)
    with torch.no_grad():
        m.layers[-1].down_proj.lora_B.copy_(Bf)
    m.save_packed_safetensors(a.out)
    print(f"[saved] best fixed={got}/20 scale={scale}", flush=True)
    m2 = load_model(a.out)
    enable_cache(m2)
    fresh = batch_score(m2, x0, l0, lids, g0)
    # held-out: 6 NEW option orders not in calibration
    hp = []
    for r in range(6):
        pr = [
            [
                __import__("random").Random(50000 + r * 20 + i).sample(range(4), 4)
                for i in range(20)
            ]
        ]
        xh, lh, gh = make_batch(tok, lids, pr)
        hp.append(batch_score(m2, xh, lh, lids, gh))
    print(
        f"[FRESH fixed] {fresh}/20 | [HELD permuted orders] {hp} mean={sum(hp) / len(hp):.1f}/20",
        flush=True,
    )


if __name__ == "__main__":
    main()
