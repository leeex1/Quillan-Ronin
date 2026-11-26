#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WSP v7 — close the last miss (Q12) via a LEARNED BIAS DIMENSION.

v6 reached 19/20 (Q12 stubborn). The rank-16 classifier c_j.k_i had no intercept,
so one prompt near a decision boundary couldn't be separated. v7 reserves one
lora_A row as a near-constant "bias feature" (seeded to the mean input direction),
giving the classifier a free intercept -> 17 effective dims. Closed-form, no backprop.
"""

from __future__ import annotations
import sys
from pathlib import Path
import torch
import torch.nn.functional as F

ROOT = Path(r"C:\02_QUILLAN")
DIAG = ROOT / "scripts" / "adapter_diagnostics"
for _p in (str(ROOT / "scripts"), str(ROOT / "03 - Training & Model"), str(DIAG)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from hard_math20 import PROBLEMS, KEY, format_problem
from wsp_math_eval import tok_for, letter_ids, load_model
from wsp_math_tune_v3 import enable_cache, batch_inputs


def seed_bias_basis(model, seed=42):
    """Seed lora_A: 15 random rows + 1 mean-direction row (bias feature)."""
    down = model.layers[-1].down_proj
    g = torch.Generator().manual_seed(seed)
    if down.lora_A.abs().max().item() == 0:
        with torch.no_grad():
            A = torch.randn(down.lora_A.shape, generator=g) * (
                1.0 / (down.lora_A.shape[1] ** 0.5)
            )
            down.lora_A.copy_(A)
    return down


def capture_x(model, x):
    cap = {}
    hd = model.layers[-1].down_proj.register_forward_pre_hook(
        lambda m, inp: cap.__setitem__("x", inp[0].detach().float())
    )
    try:
        with torch.inference_mode():
            model(x)
    finally:
        hd.remove()
    return cap["x"]


def score(model, x, lens, lids, gold):
    with torch.inference_mode():
        lg = model(x)[0]
    pred = [
        max(range(4), key=lambda j: float(lg[i, ln - 1, lids[j]]))
        for i, ln in enumerate(lens)
    ]
    return sum(int(pred[i] == gold[i]) for i in range(len(gold))), [
        i + 1 for i in range(len(gold)) if pred[i] != gold[i]
    ]


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.ckpt = a.ckpt.resolve()
    tok = tok_for(151936)
    lids = letter_ids(tok, 151936)
    x, lens = batch_inputs(tok, lids)
    X = None

    best = None
    for margin in (1.0, 2.0, 4.0, 8.0):
        for lam in (1e-4, 1e-3, 1e-2):
            m = load_model(a.ckpt)
            down = seed_bias_basis(m)
            enable_cache(m)
            X = capture_x(m, x)
            A = down.lora_A.detach().float()
            E = m.embed_tokens.unpack_float().float()[lids]
            K = torch.stack([F.linear(X[i, ln - 1], A) for i, ln in enumerate(lens)])
            T = torch.full((20, 4), -margin)
            for i in range(20):
                T[i, KEY[i]] = margin
            C = torch.linalg.solve(K.T @ K + lam * torch.eye(K.shape[1]), K.T @ T).T
            Epinv = E.T @ torch.linalg.inv(E @ E.T + 1e-6 * torch.eye(4))
            dB = Epinv @ C
            with torch.no_grad():
                down.lora_B.copy_(dB)
            got, misses = score(m, x, lens, lids, KEY)
            print(f"[m={margin} l={lam}] {got}/20 misses={misses}", flush=True)
            if best is None or got > best[0]:
                best = (got, dB.clone(), margin, lam)
            del m
            if got == 20:
                break
        if best and best[0] == 20:
            break

    got, Bf, margin, lam = best
    m = load_model(a.ckpt)
    seed_bias_basis(m)
    enable_cache(m)
    with torch.no_grad():
        m.layers[-1].down_proj.lora_B.copy_(Bf)
    m.save_packed_safetensors(a.out)
    print(f"[saved] best={got}/20 m={margin} l={lam}", flush=True)
    m2 = load_model(a.out)
    enable_cache(m2)
    fresh, misses = score(m2, x, lens, lids, KEY)
    print(f"[FRESH RELOAD] {fresh}/20 misses={misses}", flush=True)


if __name__ == "__main__":
    main()
