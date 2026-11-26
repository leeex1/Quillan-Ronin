#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WSP math tuner v4 — CORRECT closed-form derivation in LOGIT space.

Why v3 failed: it mapped the target through an incorrect RMSNorm Jacobian, so
the ridge target lived in the wrong space (orthogonal to the logits).

v4 derivation (exact, no backprop):
  tied head:  logit_j = E_j . h            E_j = embedding row of letter j
  LoRA write: h += dB . k_i                k_i = lora_A . x_i  (16-dim), dB=[896,16]
  => shift_ij = (E_j . dB) . k_i = c_j . k_i,  where c_j = E_j . dB in R^16.

So the 4 letters are a LINEAR CLASSIFIER over k_i. Fit C=[c_0..c_3]^T (4x16):
  T_ij = +margin if j==gold else -margin        (20x4 target shifts)
  C = (K^T K + lam I)^-1 K^T T                  (4x16 closed-form ridge)
Then lift C back to dB via the embedding pseudoinverse:
  dB = E^+ . C    where E (4x896) -> E^+ = E^T (E E^T)^-1 ; E . dB = C exactly.

Fresh-reload proof included. Held-out permutation arm included (position check).
Editor fidelity only — NOT reasoning (C17 #1279 / C18 #1282).
"""

from __future__ import annotations
import sys
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
from wsp_math_tune_v3 import seed_lora_basis, enable_cache, batch_inputs, batch_score


def solve_logit_space(model, x, lens, lids, gold, margin=2.0, lam=1e-2):
    down = model.layers[-1].down_proj
    A = down.lora_A.detach().float()  # [16,4864]
    E = model.embed_tokens.unpack_float().float()[lids]  # [4,896]
    captured = {}
    hd = down.register_forward_pre_hook(
        lambda m, inp: captured.__setitem__("x", inp[0].detach().float())
    )
    try:
        with torch.inference_mode():
            model(x)
    finally:
        hd.remove()
    X = captured["x"]
    Ks = []
    for i, ln in enumerate(lens):
        k = F.linear(X[i, ln - 1], A)  # [16]
        Ks.append(k)
    K = torch.stack(Ks)  # [20,16]
    T = torch.full((20, 4), -margin)
    for i in range(20):
        T[i, gold[i]] = margin
    ridge = K.T @ K + lam * torch.eye(K.shape[1])
    C = torch.linalg.solve(ridge, K.T @ T).T  # [4,16]  (c_j)
    # dB = E^+ C : E@dB = C exactly
    Epinv = E.T @ torch.linalg.inv(E @ E.T + 1e-6 * torch.eye(4))  # [896,4]
    dB = Epinv @ C  # [896,16]
    return dB


def permuted_inputs(tok, lids, seed=777):
    import random

    seqs = []
    golds = []
    for i, (q, opts, gold, sol) in enumerate(PROBLEMS):
        order = list(range(4))
        random.Random(seed + i).shuffle(order)
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
    x = torch.full((20, L), pad, dtype=torch.long)
    lens = []
    for i, s in enumerate(seqs):
        x[i, : len(s)] = torch.tensor(s)
        lens.append(len(s))
    return x, lens, golds


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    a.ckpt = a.ckpt.resolve()
    tok = tok_for(151936)
    lids = letter_ids(tok, 151936)
    x, lens = batch_inputs(tok, lids)
    xp, lp, gp = permuted_inputs(tok, lids)

    m = load_model(a.ckpt)
    enable_cache(m)
    base, _ = batch_score(m, x, lens, lids)
    bpred = None
    print(f"[base fixed] {base}/20", flush=True)
    del m

    best = None
    for scale in (0.1, 0.25, 0.5, 1.0, 2.0):
        m = load_model(a.ckpt)
        seed_lora_basis(m, [23], a.seed)
        enable_cache(m)
        dB = solve_logit_space(m, x, lens, lids, KEY, margin=2.0)
        frob = float(dB.norm())
        with torch.no_grad():
            m.layers[-1].down_proj.lora_B.copy_(dB * scale)
        got, pred = batch_score(m, x, lens, lids)
        print(f"[scale {scale}] fixed={got}/20 |dB|={frob:.3f}", flush=True)
        if best is None or got > best[0]:
            best = (got, scale, dB * scale)
        del m
        if got == 20:
            break

    got, scale, Bfinal = best
    m = load_model(a.ckpt)
    seed_lora_basis(m, [23], a.seed)
    enable_cache(m)
    with torch.no_grad():
        m.layers[-1].down_proj.lora_B.copy_(Bfinal)
    m.save_packed_safetensors(a.out)
    print(f"[saved] best={got}/20 scale={scale}", flush=True)
    m2 = load_model(a.out)
    enable_cache(m2)
    fresh, _ = batch_score(m2, x, lens, lids)
    held, _ = batch_score(m2, xp, lp, lids)
    # held scoring against permuted gold
    with torch.inference_mode():
        lg = m2(xp)[0]
    hp = 0
    for i, ln in enumerate(lp):
        v = [float(lg[i, ln - 1, lids[j]]) for j in range(4)]
        hp += int(max(range(4), key=lambda j: v[j]) == gp[i])
    print(f"[FRESH fixed] {fresh}/20 | [HELD permuted] {hp}/20", flush=True)
    print(f"[GENERALIZATION GAP] {fresh - hp}", flush=True)


if __name__ == "__main__":
    main()
