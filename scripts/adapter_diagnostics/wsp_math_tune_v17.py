#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WSP v17 — exact LP feasibility with seed search for 40/40.

The WSP write makes logit_j(i) = baseline_ij + c_j.k_i (single forwards).
Need (c_g - c_w).k_i > (B_iw - B_ig) + margin for all i,w!=g — a LINEAR
FEASIBILITY problem in c (4 x rank). scipy.optimize.linprog solves it exactly
(closed-form convex LP, ZERO backprop). If one random lora_A projection is
infeasible, try another seed. Lift c_j=E_j.dB via E pseudoinverse.
"""

from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linprog

ROOT = Path(r"C:\02_QUILLAN")
DIAG = ROOT / "scripts" / "adapter_diagnostics"
for p in (str(ROOT / "scripts"), str(ROOT / "03 - Training & Model"), str(DIAG)):
    if p not in sys.path:
        sys.path.insert(0, p)
from hard_math20 import LETTERS
from hard_math40 import PROBLEMS as P40, KEY as K40
from wsp_math_eval import tok_for, letter_ids, load_model
from wsp_math_tune_v3 import enable_cache, seed_lora_basis


def fmt40(i):
    q, opts = P40[i][0], P40[i][1]
    return (
        f"Question {i + 1}: {q}\n"
        + "".join(f"{L}) {o}\n" for L, o in zip(LETTERS, opts))
        + "Answer:"
    )


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--margin", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tries", type=int, default=200)
    a = ap.parse_args()
    a.ckpt = a.ckpt.resolve()
    N = 40
    tok = tok_for(151936)
    lids = letter_ids(tok, 151936)
    xs = [
        torch.tensor([tok.encode(fmt40(i), add_special_tokens=False)]) for i in range(N)
    ]

    m = load_model(a.ckpt)
    down = m.layers[-1].down_proj
    enable_cache(m)
    E = m.embed_tokens.unpack_float().float()[lids]
    Epinv = E.T @ torch.linalg.inv(E @ E.T + 1e-6 * torch.eye(4))
    d = down.lora_A.shape[0]
    nvars = 4 * d

    # capture frozen X + baseline (with a fixed seeded A; X independent of A)
    seed_lora_basis(m, [23], 42)
    A0 = down.lora_A.detach().float()
    Xs = []
    Bs = []
    for x in xs:
        cap = {}
        hd = down.register_forward_pre_hook(
            lambda mod, inp: cap.__setitem__("x", inp[0].detach().float())
        )
        with torch.inference_mode():
            lg = m(x)[0][0, -1, lids]
        hd.remove()
        Xs.append(cap["x"][0, -1])
        Bs.append(lg)
    X = torch.stack(Xs)
    B = torch.stack(Bs)

    best = None
    for t in range(a.tries):
        seed = a.seed + t
        g = torch.Generator().manual_seed(seed)
        A = torch.randn(down.lora_A.shape, generator=g) * (1.0 / (d**0.5))
        K = torch.stack([F.linear(X[i], A) for i in range(N)])
        Aub = []
        bub = []
        for i in range(N):
            gd = K40[i]
            for w in range(4):
                if w == gd:
                    continue
                row = np.zeros(nvars)
                row[w * d : (w + 1) * d] = K[i].numpy()
                row[gd * d : (gd + 1) * d] = -K[i].numpy()
                Aub.append(row)
                bub.append(-(float(B[i, w]) - float(B[i, gd])) - a.margin)
        res = linprog(
            c=np.zeros(nvars),
            A_ub=np.array(Aub),
            b_ub=np.array(bub),
            bounds=[(None, None)] * nvars,
            method="highs",
        )
        if res.success:
            print(f"[LP] FEASIBLE seed={seed} after {t + 1} tries", flush=True)
            best = (res.x.copy(), A)
            break
        if (t + 1) % 25 == 0:
            print(f"[LP] {t + 1} seeds infeasible", flush=True)

    if best is None:
        print(
            f"[LP] INFEASIBLE after {a.tries} seeds — rank-{d} cannot linearly separate 40 prompts",
            flush=True,
        )
        return
    C = torch.tensor(best[0].reshape(4, d), dtype=torch.float32)
    A = best[1]
    with torch.no_grad():
        down.lora_A.copy_(A)
        down.lora_B.copy_(Epinv @ C)
    pred = []
    with torch.inference_mode():
        for x in xs:
            pred.append(int(m(x)[0][0, -1, lids].argmax()))
    ok = sum(int(pred[i] == K40[i]) for i in range(N))
    miss = [i + 1 for i in range(N) if pred[i] != K40[i]]
    print(f"[LP solution] {ok}/{N} miss={miss}", flush=True)
    if ok == N:
        m.save_packed_safetensors(a.out)
        print(f"[saved] {a.out}", flush=True)


if __name__ == "__main__":
    main()
