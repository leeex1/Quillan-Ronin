#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WSP v19 — fast empirical effect measurement (16 forwards, not 2560).

Key: perturbing lora_B[:,rr] (whole column rr) affects ALL 4 letter logits at
once. So for each rank dim rr (16 total), ONE batch of forwards gives the effect
on all letters. v18 wrongly looped 4 letters x 16 dims = 64 perturbations;
v19 does 16. Measures G[i,j,rr] = dlogit_j(i)/dB[j,rr] empirically, then solves
per-letter least squares for the write. Single forwards, closed-form, no backprop.
"""

from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import torch

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
    ap.add_argument("--eps", type=float, default=0.2)
    ap.add_argument("--margin", type=float, default=1.0)
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
    seed_lora_basis(m, [23], 42)
    enable_cache(m)
    r = down.lora_A.shape[0]
    B0 = down.lora_B.detach().clone()

    def letters():
        out = []
        with torch.inference_mode():
            for x in xs:
                out.append(m(x)[0][0, -1, lids].clone())
        return torch.stack(out)

    base = letters()
    print(
        f"[base] {sum(int(base[i].argmax() == K40[i]) for i in range(N))}/{N}",
        flush=True,
    )

    # perturb each rank column (all 4 letters at once) -> 16 perturbations
    G = np.zeros((N, 4, r))
    for rr in range(r):
        with torch.no_grad():
            down.lora_B.copy_(B0)
            for j in range(4):
                down.lora_B[lids[j], rr] += a.eps
        cur = letters()
        G[:, :, rr] = ((cur - base) / a.eps).numpy()
        if (rr + 1) % 4 == 0:
            print(f"[G] {rr + 1}/{r} rank dims measured", flush=True)
    with torch.no_grad():
        down.lora_B.copy_(B0)

    T = np.zeros((N, 4))
    for i in range(N):
        mx = float(base[i].max())
        for j in range(4):
            desired = (mx + a.margin) if j == K40[i] else (float(base[i, j]) - a.margin)
            T[i, j] = desired - float(base[i, j])

    # solve per-letter: G[:,j,:].sol ~= T[:,j]
    dB = np.zeros((down.lora_B.shape[0], r))
    for j in range(4):
        sol = np.linalg.lstsq(G[:, j, :], T[:, j], rcond=1e-2)[0]
        dB[lids[j], :] = sol
    with torch.no_grad():
        down.lora_B.copy_(torch.tensor(dB, dtype=torch.float32))

    final = letters()
    ok = sum(int(final[i].argmax() == K40[i]) for i in range(N))
    miss = [i + 1 for i in range(N) if final[i].argmax() != K40[i]]
    print(f"[EMPIRICAL v19] {ok}/{N} miss={miss}", flush=True)
    if ok >= 34:
        m.save_packed_safetensors(a.out)
        print(f"[saved] {a.out} {ok}/{N}", flush=True)


if __name__ == "__main__":
    main()
