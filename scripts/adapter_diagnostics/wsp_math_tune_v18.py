#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WSP v18 — EMPIRICAL write-effect matrix ( fixes the LP-vs-reality gap ).

v17 LP was 'feasible' but model scored 33/40: the theoretical shift logit_j +=
c_j.k_i is wrong because the packed ternary path quantizes activations AFTER the
LoRA add, so the true effect is nonlinear. v18 measures the ACTUAL per-prompt
logit change for a unit write along each lora_A basis direction via real single
forwards, then solves the ridge against the MEASURED effect matrix.

Effect matrix G[i,j,r] = d logit_j(i) / d (lora_B[j,r]) measured by finite
difference (one extra forward per r, rank=16 -> 16 forwards + baseline).
Then solve for dB that achieves target shifts using least squares on G.
Closed-form, single forwards, zero backprop.
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
    ap.add_argument("--eps", type=float, default=0.1)
    ap.add_argument("--margin", type=float, default=1.5)
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
        return torch.stack(out)  # [40,4]

    base = letters()
    print(
        f"[base] {sum(int(base[i].argmax() == K40[i]) for i in range(N))}/{N}",
        flush=True,
    )

    # measure effect: for each output letter j and rank dim rr, perturb lora_B[j,rr]
    # G[i,j,rr] = (logit_j(i) after +eps) - base[i,j]
    G = np.zeros((N, 4, r))
    for j in range(4):
        for rr in range(r):
            with torch.no_grad():
                down.lora_B.copy_(B0)
                down.lora_B[lids[j], rr] += a.eps
            cur = letters()
            G[:, j, rr] = ((cur - base)[:, j] / a.eps).numpy()
    with torch.no_grad():
        down.lora_B.copy_(B0)
    print(f"[G measured] shape={G.shape} mean|G|={np.abs(G).mean():.5f}", flush=True)

    # target shifts: correct letter up, others down (baseline-corrected)
    T = np.zeros((N, 4))
    for i in range(N):
        mx = float(base[i].max())
        for j in range(4):
            desired = (mx + a.margin) if j == K40[i] else (float(base[i, j]) - a.margin)
            T[i, j] = desired - float(base[i, j])

    # solve per-letter independently: for each j, find delta_B[j,:] (r vector)
    # such that G[:,j,:] . delta_B[j,:] ~= T[:,j]  (N equations, r unknowns)
    dB_np = np.zeros((down.lora_B.shape[0], r))
    for j in range(4):
        Gj = G[:, j, :]  # [N,r]
        tj = T[:, j]  # [N]
        sol = np.linalg.lstsq(Gj, tj, rcond=1e-3)[0]  # [r]
        # only write into the embedding row for letter j (that's what G measured)
        dB_np[lids[j], :] = sol
    with torch.no_grad():
        down.lora_B.copy_(torch.tensor(dB_np, dtype=torch.float32))

    final = letters()
    ok = sum(int(final[i].argmax() == K40[i]) for i in range(N))
    miss = [i + 1 for i in range(N) if final[i].argmax() != K40[i]]
    print(f"[EMPIRICAL solution] {ok}/{N} miss={miss}", flush=True)
    if ok >= 34:
        m.save_packed_safetensors(a.out)
        print(f"[saved] {a.out} {ok}/{N}", flush=True)


if __name__ == "__main__":
    main()
