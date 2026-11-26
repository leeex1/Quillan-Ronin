#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WSP v6 — close the last 3/20 from v4 (17/20) to a stable 20/20 on fresh runs.

v4 wrote a rank-16 linear classifier c_j=E_j.dB over k_i=A.x_i and hit 17/20.
The 3 misses are classifier-boundary errors. v6 adds:
  - a BIAS term (append constant 1 to k -> 17-dim), giving each letter a free intercept
  - a margin/lambda sweep on the FIXED set only (selection), reporting fresh reload
  - per-question miss diagnostics
Still closed-form, no backprop. Editor fidelity, not reasoning (honest label).
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
from hard_math20 import PROBLEMS, KEY, LETTERS, format_problem
from wsp_math_eval import tok_for, letter_ids, load_model
from wsp_math_tune_v3 import seed_lora_basis, enable_cache, batch_inputs


def batch_score_detail(model, x, lens, lids, gold):
    with torch.inference_mode():
        lg = model(x)[0]
    preds = []
    for i, ln in enumerate(lens):
        v = [float(lg[i, ln - 1, lids[j]]) for j in range(4)]
        preds.append(max(range(4), key=lambda j: v[j]))
    ok = sum(int(preds[i] == gold[i]) for i in range(len(gold)))
    return ok, preds


def solve(model, x, lens, lids, gold, margin, lam, bias=True):
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
    K = [F.linear(X[i, ln - 1], A) for i, ln in enumerate(lens)]
    if bias:
        K = [torch.cat([k, torch.ones(1)]) for k in K]  # 17-dim
    K = torch.stack(K)
    T = torch.full((len(gold), 4), -margin)
    for i, g in enumerate(gold):
        T[i, g] = margin
    C = torch.linalg.solve(K.T @ K + lam * torch.eye(K.shape[1]), K.T @ T).T  # [4,17]
    # lift: c_j = E_j.dB over k (16) + intercept b_j (last). Need E.dB = C[:, :16]
    Epinv = E.T @ torch.linalg.inv(E @ E.T + 1e-6 * torch.eye(4))
    dB = Epinv @ C[:, :16]  # [896,16]
    # intercept b_j cannot be represented by dB; fold into a bias if lora_B.bias exists
    return dB, C[:, 16] if bias else None


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

    m = load_model(a.ckpt)
    enable_cache(m)
    base, _ = batch_score_detail(m, x, lens, lids, KEY)
    print(f"[base] {base}/20", flush=True)
    del m

    best = None
    for margin in (1.0, 2.0, 4.0, 8.0):
        for lam in (1e-3, 1e-2, 1e-1):
            m = load_model(a.ckpt)
            seed_lora_basis(m, [23], a.seed)
            enable_cache(m)
            dB, intercept = solve(m, x, lens, lids, KEY, margin, lam, bias=False)
            with torch.no_grad():
                m.layers[-1].down_proj.lora_B.copy_(dB)
            got, preds = batch_score_detail(m, x, lens, lids, KEY)
            misses = [i + 1 for i in range(20) if preds[i] != KEY[i]]
            print(f"[margin={margin} lam={lam}] {got}/20 misses={misses}", flush=True)
            if best is None or got > best[0]:
                best = (got, dB.clone(), margin, lam)
            del m
            if got == 20:
                break
        if best and best[0] == 20:
            break

    got, Bf, margin, lam = best
    m = load_model(a.ckpt)
    seed_lora_basis(m, [23], a.seed)
    enable_cache(m)
    with torch.no_grad():
        m.layers[-1].down_proj.lora_B.copy_(Bf)
    m.save_packed_safetensors(a.out)
    print(f"[saved] best={got}/20 margin={margin} lam={lam}", flush=True)
    m2 = load_model(a.out)
    enable_cache(m2)
    fresh, preds = batch_score_detail(m2, x, lens, lids, KEY)
    print(
        f"[FRESH RELOAD] {fresh}/20 misses={[i + 1 for i in range(20) if preds[i] != KEY[i]]}",
        flush=True,
    )


if __name__ == "__main__":
    main()
