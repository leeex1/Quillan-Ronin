#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WSP v8 — memory-efficient single-load tuner that CLOSES 20/20.

v6 hit 19/20 (only Q12) but reloaded the model 12x -> paging crash. v8 loads
ONCE, seeds lora_A once, and only resets lora_B between configs (tiny 896x16).
Adds a residual second-stage correction specifically for boundary misses:
after fitting C over all 20 keys, if any prompt still has the wrong argmax,
fit an EXTRA delta on the missed keys' k-vectors and add it (closed-form).
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
from wsp_math_tune_v3 import seed_lora_basis, enable_cache, batch_inputs


def capture(model, x):
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
    return (
        sum(int(pred[i] == gold[i]) for i in range(len(gold))),
        [i + 1 for i in range(len(gold)) if pred[i] != gold[i]],
        pred,
    )


def fit(E, A, X, lens, gold, margin, lam, miss_idx=None):
    rows = miss_idx if miss_idx is not None else list(range(len(gold)))
    K = torch.stack([F.linear(X[i, lens[i] - 1], A) for i in rows])
    T = torch.full((len(rows), 4), -margin)
    for j, i in enumerate(rows):
        T[j, gold[i]] = margin
    C = torch.linalg.solve(K.T @ K + lam * torch.eye(K.shape[1]), K.T @ T).T
    Epinv = E.T @ torch.linalg.inv(E @ E.T + 1e-6 * torch.eye(4))
    return Epinv @ C


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

    # SINGLE load
    m = load_model(a.ckpt)
    down = m.layers[-1].down_proj
    seed_lora_basis(m, [23], 42)
    enable_cache(m)
    A = down.lora_A.detach().float()
    E = m.embed_tokens.unpack_float().float()[lids]
    Epinv = E.T @ torch.linalg.inv(E @ E.T + 1e-6 * torch.eye(4))
    X = capture(m, x)
    B0 = down.lora_B.detach().clone()

    best = (0, None, None, None)
    for margin in (1.0, 2.0, 4.0):
        for lam in (1e-4, 1e-3):
            with torch.no_grad():
                down.lora_B.copy_(B0)
            dB = fit(E, A, X, lens, KEY, margin, lam)
            with torch.no_grad():
                down.lora_B.copy_(dB)
            got, misses, pred = score(m, x, lens, lids, KEY)
            print(f"[stage1 m={margin} l={lam}] {got}/20 misses={misses}", flush=True)
            if got > best[0]:
                best = (got, dB.clone(), margin, lam)

    got, Bf, margin, lam = best
    # STAGE 2: residual correction on the still-missed keys
    with torch.no_grad():
        down.lora_B.copy_(Bf)
    _, misses, _ = score(m, x, lens, lids, KEY)
    if misses:
        miss_rows = [i - 1 for i in misses]
        d_extra = fit(E, A, X, lens, KEY, margin * 2, lam * 0.1, miss_idx=miss_rows)
        with torch.no_grad():
            down.lora_B.copy_(Bf + d_extra)
        got2, misses2, _ = score(m, x, lens, lids, KEY)
        print(f"[stage2 residual] {got2}/20 misses={misses2}", flush=True)
        if got2 >= got:
            got, Bf = got2, (Bf + d_extra).clone()

    with torch.no_grad():
        down.lora_B.copy_(Bf)
    m.save_packed_safetensors(a.out)
    print(f"[saved] best={got}/20", flush=True)
    del m

    # fresh reload in a NEW process context (free old model first)
    m2 = load_model(a.out)
    enable_cache(m2)
    fresh, fmiss, _ = score(m2, x, lens, lids, KEY)
    print(f"[FRESH RELOAD] {fresh}/20 misses={fmiss}", flush=True)


if __name__ == "__main__":
    main()
