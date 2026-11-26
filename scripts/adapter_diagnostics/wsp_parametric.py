#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Parametric WSP: set tensors to install a COMPUTATION (a+b bucketing), not a lookup.

Calibrate the closed-form LoRA write over MANY numeral pairs; test on HELD-OUT
pairs never seen. If held-out accuracy >> 25% chance, the tensor edit generalizes
=> installed capability. If held-out ~25% while calibration ~100%, it's memorization.

This is the honest discriminator the council (C17/C26/C8) demanded.
Single-prompt evaluator, zero backprop.
"""

from __future__ import annotations
import sys, random
from pathlib import Path
import torch
import torch.nn.functional as F

ROOT = Path(r"C:\02_QUILLAN")
DIAG = ROOT / "scripts" / "adapter_diagnostics"
for p in (str(ROOT / "scripts"), str(ROOT / "03 - Training & Model"), str(DIAG)):
    if p not in sys.path:
        sys.path.insert(0, p)
from hard_math20 import LETTERS
from wsp_math_eval import tok_for, letter_ids, load_model
from wsp_math_tune_v3 import enable_cache, seed_lora_basis
from parametric_math import make_split


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--train-n", type=int, default=120)
    ap.add_argument("--test-n", type=int, default=40)
    ap.add_argument("--margin", type=float, default=1.0)
    ap.add_argument("--lam", type=float, default=1e-4)
    a = ap.parse_args()
    a.ckpt = a.ckpt.resolve()
    tok = tok_for(151936)
    lids = letter_ids(tok, 151936)

    train = make_split(a.train_n, seed=101)  # (prompt, key, pair)
    test = make_split(a.test_n, seed=999)  # disjoint pairs (different seed+range)
    # ensure disjoint pairs
    tp = {x[2] for x in train}
    test = [x for x in test if x[2] not in tp]
    print(f"[data] train={len(train)} test={len(test)} disjoint={len(tp)}", flush=True)

    def enc(rows):
        return [
            torch.tensor([tok.encode(r[0], add_special_tokens=False)]) for r in rows
        ]

    m = load_model(a.ckpt)
    down = m.layers[-1].down_proj
    seed_lora_basis(m, [len(m.layers) - 1], 42)
    enable_cache(m)
    A = down.lora_A.detach().float()
    d = A.shape[0]
    E = m.embed_tokens.unpack_float().float()[lids]
    Epinv = E.T @ torch.linalg.inv(E @ E.T + 1e-6 * torch.eye(4))

    def score(rows):
        xs = enc(rows)
        ok = 0
        with torch.inference_mode():
            for x, r in zip(xs, rows):
                lg = m(x)[0][0, -1, lids]
                ok += int(int(lg.argmax()) == r[1])
        return ok

    base_tr = score(train)
    base_te = score(test)
    print(f"[base] train={base_tr}/{len(train)} test={base_te}/{len(test)}", flush=True)

    # capture keys K_train from last-token hidden
    train_x = enc(train)
    Xs = []
    for x in train_x:
        cap = {}
        hd = down.register_forward_pre_hook(
            lambda mod, inp: cap.__setitem__("x", inp[0].detach().float())
        )
        with torch.inference_mode():
            m(x)
        hd.remove()
        Xs.append(cap["x"][0, -1])
    K = torch.stack([F.linear(Xs[i], A) for i in range(len(train))])
    N = len(train)
    T = torch.full((N, 4), -a.margin)
    for i, r in enumerate(train):
        T[i, r[1]] = a.margin
    C = torch.linalg.solve(K.T @ K + a.lam * torch.eye(d), K.T @ T).T
    dB = Epinv @ C
    with torch.no_grad():
        down.lora_B.copy_(dB)

    tr = score(train)
    te = score(test)
    print(
        f"[WSP] train={tr}/{len(train)} test={te}/{len(test)} |dB|={float(dB.norm()):.3f}",
        flush=True,
    )
    gap = tr / len(train) - te / len(test)
    verdict = (
        "GENERALIZES (installed capability)"
        if te / len(test) > 0.5
        else "MEMORIZES (lookup table)"
    )
    print(f"[VERDICT] train-test gap={gap:.3f} -> {verdict}", flush=True)
    if a.out and te / len(test) > 0.5:
        m.save_packed_safetensors(a.out)
        print(f"[saved] {a.out}", flush=True)


if __name__ == "__main__":
    main()
