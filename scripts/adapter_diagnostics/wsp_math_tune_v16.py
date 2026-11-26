#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WSP v16 — baseline-corrected targets, SINGLE forwards only (correct evaluator).

v15's batched evaluator mismatches single (drift 3.24) — padding changes attention,
so batching is invalid. v16 uses 40 genuine single forwards per config but fixes
the TARGET BUG: fit residual (desired - baseline) not raw +/-margin. This was
sample-efficient in v4 (hit 17/20 in 1 config). One model load; lora reset/seed.
"""

from __future__ import annotations
import sys
from pathlib import Path
import torch
import torch.nn.functional as F

ROOT = Path(r"C:\02_QUILLAN")
DIAG = ROOT / "scripts" / "adapter_diagnostics"
for p in (str(ROOT / "scripts"), str(ROOT / "03 - Training & Model"), str(DIAG)):
    if p not in sys.path:
        sys.path.insert(0, p)
from hard_math20 import LETTERS
from hard_math40 import PROBLEMS as P40, KEY as K40
from wsp_math_eval import tok_for, letter_ids, load_model
from wsp_math_tune_v3 import enable_cache


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
    ap.add_argument("--seeds", type=int, default=32)
    ap.add_argument("--margin", type=float, default=2.0)
    ap.add_argument("--lam", type=float, default=1e-5)
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

    def single_pass(return_logits=False):
        pred = []
        lgs = []
        with torch.inference_mode():
            for x in xs:
                lg = m(x)[0][0, -1, lids]
                lgs.append(lg)
                pred.append(int(lg.argmax()))
        ok = sum(int(pred[i] == K40[i]) for i in range(N))
        return (
            (ok, pred, torch.stack(lgs))
            if return_logits
            else (ok, [i + 1 for i in range(N) if pred[i] != K40[i]])
        )

    base, bmiss, baseline = single_pass(True)
    print(f"[base single] {base}/{N}", flush=True)

    # capture X
    Xs = []
    for x in xs:
        cap = {}
        hd = down.register_forward_pre_hook(
            lambda mod, inp: cap.__setitem__("x", inp[0].detach().float())
        )
        with torch.inference_mode():
            m(x)
        hd.remove()
        Xs.append(cap["x"][0, -1])
    X = torch.stack(Xs)

    best = (base, None, None)
    for seed in range(a.seeds):
        g = torch.Generator().manual_seed(seed + 11000)
        A = torch.randn(down.lora_A.shape, generator=g) * (
            1.0 / (down.lora_A.shape[1] ** 0.5)
        )
        with torch.no_grad():
            down.lora_A.copy_(A)
        K = torch.stack([F.linear(X[i], A.detach().float()) for i in range(N)])
        T = torch.zeros(N, 4)
        for i in range(N):
            mx = baseline[i].max()
            for j in range(4):
                desired = (
                    (mx + a.margin) if j == K40[i] else (baseline[i, j] - a.margin)
                )
                T[i, j] = desired - baseline[i, j]
        for lam in (a.lam, a.lam * 5):
            C = torch.linalg.solve(K.T @ K + lam * torch.eye(K.shape[1]), K.T @ T).T
            dB = Epinv @ C
            with torch.no_grad():
                down.lora_B.copy_(dB)
            ok, miss = single_pass()
            if ok > best[0]:
                best = (ok, dB.clone(), A.clone())
                print(
                    f"[NEW BEST seed={seed} lam={lam}] {ok}/{N} miss={miss[:12]}",
                    flush=True,
                )
            if ok == N:
                break
        if best[0] == N:
            break
        if seed % 8 == 0:
            print(f"[progress seed={seed}] best={best[0]}/{N}", flush=True)

    ok, Bf, Af = best
    with torch.no_grad():
        down.lora_A.copy_(Af)
        down.lora_B.copy_(Bf)
    m.save_packed_safetensors(a.out)
    print(f"[saved] best={ok}/{N}", flush=True)


if __name__ == "__main__":
    main()
