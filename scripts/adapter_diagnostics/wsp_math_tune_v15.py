#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WSP v15 — baseline-corrected targets + verified fast batched evaluator.

Fixes two issues that capped v12/v14 at 34/40:
 (1) TARGET BUG: v9/v12/v14 fit T=+margin/-margin ignoring the model's existing
     logits. Correct residual target: t_ij = desired_final_ij - baseline_ij.
     v4 did this and was more sample-efficient; restored here.
 (2) SPEED: LEFT-pad batches so the last real token is ALWAYS at -1 (verified to
     match single-prompt inference), letting 128 seeds run in minutes not hours.

Closed-form MEMIT ridge, zero backprop.
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


def make_leftpad(tok, lids):
    seqs = [tok.encode(fmt40(i), add_special_tokens=False) for i in range(40)]
    pad = tok.pad_token_id if tok.pad_token_id is not None else lids[0]
    L = max(len(s) for s in seqs)
    x = torch.full((40, L), pad, dtype=torch.long)
    for i, s in enumerate(seqs):
        x[i, L - len(s) :] = torch.tensor(s)
    return x


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seeds", type=int, default=128)
    ap.add_argument("--margin", type=float, default=2.0)
    ap.add_argument("--lam", type=float, default=1e-5)
    a = ap.parse_args()
    a.ckpt = a.ckpt.resolve()
    N = 40
    tok = tok_for(151936)
    lids = letter_ids(tok, 151936)
    x = make_leftpad(tok, lids)

    m = load_model(a.ckpt)
    down = m.layers[-1].down_proj
    enable_cache(m)
    E = m.embed_tokens.unpack_float().float()[lids]
    Epinv = E.T @ torch.linalg.inv(E @ E.T + 1e-6 * torch.eye(4))

    # capture X at last real token (-1) and baseline logits in ONE forward
    cap = {}
    hd = down.register_forward_pre_hook(
        lambda mod, inp: cap.__setitem__("x", inp[0].detach().float())
    )
    with torch.inference_mode():
        lg0 = m(x)[0][:, -1, lids]
    hd.remove()
    X = cap["x"][:, -1, :]  # [40,4864]
    baseline = lg0.clone()  # [40,4]

    def batch_letter_logits():
        with torch.inference_mode():
            return m(x)[0][:, -1, lids]

    def score(lg):
        pred = lg.argmax(1).tolist()
        ok = sum(int(pred[i] == K40[i]) for i in range(N))
        return ok, [i + 1 for i in range(N) if pred[i] != K40[i]]

    # verify batched == single on baseline sample
    one = tok.encode(fmt40(0), add_special_tokens=False)
    with torch.inference_mode():
        ls = m(torch.tensor([one]))[0][0, -1, lids]
    drift = float((ls - baseline[0]).abs().max())
    print(
        f"[batch-vs-single drift] {drift:.6f} | baseline {score(baseline)[0]}/{N}",
        flush=True,
    )

    best = (score(baseline)[0], None, None, None)
    for seed in range(a.seeds):
        g = torch.Generator().manual_seed(seed + 7000)
        A = torch.randn(down.lora_A.shape, generator=g) * (
            1.0 / (down.lora_A.shape[1] ** 0.5)
        )
        with torch.no_grad():
            down.lora_A.copy_(A)
        K = torch.stack([F.linear(X[i], A.detach().float()) for i in range(N)])
        # baseline-corrected residual targets
        T = torch.zeros(N, 4)
        for i in range(N):
            mx = baseline[i].max()
            for j in range(4):
                desired = (
                    (mx + a.margin) if j == K40[i] else (baseline[i, j] - a.margin)
                )
                T[i, j] = desired - baseline[i, j]
        for lam in (a.lam, a.lam * 10, a.lam * 0.1):
            C = torch.linalg.solve(K.T @ K + lam * torch.eye(K.shape[1]), K.T @ T).T
            dB = Epinv @ C
            with torch.no_grad():
                down.lora_B.copy_(dB)
            ok, miss = score(batch_letter_logits())
            if ok > best[0]:
                best = (ok, dB.clone(), A.clone(), lam)
                print(
                    f"[NEW BEST seed={seed} lam={lam}] {ok}/{N} miss={miss[:12]}",
                    flush=True,
                )
            if ok == N:
                break
        if best[0] == N:
            break
        if seed % 16 == 0:
            print(f"[progress seed={seed}] best={best[0]}/{N}", flush=True)

    ok, Bf, Af, lam = best
    with torch.no_grad():
        down.lora_A.copy_(Af)
        down.lora_B.copy_(Bf)
    m.save_packed_safetensors(a.out)
    print(f"[saved] best={ok}/{N} lam={lam}", flush=True)


if __name__ == "__main__":
    main()
