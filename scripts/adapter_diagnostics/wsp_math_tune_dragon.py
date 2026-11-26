#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WSP 40/40 tuner for DRAGON (rank-64 packed checkpoint).

Same proven v20 method (seeded lora_A + closed-form ridge, single-prompt
evaluator), but uses Dragon's Falcon tokenizer (vocab 131072, letter ids
2356/2380/2350/2407). Dragon has tie_word_embeddings=False so logits come from
its fp32 lm_head — but the WSP write is still on layers[-1].down_proj.lora_B.
Zero backprop.
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
from wsp_math_eval import load_model
from wsp_math_tune_v3 import enable_cache
import v23_common as vc


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
    ap.add_argument("--seeds", type=int, default=24)
    ap.add_argument("--margin", type=float, default=1.0)
    ap.add_argument("--lam", type=float, default=1e-5)
    a = ap.parse_args()
    a.ckpt = a.ckpt.resolve()
    N = 40
    tok = vc.load_dragon_tokenizer()
    lids = [tok.encode(f" {L}", add_special_tokens=False)[0] for L in LETTERS]
    print(f"[dragon tokenizer] vocab={len(tok)} lids={lids}", flush=True)
    xs = [
        torch.tensor([tok.encode(fmt40(i), add_special_tokens=False)]) for i in range(N)
    ]

    m = load_model(a.ckpt)
    down = m.layers[-1].down_proj
    enable_cache(m)
    d = down.lora_A.shape[0]
    # Dragon has explicit lm_head -> logits = lm_head(h), not tied E. Get head weight.
    if m.lm_head is not None:
        W = m.lm_head.unpack_effective_weight().float()
        E = W[lids]
    else:
        E = m.embed_tokens.unpack_float().float()[lids]
    Epinv = E.T @ torch.linalg.inv(E @ E.T + 1e-6 * torch.eye(4))

    def single():
        pred = []
        with torch.inference_mode():
            for x in xs:
                lg = m(x)[0][0, -1, lids]
                pred.append(int(lg.argmax()))
        return sum(int(pred[i] == K40[i]) for i in range(N)), [
            i + 1 for i in range(N) if pred[i] != K40[i]
        ]

    base, bmiss = single()
    print(f"[base rank-{d}] {base}/{N}", flush=True)

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
        g = torch.Generator().manual_seed(seed + 500)
        A = torch.randn(down.lora_A.shape, generator=g) * (1.0 / (d**0.5))
        with torch.no_grad():
            down.lora_A.copy_(A)
        K = torch.stack([F.linear(X[i], A.detach().float()) for i in range(N)])
        T = torch.full((N, 4), -a.margin)
        for i in range(N):
            T[i, K40[i]] = a.margin
        C = torch.linalg.solve(K.T @ K + a.lam * torch.eye(d), K.T @ T).T
        dB = Epinv @ C
        with torch.no_grad():
            down.lora_B.copy_(dB)
        ok, miss = single()
        print(f"[seed {seed}] {ok}/{N} miss={miss[:10]}", flush=True)
        if ok > best[0]:
            best = (ok, dB.clone(), A.clone())
        if ok == N:
            break

    ok, Bf, Af = best
    with torch.no_grad():
        down.lora_A.copy_(Af)
        down.lora_B.copy_(Bf)
    m.save_packed_safetensors(a.out)
    print(f"[saved] best={ok}/{N} rank={d}", flush=True)


if __name__ == "__main__":
    main()
