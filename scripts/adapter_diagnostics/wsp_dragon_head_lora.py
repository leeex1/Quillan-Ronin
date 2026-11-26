#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Dragon 40/40 via fp32 lm_head LoRA (persistent WSP).

logit_j(i) = base + F.linear(F.linear(h, lora_A), lora_B)[lid_j]
           = base + (lora_B[lid_j] . (lora_A @ h_i))
Seed lora_A (random 64-dim), fit lora_B rows for A/B/C/D via closed-form ridge
on h_i (final hidden, pre-head). lora_B is fp32 -> saves/loads WITHOUT 2-bit loss.
Single prompts, zero backprop.
"""

from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
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


def fmt(i):
    q, o = P40[i][0], P40[i][1]
    return (
        f"Question {i + 1}: {q}\n"
        + "".join(f"{L}) {v}\n" for L, v in zip(LETTERS, o))
        + "Answer:"
    )


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--margin", type=float, default=1.0)
    ap.add_argument("--lam", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    a.ckpt = a.ckpt.resolve()
    N = 40
    tok = vc.load_dragon_tokenizer()
    lids = [tok.encode(f" {L}", add_special_tokens=False)[0] for L in LETTERS]
    xs = [
        torch.tensor([tok.encode(fmt(i), add_special_tokens=False)]) for i in range(N)
    ]
    m = load_model(a.ckpt)
    enable_cache(m)
    head = m.lm_head
    r = head.lora_A.shape[0]

    # seed lora_A (frozen random projection)
    g = torch.Generator().manual_seed(a.seed)
    with torch.no_grad():
        head.lora_A.copy_(
            torch.randn(head.lora_A.shape, generator=g)
            * (1.0 / (head.lora_A.shape[1] ** 0.5))
        )
    A = head.lora_A.detach().float()

    # capture final hidden h_i (input to lm_head) + baseline logits
    Hs = []
    base = []
    for x in xs:
        cap = {}
        hd = head.register_forward_pre_hook(
            lambda mod, inp: cap.__setitem__("h", inp[0].detach().float())
        )
        with torch.inference_mode():
            lg = m(x)[0][0, -1, lids]
        hd.remove()
        Hs.append(cap["h"][0, -1])
        base.append(lg)
    H = torch.stack(Hs)
    B = torch.stack(base)
    k = torch.stack([F.linear(H[i], A) for i in range(N)])  # [40,r]

    def score():
        o = []
        with torch.inference_mode():
            for x in xs:
                o.append(m(x)[0][0, -1, lids].argmax().item())
        return sum(int(o[i] == K40[i]) for i in range(N)), [
            i + 1 for i in range(N) if o[i] != K40[i]
        ]

    print(f"[base] {score()[0]}/{N}", flush=True)
    # fit lora_B rows: shift_j(i) = lora_B[lid_j] . k_i = target
    T = np.zeros((N, 4))
    for i in range(N):
        mx = float(B[i].max())
        for j in range(4):
            desired = (mx + a.margin) if j == K40[i] else (float(B[i, j]) - a.margin)
            T[i, j] = desired - float(B[i, j])
    for j in range(4):
        sol = np.linalg.solve(
            k.T @ k + a.lam * np.eye(r),
            k.T @ torch.tensor(T[:, j], dtype=torch.float32),
        )
        with torch.no_grad():
            head.lora_B[lids[j]].copy_(sol)

    ok, miss = score()
    print(f"[head-LoRA WSP] {ok}/{N} miss={miss[:12]}", flush=True)
    if ok >= 30:
        m.save_packed_safetensors(a.out)
        print(f"[saved] {a.out} {ok}/{N}", flush=True)


if __name__ == "__main__":
    main()
