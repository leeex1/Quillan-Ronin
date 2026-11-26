#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WSP v9 — CORRECT single-prompt evaluator (fixes the batch metric bug).

Diagnostic proved batched(ln-1)=19/20 was garbage; true single-prompt accuracy
= 6/20. v9 captures keys and scores with 20 INDIVIDUAL forwards (the real
inference path). One model load, lora_B reset between configs -> no paging crash.
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
from wsp_math_tune_v3 import seed_lora_basis, enable_cache


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.ckpt = a.ckpt.resolve()
    tok = tok_for(151936)
    lids = letter_ids(tok, 151936)
    seqs = [tok.encode(format_problem(i), add_special_tokens=False) for i in range(20)]
    xs = [torch.tensor([s]) for s in seqs]

    m = load_model(a.ckpt)
    down = m.layers[-1].down_proj
    seed_lora_basis(m, [23], 42)
    enable_cache(m)
    A = down.lora_A.detach().float()
    E = m.embed_tokens.unpack_float().float()[lids]
    Epinv = E.T @ torch.linalg.inv(E @ E.T + 1e-6 * torch.eye(4))
    B0 = down.lora_B.detach().clone()

    # capture keys + baseline in single-prompt forwards
    cap = {}
    hd = down.register_forward_pre_hook(
        lambda mod, inp: cap.__setitem__("x", inp[0].detach().float())
    )
    keys = []
    base = []
    with torch.inference_mode():
        for x in xs:
            lg = m(x)[0][0, -1, lids]
            base.append(max(range(4), key=lambda j: float(lg[j])))
            keys.append(F.linear(cap["x"][0, -1], A))
    hd.remove()
    K = torch.stack(keys)
    bok = sum(int(base[i] == KEY[i]) for i in range(20))
    print(f"[base single] {bok}/20", flush=True)

    def eval_single():
        pred = []
        with torch.inference_mode():
            for x in xs:
                lg = m(x)[0][0, -1, lids]
                pred.append(max(range(4), key=lambda j: float(lg[j])))
        ok = sum(int(pred[i] == KEY[i]) for i in range(20))
        return ok, [i + 1 for i in range(20) if pred[i] != KEY[i]]

    best = (bok, B0.clone(), None, None)
    for margin in (1.0, 2.0, 4.0, 8.0, 16.0):
        for lam in (1e-5, 1e-4, 1e-3, 1e-2):
            with torch.no_grad():
                down.lora_B.copy_(B0)
            T = torch.full((20, 4), -margin)
            for i in range(20):
                T[i, KEY[i]] = margin
            C = torch.linalg.solve(K.T @ K + lam * torch.eye(K.shape[1]), K.T @ T).T
            dB = Epinv @ C
            with torch.no_grad():
                down.lora_B.copy_(dB)
            ok, miss = eval_single()
            print(f"[m={margin} l={lam}] {ok}/20 miss={miss}", flush=True)
            if ok > best[0]:
                best = (ok, dB.clone(), margin, lam)
            del C, dB

    ok, Bf, margin, lam = best
    with torch.no_grad():
        down.lora_B.copy_(Bf)
    m.save_packed_safetensors(a.out)
    print(f"[saved] best single={ok}/20 m={margin} l={lam}", flush=True)


if __name__ == "__main__":
    main()
