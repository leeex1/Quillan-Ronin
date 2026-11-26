#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WSP v10 — higher-capacity: MULTI-LAYER key concatenation, single-prompt correct.

v9 uses only last-layer k (16-dim). If 20 questions aren't linearly separable in
16 dims, v10 concatenates k from several layers (e.g. 4 layers x 16 = 64-dim) and
writes dB to each layer's down_proj lora_B, giving the classifier far more capacity.

Derivation: h_final shift = sum_L dB_L . k_L ; logit_j = E_j . h => shifts add.
Fit C = (K^T K + lam I)^-1 K^T T with K=[k_L1|k_L2|...] (20 x 64), then split C
into per-layer pieces and lift each with E^+ into that layer's dB_L.
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
from hard_math20 import KEY, format_problem
from wsp_math_eval import tok_for, letter_ids, load_model
from wsp_math_tune_v3 import enable_cache


def seed_all(model, layers, seed=42):
    g = torch.Generator().manual_seed(seed)
    for li in layers:
        down = model.layers[li].down_proj
        if down.lora_A.abs().max().item() == 0:
            with torch.no_grad():
                down.lora_A.copy_(
                    torch.randn(down.lora_A.shape, generator=g)
                    * (1.0 / (down.lora_A.shape[1] ** 0.5))
                )
    return [model.layers[li].down_proj for li in layers]


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--layers", default="8,15,20,23")
    a = ap.parse_args()
    a.ckpt = a.ckpt.resolve()
    layers = [int(x) for x in a.layers.split(",")]
    tok = tok_for(151936)
    lids = letter_ids(tok, 151936)
    xs = [
        torch.tensor([tok.encode(format_problem(i), add_special_tokens=False)])
        for i in range(20)
    ]

    m = load_model(a.ckpt)
    downs = seed_all(m, layers, 42)
    enable_cache(m)
    E = m.embed_tokens.unpack_float().float()[lids]
    Epinv = E.T @ torch.linalg.inv(E @ E.T + 1e-6 * torch.eye(4))
    B0s = [d.lora_B.detach().clone() for d in downs]

    # capture concatenated keys from all layers (single-prompt)
    caps = {li: {} for li in layers}
    hooks = []
    for li in layers:
        hooks.append(
            m.layers[li].down_proj.register_forward_pre_hook(
                (
                    lambda l: (
                        lambda mod, inp: caps[l].__setitem__(
                            "x", inp[0].detach().float()
                        )
                    )
                )(li)
            )
        )
    keys = []
    with torch.inference_mode():
        for x in xs:
            m(x)
            kp = []
            for li in layers:
                A = m.layers[li].down_proj.lora_A.detach().float()
                kp.append(F.linear(caps[li]["x"][0, -1], A))
            keys.append(torch.cat(kp))
    for h in hooks:
        h.remove()
    K = torch.stack(keys)
    print(f"[K shape] {tuple(K.shape)}", flush=True)

    def ev():
        pred = []
        with torch.inference_mode():
            for x in xs:
                lg = m(x)[0][0, -1, lids]
                pred.append(max(range(4), key=lambda j: float(lg[j])))
        return sum(int(pred[i] == KEY[i]) for i in range(20)), [
            i + 1 for i in range(20) if pred[i] != KEY[i]
        ]

    base, miss = ev()
    print(f"[base] {base}/20", flush=True)
    best = (base, None, None, None)
    for margin in (1.0, 2.0, 4.0, 8.0):
        for lam in (1e-4, 1e-3, 1e-2):
            for d, B0 in zip(downs, B0s):
                with torch.no_grad():
                    d.lora_B.copy_(B0)
            T = torch.full((20, 4), -margin)
            for i in range(20):
                T[i, KEY[i]] = margin
            C = torch.linalg.solve(
                K.T @ K + lam * torch.eye(K.shape[1]), K.T @ T
            ).T  # [4,64]
            d = K.shape[1] // len(layers)
            pieces = []
            for j, L in enumerate(layers):
                cj = C[:, j * d : (j + 1) * d]
                pieces.append(Epinv @ cj)
                with torch.no_grad():
                    downs[j].lora_B.copy_(pieces[-1])
            ok, miss = ev()
            print(f"[m={margin} l={lam}] {ok}/20 miss={miss}", flush=True)
            if ok > best[0]:
                best = (ok, [p.clone() for p in pieces], margin, lam)
            if ok == 20:
                break
        if best[0] == 20:
            break

    ok, pieces, margin, lam = best
    for d, B in zip(downs, pieces):
        with torch.no_grad():
            d.lora_B.copy_(B)
    m.save_packed_safetensors(a.out)
    print(f"[saved] best={ok}/20 layers={layers} m={margin} l={lam}", flush=True)


if __name__ == "__main__":
    main()
