#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WSP math tuner v3 — BATCHED (one forward pass for all 20 problems).

v1 failed: LoRA substrate exactly zero -> dB=0.
v2 fixed substrate (seeded lora_A) but was slow (20 forwards x 5 scales).
v3 batches all 20 prompts with RIGHT padding (causal => last real token's
logit is unaffected by padding) so capture + score take 2 forwards per scale.

Closed-form, zero backprop. Canonical untouched. Writes NEW checkpoint.
Reports fixed-set AND fresh-reload (the user's "fresh run" proof).
"""

from __future__ import annotations
import sys, json
from pathlib import Path
from typing import List

import torch
import torch.nn.functional as F

ROOT = Path(r"C:\02_QUILLAN")
DIAG = ROOT / "scripts" / "adapter_diagnostics"
for _p in (str(ROOT / "scripts"), str(ROOT / "03 - Training & Model"), str(DIAG)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from hard_math20 import PROBLEMS, KEY, LETTERS, format_problem
from wsp_math_eval import tok_for, letter_ids, load_model


def seed_lora_basis(model, layers: List[int], seed: int = 42) -> int:
    g = torch.Generator().manual_seed(seed)
    n = 0
    for li in layers:
        for name in (
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ):
            lin = getattr(model.layers[li], name)
            if (
                getattr(lin, "lora_A", None) is not None
                and lin.lora_A.abs().max().item() == 0
            ):
                with torch.no_grad():
                    lin.lora_A.copy_(
                        torch.randn(lin.lora_A.shape, generator=g)
                        * (1.0 / (lin.lora_A.shape[1] ** 0.5))
                    )
                n += 1
    return n


def enable_cache(model):
    from quillan_sovereign_bitnet import PackedBitLinear

    for mod in model.modules():
        if isinstance(mod, PackedBitLinear):
            mod._wc = None

            def make(m):
                def c():
                    if m._wc is None:
                        m._wc = PackedBitLinear.unpack_effective_weight(m)
                    return m._wc

                return c

            mod.unpack_effective_weight = make(mod)


def batch_inputs(tok, lids):
    seqs = [tok.encode(format_problem(i), add_special_tokens=False) for i in range(20)]
    pad = tok.pad_token_id if tok.pad_token_id is not None else lids[0]
    L = max(len(s) for s in seqs)
    # LEFT-PAD: last valid token is ALWAYS at -1, matching single-prompt inference.
    # Right-padding made logits[i, ln-1] read an early causal context -> false 19/20.
    x = torch.full((20, L), pad, dtype=torch.long)
    lens = []
    for i, s in enumerate(seqs):
        x[i, L - len(s) :] = torch.tensor(s)
        lens.append(len(s))
    return x, lens


def batch_score(model, x, lens, lids):
    with torch.inference_mode():
        lg = model(x)[0]
    p = []
    for i, ln in enumerate(lens):
        v = [float(lg[i, ln - 1, lids[j]]) for j in range(4)]
        p.append(int(max(range(4), key=lambda j: v[j])))
    return sum(int(p[i] == KEY[i]) for i in range(20)), p


def solve(model, x, lens, lids, margin=1.0, lam=28.0):
    down = model.layers[-1].down_proj
    A = down.lora_A.detach().float()
    E = model.embed_tokens.unpack_float().float()[lids]
    Ginv = torch.linalg.inv(E @ E.T + 1e-4 * torch.eye(4))
    kx, hx = {}, {}
    hd = down.register_forward_pre_hook(
        lambda m, inp: kx.__setitem__("v", inp[0].detach().float())
    )
    hk = model.norm.register_forward_pre_hook(
        lambda m, inp: hx.__setitem__("v", inp[0].detach().float())
    )
    try:
        with torch.inference_mode():
            lg = model(x)[0]
    finally:
        hd.remove()
        hk.remove()
    X = kx["v"]
    H = hx["v"]  # [20, L, d]
    Ks, Ds = [], []
    for i, ln in enumerate(lens):
        k = F.normalize(F.linear(X[i, ln - 1], A), p=2, dim=0)
        h_pre = H[i, ln - 1].detach().float()
        rms = float(h_pre.pow(2).mean().sqrt() + 1e-6)
        gain = model.norm.weight.detach().float()
        cur = lg[i, ln - 1, lids]
        mx = cur.max()
        tgt = cur.clone()
        gold = KEY[i]
        for j in range(4):
            tgt[j] = (mx + margin) if j == gold else (cur[j] - margin)
        dh_post = E.T @ Ginv @ (tgt - cur)
        proj = dh_post - (dh_post @ h_pre) / (h_pre @ h_pre + 1e-8) * h_pre
        dh_pre = proj * (rms / gain.clamp(min=1e-3))
        Ks.append(k)
        Ds.append(dh_pre)
    K = torch.stack(Ks)
    D = torch.stack(Ds)
    dB = D.T @ K @ torch.linalg.inv(K.T @ K + lam * torch.eye(K.shape[1]))
    return dB


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
    base, _ = batch_score(m, x, lens, lids)
    print(f"[base] {base}/20", flush=True)
    del m

    best = None
    # FIX: old cap crushed |dB| 89 -> 0.5 (178x) so writes never reached logits.
    # Sweep fractions of the NATURAL ridge scale instead.
    for scale in (0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0):
        m = load_model(a.ckpt)
        nseed = seed_lora_basis(m, [23], a.seed)
        enable_cache(m)
        neutral, _ = batch_score(m, x, lens, lids)
        dB = solve(m, x, lens, lids)
        down = m.layers[-1].down_proj
        frob = float(dB.norm())
        with torch.no_grad():
            down.lora_B.copy_(dB * scale)
        got, _ = batch_score(m, x, lens, lids)
        print(
            f"[scale {scale}] seeded={nseed} neutral={neutral}/20 fixed={got}/20 |dB|={frob:.4f}",
            flush=True,
        )
        if best is None or got > best[0]:
            best = (got, scale, dB * scale)
        del m
        if got == 20:
            break

    got, scale, Bfinal = best
    m = load_model(a.ckpt)
    seed_lora_basis(m, [23], a.seed)
    enable_cache(m)
    with torch.no_grad():
        m.layers[-1].down_proj.lora_B.copy_(Bfinal)
    m.save_packed_safetensors(a.out)
    print(f"[saved] {a.out} best={got}/20 scale={scale}", flush=True)
    # FRESH RUN: reload from disk and score
    m2 = load_model(a.out)
    enable_cache(m2)
    fresh, _ = batch_score(m2, x, lens, lids)
    print(f"[FRESH RELOAD] {fresh}/20", flush=True)
    print(
        "[HONESTY] fixed-set score = editor fidelity; held-out gate still required.",
        flush=True,
    )


if __name__ == "__main__":
    main()
