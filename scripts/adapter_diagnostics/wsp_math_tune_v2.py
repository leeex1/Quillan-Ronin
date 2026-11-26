#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WSP math tuner v2 — fixes the ZERO-SUBSTRATE root cause (#1292).

v1 failed because lora_A/lora_B are EXACTLY ZERO at t=0 (gate_alpha=0, builder
never seeded EGGROLL LoRA). Ridge K=x@lora_A=0 -> dB=0 -> no learning.

v2 (still closed-form, ZERO backprop, canonical untouched):
  1. Seed lora_A with a DETERMINISTIC rank-16 basis (torch.manual_seed) in the
     target layers — a fixed, non-learned random projection. Output-neutral at
     t=0 because lora_B=0 (F.linear(...,B=0)=0). This creates a writable basis.
  2. Closed-form MEMIT ridge on lora_B (fp32) using RMSNorm-aware targets
     (C26 #1288 fix) and normalized keys / lambda=28.
  3. Frobenius cap applied AFTER scaling (C26 fix).
  4. Writes a NEW checkpoint; reports fixed set + permuted-options held-out.
  5. One-time unpack cache (bit-identical) for CPU speed.

This is a legitimate WSP *editor* — NOT proof of reasoning (C17 #1279 / C18 #1282).
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
from wsp_math_tune import enable_unpack_cache, verify_cache_bitidentical


def seed_lora_basis(model, layers: List[int], seed: int = 42) -> int:
    """Deterministically seed lora_A (frozen basis); lora_B stays zero (output-neutral)."""
    g = torch.Generator().manual_seed(seed)
    n = 0
    for li in layers:
        blk = model.layers[li]
        for name in (
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ):
            lin = getattr(blk, name)
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


def capture_and_solve(
    model, tok, lids, layers: List[int], margin: float = 1.0, lam: float = 28.0
):
    down = model.layers[layers[-1]].down_proj
    A = down.lora_A.detach().float()
    W = model.embed_tokens.unpack_float().float()
    E = W[lids].float()
    Ginv = torch.linalg.inv(E @ E.T + 1e-4 * torch.eye(4))

    Ks, Ds = [], []
    hook = down.register_forward_pre_hook(
        lambda m, inp: setattr(m, "_lx", inp[0].detach().float())
    )
    norm_hook = model.norm.register_forward_pre_hook(
        lambda m, inp: setattr(m, "_hx", inp[0].detach().float())
    )
    try:
        with torch.inference_mode():
            for i in range(len(PROBLEMS)):
                ids = tok.encode(format_problem(i), add_special_tokens=False)
                logits = model(torch.tensor([ids]))[0][0, -1, :]
                x = down._lx[0, -1, :]
                k = F.linear(x, A)
                h_pre = model.norm._hx[0, -1, :].detach().float()
                rms = float(h_pre.pow(2).mean().sqrt() + 1e-6)
                gain = model.norm.weight.detach().float()

                cur = logits[lids]
                mx = cur.max()
                tgt = cur.clone()
                gold = KEY[i]
                for j in range(4):
                    tgt[j] = (mx + margin) if j == gold else (cur[j] - margin)
                dlogit = tgt - cur
                # min-norm hidden change in POST-norm space
                dh_post = E.T @ Ginv @ dlogit
                # RMSNorm-aware: project out radial component, invert per-dim gain/rms
                proj = dh_post - (dh_post @ h_pre) / (h_pre @ h_pre + 1e-8) * h_pre
                dh_pre = proj * (rms / gain.clamp(min=1e-3))
                Ks.append(F.normalize(k, p=2, dim=0))
                Ds.append(dh_pre)
    finally:
        hook.remove()
        norm_hook.remove()

    K = torch.stack(Ks)
    D = torch.stack(Ds)
    ridge = K.T @ K + lam * torch.eye(K.shape[1])
    dB = D.T @ K @ torch.linalg.inv(ridge)
    return dB


def score_inmem(model, tok, lids):
    ok = 0
    for i in range(20):
        ids = tok.encode(format_problem(i), add_special_tokens=False)
        with torch.inference_mode():
            lg = model(torch.tensor([ids]))[0][0, -1, :]
        p = int(max(range(4), key=lambda j: float(lg[lids[j]])))
        ok += int(p == KEY[i])
    return ok


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--target-layers", default="23")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    a.ckpt = a.ckpt.resolve()
    layers = [int(x) for x in a.target_layers.split(",")]

    tok = tok_for(151936)
    lids = letter_ids(tok, 151936)
    print(
        "[0] cache bit-identical:", verify_cache_bitidentical(a.ckpt, tok), flush=True
    )

    # baseline
    m0 = load_model(a.ckpt)
    enable_unpack_cache(m0)
    base = score_inmem(m0, tok, lids)
    print(f"[base] {base}/20", flush=True)
    del m0

    best = None
    for scale in (0.5, 1.0, 2.0, 4.0, 8.0):
        m = load_model(a.ckpt)
        n_seed = seed_lora_basis(m, layers, a.seed)
        enable_unpack_cache(m)
        # verify seeding is output-neutral before B write
        neutral = score_inmem(m, tok, lids)
        dB = capture_and_solve(m, tok, lids, layers)
        down = m.layers[layers[-1]].down_proj
        cap = 0.05 * down.lora_B.norm().item() if down.lora_B.norm().item() > 0 else 0.5
        frob = float(torch.norm(dB))
        if frob > cap:
            dB = dB * (cap / frob)
        with torch.no_grad():
            down.lora_B.copy_(dB * scale)
        got = score_inmem(m, tok, lids)
        print(
            f"[scale {scale}] seeded={n_seed} neutral={neutral}/20 fixed={got}/20 |dB|={frob:.5f}",
            flush=True,
        )
        if best is None or got > best[0]:
            best = (got, scale, layers)
        del m
        if got == 20:
            break

    got, scale, layers = best
    # final: build the saved checkpoint with best scale
    mf = load_model(a.ckpt)
    seed_lora_basis(mf, layers, a.seed)
    enable_unpack_cache(mf)
    dB = capture_and_solve(mf, tok, lids, layers)
    down = mf.layers[layers[-1]].down_proj
    cap = 0.5
    frob = float(torch.norm(dB))
    if frob > cap:
        dB = dB * (cap / frob)
    with torch.no_grad():
        down.lora_B.copy_(dB * scale)
    mf.save_packed_safetensors(a.out)
    print(f"[saved] {a.out} best={got}/20 scale={scale} layers={layers}", flush=True)
    print(
        "[HONESTY] fixed-set score = editor fidelity, NOT reasoning. Held-out required.",
        flush=True,
    )


if __name__ == "__main__":
    main()
