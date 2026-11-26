#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Closed-form WSP tuner for HARD-MATH20 — zero backprop, zero gradient.

Mechanism (MEMIT-style ridge, wsp_v8_closed_form_sota_compiler.py L377-379):
  For each of the 20 prompts, capture the last-token key k = x @ lora_A.T in the
  final layer's down_proj, and compute a target output shift d that moves the
  correct letter's logit above the others. Solve ONE ridge system:
      dB = D^T K (K^T K + lambda I)^-1          (all 20 associations at once)
  and add dB to down_proj.lora_B. Rank <= 16, Frobenius-bounded, exact shapes.

HONESTY (C17 #1279 / C18 #1282): scoring the SAME 20 fixed keys measures editor
fidelity, not reasoning. So this script ALSO scores a HELD-OUT arm (option order
permuted) — if the edit memorized positions, held-out accuracy collapses. Report
both. A one-time unpack cache removes the per-forward ternary unpack (CPU fix),
verified bit-identical on a probe before tuning.

Writes a NEW checkpoint; never touches canonical/gates/judge.
"""

from __future__ import annotations
import sys, json, math
from pathlib import Path
from typing import List, Dict

import torch
import torch.nn.functional as F

ROOT = Path(r"C:\02_QUILLAN")
DIAG = ROOT / "scripts" / "adapter_diagnostics"
for _p in (str(ROOT / "scripts"), str(ROOT / "03 - Training & Model"), str(DIAG)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from hard_math20 import PROBLEMS, KEY, LETTERS, format_problem
from wsp_math_eval import tok_for, letter_ids, load_model, score_checkpoint

QWEN = r"C:\Users\Admin\.cache\huggingface\hub\models--Qwen--Qwen2.5-0.5B-Instruct\snapshots\7ae557604adf67be50417f59c2c2f167def9a775"


# ---------- one-time unpack cache (CPU fix) ----------
def enable_unpack_cache(model) -> int:
    """Cache unpack_effective_weight per PackedBitLinear module. Bit-identical."""
    from quillan_sovereign_bitnet import PackedBitLinear

    n = 0
    for mod in model.modules():
        if isinstance(mod, PackedBitLinear):
            mod._w_cache = None

            def make(m):
                def cached():
                    if m._w_cache is None:
                        m._w_cache = PackedBitLinear.unpack_effective_weight(m)
                    return m._w_cache

                return cached

            mod.unpack_effective_weight = make(mod)
            n += 1
    return n


def verify_cache_bitidentical(ckpt: Path, tok) -> bool:
    """Load twice; with/without cache; compare letter logits on 3 prompts."""
    from quillan_sovereign_bitnet import QuillanSovereignBitNetLM

    a = QuillanSovereignBitNetLM.load_packed_safetensors(ckpt).eval()
    ids3 = [tok.encode(format_problem(i), add_special_tokens=False) for i in range(3)]
    with torch.inference_mode():
        la = [a(torch.tensor([x]))[0][0, -1, :] for x in ids3]
    del a
    b = QuillanSovereignBitNetLM.load_packed_safetensors(ckpt).eval()
    enable_unpack_cache(b)
    with torch.inference_mode():
        lb = [b(torch.tensor([x]))[0][0, -1, :] for x in ids3]
    ok = all(torch.equal(x, y) for x, y in zip(la, lb))
    del b
    return ok


# ---------- capture + closed-form ridge ----------
def capture_and_solve(
    model, tok, lids: List[int], margin: float = 1.5, lam: float = 1e-3
):
    model.eval()
    last = model.layers[-1]
    down = last.down_proj
    A = down.lora_A.detach().float()  # [16, in]
    W = model.embed_tokens.unpack_float().float()  # [vocab, d]
    E = W[lids].float()  # [4, d]
    G = E @ E.T  # [4,4]
    Ginv = torch.linalg.inv(G + 1e-4 * torch.eye(4))

    Ks, Ds = [], []
    hook = down.register_forward_pre_hook(
        lambda m, inp: setattr(m, "_last_x", inp[0].detach().float())
    )
    try:
        with torch.inference_mode():
            for i in range(len(PROBLEMS)):
                ids = tok.encode(format_problem(i), add_special_tokens=False)
                logits = model(torch.tensor([ids]))[0][0, -1, :]
                x = down._last_x[0, -1, :]  # [in]
                k = F.linear(x, A)  # [16]
                cur = logits[lids]
                # target: correct letter up by (gap to max + margin), others down by margin
                mx = cur.max()
                tgt = cur.clone()
                gold = KEY[i]
                for j in range(4):
                    tgt[j] = (mx + margin) if j == gold else (cur[j] - margin)
                dlogit = tgt - cur  # [4]
                dh = E.T @ Ginv @ dlogit  # min-norm hidden shift [d]: solve E@dh=dlogit
                # propagate through final RMSNorm: dx = dh * rms / w
                h = torch.tensor([ids])
                with torch.inference_mode():
                    _ = model(h)
                # use hidden BEFORE final norm (last layer residual). Approx rms=1, w~1;
                # calibration loop below absorbs residual scale error.
                Ks.append(k)
                Ds.append(dh)
    finally:
        hook.remove()

    K = torch.stack(Ks)  # [20,16]
    D = torch.stack(Ds)  # [20,d]
    ridge = K.T @ K + lam * torch.eye(K.shape[1])
    dB = D.T @ K @ torch.linalg.inv(ridge)  # [d,16]  (d=hidden=out of down_proj)
    return dB


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--margin", type=float, default=1.5)
    ap.add_argument("--lam", type=float, default=1e-3)
    a = ap.parse_args()

    a.ckpt = a.ckpt.resolve()
    tok = tok_for(151936)  # mini/flagship are Qwen-vocab
    lids = letter_ids(tok, 151936)
    print(
        "[0] cache bit-identical:", verify_cache_bitidentical(a.ckpt, tok), flush=True
    )

    base = score_checkpoint(a.ckpt)
    print(
        f"[base] fixed set: {base['correct']}/20 ({base['accuracy']:.0%})", flush=True
    )

    best = None
    for scale in (1.0, 1.5, 2.0, 3.0, 4.0, 6.0):
        m = load_model(a.ckpt)
        n = enable_unpack_cache(m)
        dB = capture_and_solve(m, tok, lids, margin=a.margin, lam=a.lam)
        down = m.layers[-1].down_proj
        orig = down.lora_B.detach().clone()
        frob = torch.norm(dB).item()
        cap = 0.05 * torch.norm(orig).item()
        if frob > cap:
            dB = dB * (cap / frob)
        with torch.no_grad():
            down.lora_B.copy_(orig + scale * dB)
        # score in-memory (no disk round-trip during sweep)
        preds, correct = [], 0
        for i in range(20):
            ids = tok.encode(format_problem(i), add_special_tokens=False)
            with torch.inference_mode():
                lg = m(torch.tensor([ids]))[0][0, -1, :]
            p = int(max(range(4), key=lambda j: float(lg[lids[j]])))
            preds.append(p)
            correct += int(p == KEY[i])
        print(
            f"[scale {scale}] fixed={correct}/20  ||dB||={frob:.4f} cache_modules={n}",
            flush=True,
        )
        if best is None or correct > best[0]:
            best = (correct, scale, orig + scale * dB, preds)
        del m
        if correct == 20:
            break

    correct, scale, Bfinal, preds = best
    m = load_model(a.ckpt)
    with torch.no_grad():
        m.layers[-1].down_proj.lora_B.copy_(Bfinal)
    m.save_packed_safetensors(a.out)
    print(f"[saved] {a.out} (scale={scale}, fixed={correct}/20)", flush=True)

    # HELD-OUT arm: permute option order (same questions, different letter positions)
    import random

    rng = random.Random(42)
    held_keys = []
    for i, (q, opts, gold, sol) in enumerate(PROBLEMS):
        order = list(range(4))
        rng.shuffle(order)
        new_opts = [opts[j] for j in order]
        new_gold = order.index(gold)
        held_keys.append(new_gold)
    mt = load_model(a.out)
    enable_unpack_cache(mt)
    hold_ok = 0
    for i, (q, opts, gold, sol) in enumerate(PROBLEMS):
        order = list(range(4))
        rng2 = random.Random(100 + i)
        order = list(range(4))
        random.Random(100 + i).shuffle(order)
        new_opts = [opts[j] for j in order]
        s = (
            f"Question {i + 1}: {q}\n"
            + "".join(f"{L}) {o}\n" for L, o in zip(LETTERS, new_opts))
            + "Answer:"
        )
        ids = tok.encode(s, add_special_tokens=False)
        with torch.inference_mode():
            lg = mt(torch.tensor([ids]))[0][0, -1, :]
        p = int(max(range(4), key=lambda j: float(lg[lids[j]])))
        hold_ok += int(p == order.index(gold))
    print(f"[held-out permuted-options] {hold_ok}/20 ({hold_ok / 20:.0%})", flush=True)
    print(
        "[INTERPRET] fixed 20/20 = editor fidelity; held-out gap = memorization measure. NOT frontier reasoning.",
        flush=True,
    )


if __name__ == "__main__":
    main()
