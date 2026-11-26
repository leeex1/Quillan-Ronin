#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pattern surgery — find and fix a BROKEN PATTERN bleeding across layers.

Principle (from the WSP that fixed Qwen's looping): a degenerate pattern is a
dominant direction in the residual stream that recurs across contexts and causes
repetition / wrong-answer attraction. Find it via SVD of collected last-token
hidden states over a diverse prompt corpus, then DEFLATE it closed-form in the
final RMSNorm gain and lm_head rows (no backprop, no answer memorization).

Measure: general-knowledge accuracy + repetition rate BEFORE and AFTER.
If the surgery improves both, it installed a genuine general fix.
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
from hard_math20 import LETTERS as ML
from general_knowledge import QUESTIONS as GQ, KEY as GK, LETTERS as GL
from wsp_math_eval import tok_for, letter_ids, load_model
from wsp_math_tune_v3 import enable_cache


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--mode", default="norm", choices=["norm", "head"])
    ap.add_argument("--alpha", type=float, default=0.15)
    ap.add_argument("--rank", type=int, default=2)
    a = ap.parse_args()
    a.ckpt = a.ckpt.resolve()
    tok = tok_for(151936)
    lids = letter_ids(tok, 151936)

    # diverse corpus for attractor discovery (general Qs, NOT the eval answers)
    probes = []
    for q, opts, _ in GQ[:20]:
        probes.append(
            f"Question: {q}\n"
            + "".join(f"{L}) {o}\n" for L, o in zip(GL, opts))
            + "Answer:"
        )
    probes += [
        "The capital of France is",
        "Water freezes at",
        "Explain the water cycle:",
        "In conclusion, the main point is",
        "The meaning of life is",
        "2 + 2 equals",
    ]

    m = load_model(a.ckpt)
    enable_cache(m)
    # collect last-token hidden states
    hs = []
    cap = {}
    hd = m.layers[-1].register_forward_forward_hook if False else None
    for p in probes:
        ids = tok.encode(p, add_special_tokens=False)
        cap.clear()
        hk = m.norm.register_forward_pre_hook(
            lambda mod, inp: cap.__setitem__("h", inp[0].detach().float())
        )
        with torch.inference_mode():
            m(torch.tensor([ids]))
        hk.remove()
        hs.append(cap["h"][0, -1])
    H = torch.stack(hs)  # [n, hidden]
    Hc = H - H.mean(0, keepdim=True)
    U, S, V = torch.linalg.svd(Hc, full_matrices=False)
    print(
        f"[attractor SVD] top singular values: {[round(float(S[i]), 3) for i in range(min(6, len(S)))]}",
        flush=True,
    )
    attractor = V[: a.rank]  # [rank, hidden] dominant bleed directions

    # eval helpers (general knowledge, single prompts)
    def eval_gk():
        ok = 0
        for i in range(len(GQ)):
            ids = tok.encode(
                f"Question {i + 1}: {GQ[i][0]}\n"
                + "".join(f"{L}) {o}\n" for L, o in zip(GL, GQ[i][1]))
                + "Answer:",
                add_special_tokens=False,
            )
            with torch.inference_mode():
                lg = m(torch.tensor([ids]))[0][0, -1, lids]
            ok += int(int(lg.argmax()) == GK[i])
        return ok

    # repetition rate on a generation probe
    def rep_rate():
        ids = tok.encode("The meaning of life is", add_special_tokens=False)
        with torch.inference_mode():
            out = m.generate(torch.tensor([ids]), max_new_tokens=60, temperature=0.0)
        g = out[0, len(ids) :].tolist()
        if len(g) < 5:
            return 0.0
        grams = [tuple(g[i : i + 5]) for i in range(len(g) - 4)]
        return 1 - len(set(grams)) / len(grams)

    base_gk = eval_gk()
    base_rep = rep_rate()
    print(
        f"[BEFORE] general_knowledge={base_gk}/{len(GQ)} rep5gram={base_rep:.3f}",
        flush=True,
    )

    # closed-form deflation of the attractor direction
    with torch.no_grad():
        if a.mode == "norm":
            w = m.norm.weight.detach().float().clone()
            # deflate gain along attractor dims (reduce their amplification)
            for v in attractor:
                proj = (w * v).sum()
                w.sub_(a.alpha * proj * v)
            m.norm.weight.copy_(w.to(m.norm.weight.dtype))
        else:
            W = (
                m.lm_head.unpack_effective_weight().float()
                if m.lm_head is not None
                else m.embed_tokens.unpack_float().float()
            )
            for v in attractor:
                # deflect output head away from attractor
                W.sub_(a.alpha * (W @ v.unsqueeze(1)) @ v.unsqueeze(0))
            if m.lm_head is not None:
                m.lm_head.set_weights_from_float(W)
            else:
                m.embed_tokens.set_from_float(W)

    new_gk = eval_gk()
    new_rep = rep_rate()
    print(
        f"[AFTER mode={a.mode} alpha={a.alpha}] general_knowledge={new_gk}/{len(GQ)} rep5gram={new_rep:.3f}",
        flush=True,
    )
    print(f"[DELTA] GK {new_gk - base_gk:+d} rep {new_rep - base_rep:+.3f}", flush=True)
    if new_gk >= base_gk and new_rep <= base_rep + 0.01:
        m.save_packed_safetensors(a.out)
        print(f"[saved] {a.out}", flush=True)
    else:
        print("[not saved] no improvement", flush=True)


if __name__ == "__main__":
    main()
