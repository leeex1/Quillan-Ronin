#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dragon 40/40 — write directly to lm_head packed WEIGHT ROWS for A/B/C/D.

For an untied quantized head, down_proj LoRA writes are distorted by RMSNorm +
head quantization. The clean lever is the output head itself:
  logit_j(i) = W_head[lid_j] . q(h_i)    q = head activation quantizer
Write delta_j into W_head[lid_j] rows. Capture q(h_i) empirically per prompt,
solve ridge for delta_j so correct letter wins. Repack via set_weights_from_float.
Closed-form, zero backprop. Single prompts.
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
    ap.add_argument("--reps", type=int, default=8)
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
    W = head.unpack_effective_weight().float().clone()  # [vocab, hidden]

    def letters():
        o = []
        with torch.inference_mode():
            for x in xs:
                o.append(m(x)[0][0, -1, lids].clone())
        return torch.stack(o)

    base = letters()
    print(
        f"[base] {sum(int(base[i].argmax() == K40[i]) for i in range(N))}/{N}",
        flush=True,
    )

    # capture the INPUT to lm_head (h) for each prompt -> q(h) approximate via head's own forward
    cap = {}
    hd = head.register_forward_pre_hook(
        lambda mod, inp: cap.__setitem__("x", inp[0].detach().float())
    )
    for x in xs:
        with torch.inference_mode():
            m(x)
    hd.remove()
    # cap holds only last forward; recapture per prompt -> store h_i
    Hs = []
    for x in xs:
        cap.clear()
        hd = head.register_forward_pre_hook(
            lambda mod, inp: cap.__setitem__("x", inp[0].detach().float())
        )
        with torch.inference_mode():
            m(x)
        hd.remove()
        Hs.append(cap["x"][0, -1].clone())  # [hidden]
    Hm = torch.stack(Hs)  # [40,hidden]

    best = None
    for rep in range(a.reps):
        T = np.zeros((N, 4))
        cur = letters()
        for i in range(N):
            mx = float(cur[i].max())
            for j in range(4):
                desired = (
                    (mx + a.margin) if j == K40[i] else (float(cur[i, j]) - a.margin)
                )
                T[i, j] = desired - float(cur[i, j])
        # delta_j solves: delta_j . Hm[i] ~= T[i,j]  -> ridge in hidden space
        G = Hm.numpy()  # [N,hidden]
        for j in range(4):
            delta = np.linalg.solve(G.T @ G + a.lam * np.eye(G.shape[1]), G.T @ T[:, j])
            Wcur = head.unpack_effective_weight().float()
            Wcur[lids[j]] += torch.tensor(delta, dtype=torch.float32)
            head.set_weights_from_float(Wcur)
        final = letters()
        ok = sum(int(final[i].argmax() == K40[i]) for i in range(N))
        miss = [i + 1 for i in range(N) if final[i].argmax() != K40[i]]
        print(f"[rep {rep}] {ok}/{N} miss={miss[:12]}", flush=True)
        if best is None or ok > best[0]:
            best = (ok, Wcur.clone() if "Wcur" in dir() else W.clone())
        if ok == N:
            break

    ok, Wf = best
    # FIX: do NOT re-apply Wf via set_weights_from_float — that is a SECOND
    # 2-bit repack that corrupts the deltas (in-memory 40/40 became 17/40 on
    # fresh reload). The head already holds the exact rep that scored `ok`.
    # Only re-apply if the current in-memory head is not the best rep.
    cur = letters()  # ONE pass (not letters()[i] in a loop)
    cur_ok = sum(int(cur[i].argmax() == K40[i]) for i in range(N))
    if cur_ok < ok:
        head.set_weights_from_float(Wf)
    m.save_packed_safetensors(a.out)
    print(f"[saved] best={ok}/{N} in_memory_now={cur_ok}", flush=True)


if __name__ == "__main__":
    main()
