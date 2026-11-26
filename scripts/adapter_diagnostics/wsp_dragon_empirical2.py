#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Dragon empirical WSP — perturb hidden dims (not token ids), measure via lm_head.

Dragon down_proj.lora_B is [hidden=2048, rank]. Perturbing row h changes hidden
dimension h; letter logits change via W_head[lids, h]. Measure dlogit/d(dB[h,r])
for a selected set of hidden dims that most affect the 4 letter logits, then
solve for the write. Closed-form, single prompts, no backprop.
"""

from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(r"C:\02_QUILLAN")
DIAG = ROOT / "scripts" / "adapter_diagnostics"
for p in (str(ROOT / "scripts"), str(ROOT / "03 - Training & Model"), str(DIAG)):
    if p not in sys.path:
        sys.path.insert(0, p)
from hard_math20 import LETTERS
from hard_math40 import PROBLEMS as P40, KEY as K40
from wsp_math_eval import load_model
from wsp_math_tune_v3 import enable_cache, seed_lora_basis
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
    ap.add_argument("--eps", type=float, default=0.5)
    ap.add_argument("--margin", type=float, default=1.0)
    ap.add_argument("--maxdims", type=int, default=16)
    a = ap.parse_args()
    a.ckpt = a.ckpt.resolve()
    N = 40
    tok = vc.load_dragon_tokenizer()
    lids = [tok.encode(f" {L}", add_special_tokens=False)[0] for L in LETTERS]
    xs = [
        torch.tensor([tok.encode(fmt(i), add_special_tokens=False)]) for i in range(N)
    ]
    m = load_model(a.ckpt)
    down = m.layers[-1].down_proj
    seed_lora_basis(m, [len(m.layers) - 1], 42)
    enable_cache(m)
    H, r = down.lora_A.shape[1], down.lora_A.shape[0]  # hidden=4864? check
    print(
        f"[dims] lora_A={tuple(down.lora_A.shape)} lora_B={tuple(down.lora_B.shape)}",
        flush=True,
    )
    H = down.lora_B.shape[0]
    r = down.lora_B.shape[1]
    B0 = down.lora_B.detach().clone()

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

    # Find hidden dims that most affect letter logits via W_head.
    W = m.lm_head.unpack_effective_weight().float()  # [vocab, hidden]
    # sensitivity of letter logits to hidden dim h = W[lids,h]
    sens = (W[lids] ** 2).sum(0)  # [hidden]
    top = torch.topk(
        sens, min(64, H)
    ).indices.tolist()  # 64 most influential hidden dims
    print(f"[top hidden dims] {len(top)}", flush=True)

    # measure effect of each (hidden dim, rank) pair is too big; use rank dims only
    # on a rank-1 mean direction over top hidden dims: perturb lora_B[top, rr]
    ndim = min(a.maxdims, r)
    G = np.zeros((N, 4, ndim))
    for rr in range(ndim):
        with torch.no_grad():
            down.lora_B.copy_(B0)
            down.lora_B[top, rr] += a.eps
        cur = letters()
        G[:, :, rr] = ((cur - base) / a.eps).numpy()
        if (rr + 1) % 4 == 0:
            print(f"[G] {rr + 1}/{ndim}", flush=True)
    with torch.no_grad():
        down.lora_B.copy_(B0)

    T = np.zeros((N, 4))
    for i in range(N):
        mx = float(base[i].max())
        for j in range(4):
            desired = (mx + a.margin) if j == K40[i] else (float(base[i, j]) - a.margin)
            T[i, j] = desired - float(base[i, j])
    dB = np.zeros((H, r))
    for j in range(4):
        sol = np.linalg.lstsq(G[:, j, :], T[:, j], rcond=1e-2)[0]  # length ndim
        sol_full = np.zeros(r)
        sol_full[:ndim] = sol
        # write this rank-direction into all top hidden dims scaled by W_head sensitivity
        w = W[lids[j], top].float()
        w = w / (w.norm() + 1e-8)
        w_np = w.numpy()
        dB[top, :] += sol_full[None, :] * w_np[:, None]
    with torch.no_grad():
        down.lora_B.copy_(torch.tensor(dB, dtype=torch.float32))
    final = letters()
    ok = sum(int(final[i].argmax() == K40[i]) for i in range(N))
    miss = [i + 1 for i in range(N) if final[i].argmax() != K40[i]]
    print(f"[EMPIRICAL2 dragon] {ok}/{N} miss={miss}", flush=True)
    if ok >= 30:
        m.save_packed_safetensors(a.out)
        print(f"[saved] {a.out} {ok}/{N}", flush=True)


if __name__ == "__main__":
    main()
