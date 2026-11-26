#!/usr/bin/env python3
"""Model reads / writes / updates its OWN weights in real time, zero backprop.

Loop per step:
  1. READ   : capture the model's own last-token activation k (down_proj input)
  2. TARGET: derive v from the model's OWN next-token distribution (self-supervised)
  3. WRITE  : layer.online_wsp_update(k, v)  -> closed-form Woodbury ridge into LoRA A/B
  4. RE-READ: measure the model's output before/after the self-write

Everything runs under torch.inference_mode(); no gradients, no optimizer, no SFT.
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
from wsp_math_eval import tok_for, letter_ids, load_model
from wsp_math_tune_v3 import enable_cache
from general_knowledge import QUESTIONS as GQ, KEY as GK, LETTERS


def snap(m, tok, lids, ids):
    cap = {}
    hk = m.layers[-1].down_proj.register_forward_pre_hook(
        lambda mod, inp: cap.__setitem__("k", inp[0].detach().float())
    )
    with torch.inference_mode():
        out = m(torch.tensor([ids]))[0]
    hk.remove()
    lg = out[0, -1, lids].float()
    return lg, cap["k"][:, -1, :].detach().clone()


def score(m, tok, lids):
    ok = 0
    for i, (q, opts, _) in enumerate(GQ):
        ids = tok.encode(
            f"Question {i + 1}: {q}\n"
            + "".join(f"{L}) {o}\n" for L, o in zip(LETTERS, opts))
            + "Answer:",
            add_special_tokens=False,
        )
        with torch.inference_mode():
            lg = m(torch.tensor([ids]))[0][0, -1, lids]
        ok += int(int(lg.argmax()) == GK[i])
    return ok


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--ckpt",
        type=Path,
        default=Path(r"C:\02_QUILLAN\exports\quillan-mini-rank64.safetensors"),
    )
    ap.add_argument("--steps", type=int, default=6)
    a = ap.parse_args()
    tok = tok_for(151936)
    lids = letter_ids(tok, 151936)
    m = load_model(a.ckpt)
    enable_cache(m)
    down = m.layers[-1].down_proj
    print(f"[self-modify] rank={down.eggroll_rank} ckpt={a.ckpt.name}", flush=True)
    base = score(m, tok, lids)
    print(f"[before] GK={base}/{len(GQ)}", flush=True)

    # self-supervised prompts: a subset of GK the model itself is confident about
    for step in range(a.steps):
        i = step % len(GQ)
        q, opts, _ = GQ[i]
        ids = tok.encode(
            f"Question {i + 1}: {q}\n"
            + "".join(f"{L}) {o}\n" for L, o in zip(LETTERS, opts))
            + "Answer:",
            add_special_tokens=False,
        )
        lg, k = snap(m, tok, lids, ids)
        # TARGET = sharpen the model's own distribution (its own belief), closed-form
        p = F.softmax(lg, 0)
        target = lg + (p - lg.mean()) * 2.0  # self-consistency target
        # v_target must match down_proj output dim (= hidden size)
        E = m.embed_tokens.unpack_float().float()[lids]  # [4, hidden]
        pred = int(lg.argmax())  # model's OWN belief (no answer key)
        conf = float(p[pred])
        v = E[pred].clone()  # reinforce its own belief
        if conf < 0.30:  # self-confidence gate
            print(
                f"[step {step}] skipped (own-confidence {conf:.2f} < 0.30)", flush=True
            )
            continue
        # READ -> WRITE (closed-form self update)
        down._wc = None
        # k=[1,4864] -> [in_features=4864, n_tok=1]; v=[896] -> [out_features=896, n_tok=1]
        res = down.online_wsp_update(
            k.T.contiguous(), v.unsqueeze(1), ridge_lambda=8.0, max_rel_frob=0.0030
        )
        cur = score(m, tok, lids)
        print(
            f"[step {step}] self-write applied={res['applied']:.0f} rel_frob={res['rel_frob']:.5f} -> GK={cur}/{len(GQ)}",
            flush=True,
        )

    final = score(m, tok, lids)
    print(f"[VERDICT] before={base} after={final} delta={final - base:+d}", flush=True)


if __name__ == "__main__":
    main()
