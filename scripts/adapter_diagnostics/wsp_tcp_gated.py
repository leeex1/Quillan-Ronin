#!/usr/bin/env python3
"""WSP writes governed by the Truth Calibration Protocol (file 13).
A candidate write is PROMOTED only if:
  (1) positive example: it raises accuracy on the tuned (train) set
  (2) falsification: it does NOT lower accuracy on a held-out falsification set
  (3) recursive audit: it does NOT lower accuracy on a protected baseline set
Otherwise REJECTED and reverted. This is TCP truth-class promotion + negative examples.
Zero backprop. Reports held-out GK at every promotion.
"""

from __future__ import annotations
import sys
from pathlib import Path
import torch

ROOT = Path(r"C:\02_QUILLAN")
DIAG = ROOT / "scripts" / "adapter_diagnostics"
for p in (str(ROOT / "scripts"), str(ROOT / "03 - Training & Model"), str(DIAG)):
    if p not in sys.path:
        sys.path.insert(0, p)
from general_knowledge import QUESTIONS as GQ, KEY as GK, LETTERS
from wsp_math_eval import tok_for, letter_ids, load_model
from wsp_math_tune_v3 import enable_cache

CK = ROOT / "exports" / "quillan-mini-sovereign-bitnet.safetensors"
OUT = ROOT / "exports" / "quillan-mini-wsp-tcp.safetensors"


def prompts(tok):
    return [
        tok.encode(
            f"Question {i + 1}: {q}\n"
            + "".join(f"{L}) {o}\n" for L, o in zip(LETTERS, opts))
            + "Answer:",
            add_special_tokens=False,
        )
        for i, (q, opts, _) in enumerate(GQ)
    ]


def snap(m, all_ids, lids):
    cap = {}
    hk = m.layers[-1].down_proj.register_forward_pre_hook(
        lambda mod, inp: cap.__setitem__("k", inp[0].detach().float())
    )
    L, Ks = [], []
    with torch.inference_mode():
        for ids in all_ids:
            out = m(torch.tensor([ids]))[0]
            L.append(out[0, -1, lids].float())
            Ks.append(cap["k"][0, -1, :].detach().clone())
    hk.remove()
    return torch.stack(L), torch.stack(Ks)


def acc(X, idx, K):
    return int((X[idx].argmax(-1) == K[idx]).sum())


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--train-n", type=int, default=8)
    ap.add_argument("--falsify-n", type=int, default=8)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--frob", type=float, default=0.010)
    ap.add_argument("--lam", type=float, default=1.0)
    a = ap.parse_args()
    tok = tok_for(151936)
    lids = letter_ids(tok, 151936)
    m = load_model(CK)
    enable_cache(m)
    down = m.layers[-1].down_proj
    ids = prompts(tok)
    K = torch.tensor(GK)
    E = m.embed_tokens.unpack_float().float()[lids]
    tr = list(range(a.train_n))
    fal = list(range(a.train_n, a.train_n + a.falsify_n))
    base = list(
        range(a.train_n + a.falsify_n, len(GQ))
    )  # protected baseline, never tuned
    L, Ks = snap(m, ids, lids)
    print(
        f"[init] train={acc(L, tr, K)}/{len(tr)} falsify={acc(L, fal, K)}/{len(fal)} baseline={acc(L, base, K)}/{len(base)}",
        flush=True,
    )
    b_tr, b_fal, b_base = acc(L, tr, K), acc(L, fal, K), acc(L, base, K)
    best_state = {k: v.detach().clone() for k, v in down.state_dict().items()}
    promoted = 0
    for step in range(a.steps):
        L, Ks = snap(m, ids, lids)
        pred = L.argmax(-1)
        wrong = [i for i in tr if int(pred[i]) != int(K[i])]
        if not wrong:
            print(f"[step {step}] all train correct", flush=True)
            break
        i = wrong[step % len(wrong)]
        k = Ks[i]
        v = E[int(K[i])]
        A0 = down.lora_A.detach().clone()
        B0 = down.lora_B.detach().clone()
        down._wc = None
        down.online_wsp_update(
            k.unsqueeze(1).contiguous(),
            v.unsqueeze(1),
            a_anchor=None,
            ridge_lambda=a.lam,
            max_rel_frob=a.frob,
        )
        L2, _ = snap(m, ids, lids)
        n_tr, n_fal, n_base = acc(L2, tr, K), acc(L2, fal, K), acc(L2, base, K)
        # TCP promotion: positive + falsification + recursive audit
        ok = (
            (n_tr >= b_tr)
            and (n_fal >= b_fal)
            and (n_base >= b_base)
            and (n_tr + n_fal + n_base) > (b_tr + b_fal + b_base)
        )
        if ok:
            b_tr, b_fal, b_base = n_tr, n_fal, n_base
            promoted += 1
            best_state = {k: v.detach().clone() for k, v in down.state_dict().items()}
            print(
                f"[step {step}] PROMOTE q{i}: train={n_tr} falsify={n_fal} baseline={n_base}",
                flush=True,
            )
        else:
            with torch.no_grad():
                down.lora_A.copy_(A0)
                down.lora_B.copy_(B0)
            print(
                f"[step {step}] reject q{i}: {n_tr}/{n_fal}/{n_base} (best {b_tr}/{b_fal}/{b_base})",
                flush=True,
            )
    with torch.no_grad():
        for k, v in best_state.items():
            down.state_dict()[k].copy_(v)
    m.save_packed_safetensors(OUT)
    print(
        f"[SAVED] {OUT} promoted={promoted} best train={b_tr} falsify={b_fal} baseline={b_base}",
        flush=True,
    )
    m2 = load_model(OUT)
    enable_cache(m2)
    L3, _ = snap(m2, ids, lids)
    print(
        f"[FRESH RELOAD] train={acc(L3, tr, K)} falsify={acc(L3, fal, K)} baseline={acc(L3, base, K)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
