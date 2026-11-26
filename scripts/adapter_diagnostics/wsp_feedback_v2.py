#!/usr/bin/env python3
"""WSP feedback v2: anchor OFF (it erased the write), repeated clamped writes on failures,
measure held-out transfer every step. Ground-truth error signal, zero backprop."""

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
            Ks.append(cap["k"][0, -1, :].detach().clone())  # this prompt's last-token k
    hk.remove()
    return torch.stack(L), torch.stack(Ks)


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--train-n", type=int, default=12)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--frob", type=float, default=0.010)
    ap.add_argument("--lam", type=float, default=1.0)
    a = ap.parse_args()
    tok = tok_for(151936)
    lids = letter_ids(tok, 151936)
    m = load_model(ROOT / "exports" / "quillan-mini-sovereign-bitnet.safetensors")
    enable_cache(m)
    down = m.layers[-1].down_proj
    ids = prompts(tok)
    K = torch.tensor(GK)
    tr = list(range(a.train_n))
    te = list(range(a.train_n, len(GQ)))
    E = m.embed_tokens.unpack_float().float()[lids]

    L, Ks = snap(m, ids, lids)
    acc = lambda idx: int((L[idx].argmax(-1) == K[idx]).sum())
    print(f"[init] train={acc(tr)}/{len(tr)} heldout={acc(te)}/{len(te)}", flush=True)
    for step in range(a.steps):
        L, Ks = snap(m, ids, lids)
        pred = L.argmax(-1)
        wrong = [i for i in tr if int(pred[i]) != int(K[i])]
        if not wrong:
            print(f"[step {step}] all train correct, stop", flush=True)
            break
        i = wrong[step % len(wrong)]
        k = Ks[i]
        v = E[int(K[i])]
        down._wc = None
        res = down.online_wsp_update(
            k.unsqueeze(1).contiguous(),
            v.unsqueeze(1),
            a_anchor=None,
            ridge_lambda=a.lam,
            max_rel_frob=a.frob,
        )
        L, _ = snap(m, ids, lids)  # measure after
        newpred = int(L[i].argmax())
        print(
            f"[step {step}] q{i} {LETTERS[int(pred[i])]}->{LETTERS[int(K[i])]} "
            f"rel={res['rel_frob']:.5f} now={LETTERS[newpred]} | "
            f"train={int((L[tr].argmax(-1) == K[tr]).sum())}/{len(tr)} "
            f"heldout={int((L[te].argmax(-1) == K[te]).sum())}/{len(te)}",
            flush=True,
        )


if __name__ == "__main__":
    main()
