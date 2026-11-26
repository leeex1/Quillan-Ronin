#!/usr/bin/env python3
"""WSP-style ITERATIVE closed-form weight tuning with ground-truth feedback.
No backprop. On each failure: capture activation k, target = correct answer dir,
write closed-form ridge correction via online_wsp_update, retest. Measure held-out.
This is the user's 'try 1x4->3, adjust, ->5, adjust, ->4' loop in weight space.
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
from general_knowledge import QUESTIONS as GQ, KEY as GK, LETTERS
from wsp_math_eval import tok_for, letter_ids, load_model
from wsp_math_tune_v3 import enable_cache


def prompts_lids(tok, lids):
    ps = []
    for i, (q, opts, _) in enumerate(GQ):
        ps.append(
            f"Question {i + 1}: {q}\n"
            + "".join(f"{L}) {o}\n" for L, o in zip(LETTERS, opts))
            + "Answer:"
        )
    return [tok.encode(p, add_special_tokens=False) for p in ps]


def snap(m, tok, lids, all_ids):
    cap = {}
    hk = m.layers[-1].down_proj.register_forward_pre_hook(
        lambda mod, inp: cap.__setitem__("k", inp[0].detach().float())
    )
    L = []
    Ks = []
    with torch.inference_mode():
        for ids in all_ids:
            out = m(torch.tensor([ids]))[0]
            L.append(out[0, -1, lids].float())
            Ks.append(cap["k"][0, -1].detach().clone())
    hk.remove()
    return torch.stack(L), torch.stack(Ks)


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--ckpt",
        type=Path,
        default=ROOT / "exports" / "quillan-mini-sovereign-bitnet.safetensors",
    )
    ap.add_argument("--train-n", type=int, default=12)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--lam", type=float, default=8.0)
    ap.add_argument("--frob", type=float, default=0.0030)
    a = ap.parse_args()
    tok = tok_for(151936)
    lids = letter_ids(tok, 151936)
    m = load_model(a.ckpt)
    enable_cache(m)
    down = m.layers[-1].down_proj
    all_ids = prompts_lids(tok, lids)
    K = torch.tensor(GK)
    train_idx = list(range(a.train_n))
    test_idx = list(range(a.train_n, len(GQ)))
    E = m.embed_tokens.unpack_float().float()[lids]  # [4,hidden] answer dirs

    L, Ks = snap(m, tok, lids, all_ids)

    def acc(idx):
        return int((L[idx].argmax(-1) == K[idx]).sum())

    print(
        f"[init] train={acc(train_idx)}/{len(train_idx)} heldout={acc(test_idx)}/{len(test_idx)}",
        flush=True,
    )
    anchor = Ks[train_idx].contiguous()  # [n,4864] so SVD V=[4864,q] for dw@V

    for step in range(a.steps):
        L, Ks = snap(m, tok, lids, all_ids)
        pred = L.argmax(-1)
        wrong = [i for i in train_idx if int(pred[i]) != int(K[i])]
        if not wrong:
            print(f"[step {step}] all train correct; stop", flush=True)
            break
        i = wrong[0]  # target the FIRST failure (WSP-style one at a time)
        k = Ks[i]  # [4864]
        v = E[int(K[i])]  # [896] correct answer direction
        down._wc = None
        res = down.online_wsp_update(
            k.unsqueeze(1).contiguous(),
            v.unsqueeze(1),
            a_anchor=anchor,
            ridge_lambda=a.lam,
            max_rel_frob=a.frob,
        )
        L2, Ks2 = snap(m, tok, lids, all_ids)
        L = L2
        Ks = Ks2
        tr = acc(train_idx)
        te = acc(test_idx)
        print(
            f"[step {step}] fix q{i} ({LETTERS[int(pred[i])]}->{LETTERS[int(K[i])]}) "
            f"applied={res['applied']:.0f} rel_frob={res['rel_frob']:.5f} | "
            f"train={tr}/{len(train_idx)} heldout={te}/{len(test_idx)}",
            flush=True,
        )


if __name__ == "__main__":
    main()
