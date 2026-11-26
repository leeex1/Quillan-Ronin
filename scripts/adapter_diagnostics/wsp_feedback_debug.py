#!/usr/bin/env python3
"""Debug why the WSP feedback write does not flip the answer: anchor on/off, lam sweep, logit delta."""

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

tok = tok_for(151936)
lids = letter_ids(tok, 151936)


def prompt(i):
    q, opts, _ = GQ[i]
    return (
        f"Question {i + 1}: {q}\n"
        + "".join(f"{L}) {o}\n" for L, o in zip(LETTERS, opts))
        + "Answer:"
    )


ids = [tok.encode(prompt(i), add_special_tokens=False) for i in range(len(GQ))]
E = None


def snap(m):
    cap = {}
    hk = m.layers[-1].down_proj.register_forward_pre_hook(
        lambda mod, inp: cap.__setitem__("k", inp[0].detach().float())
    )
    L = []
    with torch.inference_mode():
        for x in ids:
            out = m(torch.tensor([x]))[0]
            L.append(out[0, -1, lids].float())
    hk.remove()
    return torch.stack(L), cap["k"][0, -1].detach().clone()


for use_anchor in [True, False]:
    for lam in [8.0, 1.0, 0.1]:
        m = load_model(ROOT / "exports" / "quillan-mini-sovereign-bitnet.safetensors")
        enable_cache(m)
        down = m.layers[-1].down_proj
        E = m.embed_tokens.unpack_float().float()[lids]
        L0, k0 = snap(m)
        i = 0
        pred = int(L0[i].argmax())
        tgt = int(GK[i])
        v = E[tgt]
        anchor = k0.unsqueeze(0).contiguous() if use_anchor else None
        down._wc = None
        res = down.online_wsp_update(
            k0.unsqueeze(1).contiguous(),
            v.unsqueeze(1),
            a_anchor=anchor,
            ridge_lambda=lam,
            max_rel_frob=0.0030,
        )
        L1, _ = snap(m)
        dlog = L1[i] - L0[i]
        print(
            f"anchor={use_anchor} lam={lam} rel_frob={res['rel_frob']:.6f} "
            f"pred {pred}->{int(L1[i].argmax())} (tgt {tgt}) "
            f"dlogit[tgt]={float(dlog[tgt]):+.4f} dlogit_max={float(dlog.abs().max()):.4f}",
            flush=True,
        )
