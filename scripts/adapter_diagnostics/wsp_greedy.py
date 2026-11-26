#!/usr/bin/env python3
"""WSP greedy: propose a closed-form write on each failure, ACCEPT only if it raises
train accuracy (no held-out peeking for acceptance), save best state to checkpoint.
Then report held-out at the best state. Zero backprop.
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
OUT = ROOT / "exports" / "quillan-mini-wsp-greedy.safetensors"


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


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--train-n", type=int, default=12)
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
    tr = list(range(a.train_n))
    te = list(range(a.train_n, len(GQ)))
    E = m.embed_tokens.unpack_float().float()[lids]
    L, Ks = snap(m, ids, lids)
    acc_tr = lambda X: int((X[tr].argmax(-1) == K[tr]).sum())
    acc_te = lambda X: int((X[te].argmax(-1) == K[te]).sum())
    best_tr = acc_tr(L)
    best_te = acc_te(L)
    print(f"[init] train={best_tr}/{len(tr)} heldout={best_te}/{len(te)}", flush=True)
    best_state = {k: v.detach().clone() for k, v in down.state_dict().items()}
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
        # snapshot current A/B for reject
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
        tr2 = acc_tr(L2)
        if tr2 > best_tr:
            best_tr = tr2
            best_te = acc_te(L2)
            best_state = {k: v.detach().clone() for k, v in down.state_dict().items()}
            print(
                f"[step {step}] ACCEPT q{i} train={tr2} heldout={best_te}", flush=True
            )
        else:
            with torch.no_grad():
                down.lora_A.copy_(A0)
                down.lora_B.copy_(B0)
            print(f"[step {step}] reject q{i} train={tr2} (best {best_tr})", flush=True)
    # restore best and save whole model
    with torch.no_grad():
        for k, v in best_state.items():
            down.state_dict()[k].copy_(v)
    m.save_packed_safetensors(OUT)
    print(
        f"[SAVED] {OUT} best_train={best_tr}/{len(tr)} best_heldout={best_te}/{len(te)}",
        flush=True,
    )
    # fresh reload verify
    m2 = load_model(OUT)
    enable_cache(m2)
    L3, _ = snap(m2, ids, lids)
    print(
        f"[FRESH RELOAD] train={acc_tr(L3)}/{len(tr)} heldout={acc_te(L3)}/{len(te)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
