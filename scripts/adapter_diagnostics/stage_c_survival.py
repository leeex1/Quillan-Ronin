#!/usr/bin/env python3
"""Full-pipeline diagnostic: with digit features injected at the layer-8 MoE input
(hook, NOT weight-installed - explicit), does the circuit's sum signal survive to
the final hidden state such that a closed-form head readout decodes a+b on HELD-OUT
digit pairs? Also compare no-injection baseline. Stage C feasibility.
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
from hard_math20 import LETTERS
from wsp_math_eval import tok_for, load_model
from wsp_math_tune_v3 import enable_cache
from ptp_adder import DIG, SUMS


def prompt(a, b):
    return (
        f"Question: What is {a} + {b}?\n"
        + "".join(f"{L}) {v}\n" for L, v in zip(LETTERS, [0, 5, 10, 15]))
        + "Answer:"
    )


def capture(m, tok, prompts, inject):
    cap = {}
    hooks = []
    if inject:

        def mk():
            state = {}

            def pre(mod, inp):
                x = inp[0].clone()
                if "pair" in state:
                    a, b = state["pair"]
                    x = x.clone()
                    x[0, -1, :DIG] = 0
                    x[0, -1, DIG : 2 * DIG] = 0
                    x[0, -1, a] = 1.0
                    x[0, -1, DIG + b] = 1.0
                return (x,)

            return pre, state

        pre, state = mk()
        hooks.append(m.layers[8].council_moe.register_forward_pre_hook(pre))
    hk = m.norm.register_forward_pre_hook(
        lambda mod, inp: cap.__setitem__("h", inp[0].detach().float())
    )
    g = {}
    for i, p in enumerate(prompts):
        g.setdefault(len(tok.encode(p, add_special_tokens=False)), []).append((i, p))
    out = {}
    for _, items in g.items():
        ids = [tok.encode(p, add_special_tokens=False) for _, p in items]
        # pair order matches items; attach to injection state in order
        with torch.inference_mode():
            m(torch.tensor(ids))
        h = cap["h"][:, -1, :]
        for j, (idx, _) in enumerate(items):
            out[idx] = h[j]
    for hh in hooks:
        hh.remove()
    hk.remove()
    return torch.stack([out[i] for i in range(len(prompts))])


def main():
    tok = tok_for(151936)
    m = load_model(
        Path(r"C:\02_QUILLAN\exports\quillan-mini-carved-gate010.safetensors")
    )
    enable_cache(m)
    train = [(a, b) for a in range(DIG) for b in range(DIG) if a < 6 and b < 6]
    test = [(a, b) for a in range(DIG) for b in range(DIG) if a >= 6 or b >= 6]
    trp = [prompt(a, b) for a, b in train]
    tep = [prompt(a, b) for a, b in test]

    # NOTE: injection needs per-prompt pair; simplify by injecting via direct feature vector
    # through the MoE is not possible in batched form here -> use manual single-pass injection
    def inject_capture(rows):
        Hs = []
        moe = m.layers[8].council_moe
        for a, b in rows:
            state = {"pair": (a, b)}

            def pre(mod, inp, state=state):
                x = inp[0].clone()
                x[0, -1, :DIG] = 0
                x[0, -1, DIG : 2 * DIG] = 0
                x[0, -1, a] = 1.0
                x[0, -1, DIG + b] = 1.0
                return (x,)

            hh = moe.register_forward_pre_hook(pre)
            cap = {}
            hk = m.norm.register_forward_pre_hook(
                lambda mod, inp: cap.__setitem__("h", inp[0].detach().float())
            )
            ids = tok.encode(prompt(a, b), add_special_tokens=False)
            with torch.inference_mode():
                m(torch.tensor([ids]))
            hh.remove()
            hk.remove()
            Hs.append(cap["h"][0, -1])
        return torch.stack(Hs)

    Xtr = inject_capture(train)
    Xte = inject_capture(test)
    Ytr = torch.tensor([a + b for a, b in train])
    Yte = torch.tensor([a + b for a, b in test])
    d = Xtr.shape[1]
    mu = Xtr.mean(0)
    A = Xtr - mu
    B = Xte - mu
    # classify sum into 19 classes
    for lam in [1.0, 10.0, 100.0]:
        Yoh = torch.zeros(len(Ytr), SUMS)
        Yoh.scatter_(1, Ytr.unsqueeze(1), 1)
        W = torch.linalg.solve(A.T @ A + lam * torch.eye(d), A.T @ Yoh)
        ta = ((A @ W).argmax(1) == Ytr).float().mean().item()
        tb = ((B @ W).argmax(1) == Yte).float().mean().item()
        print(
            f"[injected lam={lam:g}] sum train={ta:.2f} held-out={tb:.2f}", flush=True
        )
    print(
        f"[VERDICT] chance sum=1/19=5% | {'STAGE C FEASIBLE: circuit sum signal survives to output' if tb > 0.30 else 'BLOCKED: signal does not survive'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
