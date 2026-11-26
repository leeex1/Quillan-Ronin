#!/usr/bin/env python3
"""Can a closed-form read recover a,b from the model's OWN layer-8 MoE input
on a natural 'What is a + b?' prompt? If yes, Stage A can be a weight write.
Digit-disjoint holdout + wrong-target control.
"""

from __future__ import annotations
import sys, random
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
from ptp_adder import DIG


def prompt(a, b):
    return (
        f"Question: What is {a} + {b}?\n"
        + "".join(f"{L}) {v}\n" for L, v in zip(LETTERS, [0, 5, 10, 15]))
        + "Answer:"
    )


def capture(m, tok, prompts):
    cap = {}
    hk = m.layers[8].council_moe.register_forward_pre_hook(
        lambda mod, inp: cap.__setitem__("x", inp[0].detach().float())
    )
    g = {}
    for i, p in enumerate(prompts):
        g.setdefault(len(tok.encode(p, add_special_tokens=False)), []).append((i, p))
    out = [None] * len(prompts)
    for _, items in g.items():
        ids = [tok.encode(p, add_special_tokens=False) for _, p in items]
        with torch.inference_mode():
            m(torch.tensor(ids))
        x = cap["x"][:, -1, :]
        for j, (idx, _) in enumerate(items):
            out[idx] = x[j]
    hk.remove()
    return torch.stack(out)


def main():
    tok = tok_for(151936)
    m = load_model(
        Path(r"C:\02_QUILLAN\exports\quillan-mini-carved-gate010.safetensors")
    )
    enable_cache(m)
    rng = random.Random(2026)
    train = []
    test = []
    for a in range(DIG):
        for b in range(DIG):
            (train if a < 6 and b < 6 else test).append((a, b))

    # two regression targets: digit a and digit b
    def build(rows):
        X = capture(m, tok, [prompt(a, b) for a, b in rows])
        Y = torch.tensor([[a, b] for a, b in rows], dtype=torch.float32)
        return X, Y

    Xtr, Ytr = build(train)
    Xte, Yte = build(test)
    d = Xtr.shape[1]
    mu = Xtr.mean(0)
    A = Xtr - mu
    B = Xte - mu
    for lam in [0.1, 1.0, 10.0]:
        W = torch.linalg.solve(A.T @ A + lam * torch.eye(d), A.T @ Ytr)
        P = A @ W
        Q = B @ W
        # accuracy: both digits correct (rounded)
        pa = ((P.round() == Ytr).all(1)).float().mean().item()
        qa = ((Q.round() == Yte).all(1)).float().mean().item()
        ma = (P[:, 0].round() == Ytr[:, 0]).float().mean().item()
        mb = (P[:, 1].round() == Ytr[:, 1]).float().mean().item()
        print(
            f"[lam={lam:g}] train both={pa:.2f} test both={qa:.2f} | train digitA={ma:.2f} digitB={mb:.2f}",
            flush=True,
        )
    print(
        f"[VERDICT] chance both=1% | {'STAGE A FEASIBLE (digits decodable)' if qa > 0.30 else 'BLOCKED (residual does not carry digits)'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
