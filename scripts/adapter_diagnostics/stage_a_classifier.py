#!/usr/bin/env python3
"""Stage A as CLASSIFIER: can a linear read recover one-hot(a), one-hot(b)
from the model's own layer-8 MoE input on natural prompts? Misconceived regression
in stage_a_feasibility used round() on nonlinear digits; circuit actually uses one-hot.
Digit-disjoint holdout.
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
from wsp_math_eval import tok_for, load_model
from wsp_math_tune_v3 import enable_cache
from stage_a_feasibility import prompt, capture
from ptp_adder import DIG


def main():
    tok = tok_for(151936)
    m = load_model(
        Path(r"C:\02_QUILLAN\exports\quillan-mini-carved-gate010.safetensors")
    )
    enable_cache(m)
    train = [(a, b) for a in range(DIG) for b in range(DIG) if a < 6 and b < 6]
    test = [(a, b) for a in range(DIG) for b in range(DIG) if a >= 6 or b >= 6]

    def build(rows):
        X = capture(m, tok, [prompt(a, b) for a, b in rows])
        Y = torch.zeros(len(rows), 2, DIG)
        for i, (a, b) in enumerate(rows):
            Y[i, 0, a] = 1
            Y[i, 1, b] = 1
        return X, Y

    Xtr, Ytr = build(train)
    Xte, Yte = build(test)
    d = Xtr.shape[1]
    mu = Xtr.mean(0)
    A = Xtr - mu
    B = Xte - mu
    for lam in [0.01, 0.1, 1.0, 10.0]:
        W = torch.linalg.solve(
            A.T @ A + lam * torch.eye(d), A.T @ Ytr.reshape(len(Ytr), -1)
        )  # [d,20]
        P = (A @ W).view(-1, 2, DIG).argmax(-1)
        Q = (B @ W).view(-1, 2, DIG).argmax(-1)
        ytr = Ytr.argmax(-1)
        yte = Yte.argmax(-1)
        ta = ((P == ytr).all(1)).float().mean().item()
        tb = ((Q == yte).all(1)).float().mean().item()
        print(
            f"[classifier lam={lam:g}] train both={ta:.2f} held-out both={tb:.2f}",
            flush=True,
        )
    print(
        f"[VERDICT] chance both=1% | {'STAGE A FEASIBLE' if tb > 0.30 else 'BLOCKED from residual linear read'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
