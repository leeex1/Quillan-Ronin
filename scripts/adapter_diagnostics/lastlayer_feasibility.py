#!/usr/bin/env python3
"""Test placing the circuit at the LAST layer (23):
  (A) are digits decodable from layer-23 MoE input (classifier, digit-disjoint)?
  (C) if digit one-hot is injected at layer 23, does the sum survive to the head
      (linear readout on the final hidden, held-out)?
No injection = natural text. Injection is a diagnostic (explicitly not weight-installed).
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

LAYER = 23


def prompt(a, b):
    return (
        f"Question: What is {a} + {b}?\n"
        + "".join(f"{L}) {v}\n" for L, v in zip(LETTERS, [0, 5, 10, 15]))
        + "Answer:"
    )


def capture(m, tok, prompts, inject_pairs=None):
    cap = {}
    hk = m.norm.register_forward_pre_hook(
        lambda mod, inp: cap.__setitem__("h", inp[0].detach().float())
    )
    mh = {}
    if m.layers[LAYER].council_moe is not None:

        def pre(mod, inp):
            x = inp[0].clone()
            idx = state.get("idx", 0)
            if inject_pairs is not None and idx < len(inject_pairs):
                a, b = inject_pairs[idx]
                x[0, -1, :DIG] = 0
                x[0, -1, DIG : 2 * DIG] = 0
                x[0, -1, a] = 1.0
                x[0, -1, DIG + b] = 1.0
            state["idx"] = idx + 1
            return (x,)

        state = {"idx": 0}
        mh = m.layers[LAYER].council_moe.register_forward_pre_hook(pre)
    g = {}
    for i, p in enumerate(prompts):
        g.setdefault(len(tok.encode(p, add_special_tokens=False)), []).append((i, p))
    outH = {}
    for _, items in g.items():
        ids = [tok.encode(p, add_special_tokens=False) for _, p in items]
        with torch.inference_mode():
            m(torch.tensor(ids))
        h = cap["h"][:, -1, :]
        for j, (idx, _) in enumerate(items):
            outH[idx] = h[j]
    if mh:
        mh.remove()
    hk.remove()
    return torch.stack([outH[i] for i in range(len(prompts))])


def main():
    tok = tok_for(151936)
    m = load_model(
        Path(r"C:\02_QUILLAN\exports\quillan-mini-carved-gate010.safetensors")
    )
    enable_cache(m)
    train = [(a, b) for a in range(DIG) for b in range(DIG) if a < 6 and b < 6]
    test = [(a, b) for a in range(DIG) for b in range(DIG) if a >= 6 or b >= 6]
    # (A) digits from final hidden, natural text
    Xtr = capture(m, tok, [prompt(a, b) for a, b in train])
    Xte = capture(m, tok, [prompt(a, b) for a, b in test])
    d = Xtr.shape[1]
    mu = Xtr.mean(0)
    A = Xtr - mu
    B = Xte - mu
    Ytr = torch.zeros(len(train), 2, DIG)
    Yte = torch.zeros(len(test), 2, DIG)
    for i, (a, b) in enumerate(train):
        Ytr[i, 0, a] = 1
        Ytr[i, 1, b] = 1
    for i, (a, b) in enumerate(test):
        Yte[i, 0, a] = 1
        Yte[i, 1, b] = 1
    W = torch.linalg.solve(
        A.T @ A + 1.0 * torch.eye(d), A.T @ Ytr.reshape(len(Ytr), -1)
    )
    pa = (
        ((A @ W).view(-1, 2, DIG).argmax(-1) == Ytr.argmax(-1))
        .all(1)
        .float()
        .mean()
        .item()
    )
    pb = (
        ((B @ W).view(-1, 2, DIG).argmax(-1) == Yte.argmax(-1))
        .all(1)
        .float()
        .mean()
        .item()
    )
    print(f"[A: digits from final hidden] train={pa:.2f} held-out={pb:.2f}", flush=True)


if __name__ == "__main__":
    main()
