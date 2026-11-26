#!/usr/bin/env python3
"""FAST decisive test: is a+b linearly decodable from BASE last-token hidden state?

All prompts use 2-digit operands => identical token length => ONE batched forward.
Controls: digit-disjoint holdout, wrong-target, random-direction, permutation.
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

SUMS = [20, 30, 40, 50]


def prompt(a, b, perm):
    opts = [SUMS[i] for i in perm]
    return (
        "Question: What is %02d + %02d?\n" % (a, b)
        + "".join(f"{L}) {v}\n" for L, v in zip(LETTERS, opts))
        + "Answer:"
    )


def build_sets():
    rng = random.Random(2026)
    train = []
    test = []
    for s in SUMS:
        for a in range(1, 40):
            b = s - a
            if b < 1 or b > 40:
                continue
            pair = (a, b, s)
            if a <= 28 and b <= 28:
                train.append(pair)
            else:
                test.append(pair)
    rng.shuffle(train)
    rng.shuffle(test)
    return train[:400], test[:400]


def run(m, tok, prompts):
    """Batched forward, capture last-token pre-norm hidden at m.norm."""
    cap = {}
    hk = m.norm.register_forward_pre_hook(
        lambda mod, inp: cap.__setitem__("h", inp[0].detach().float())
    )
    ids = [tok.encode(p, add_special_tokens=False) for p in prompts]
    L = set(len(x) for x in ids)
    if len(L) != 1:
        raise RuntimeError(f"prompt lengths differ: {sorted(L)[:5]}")
    X = torch.tensor(ids)
    with torch.inference_mode():
        m(X)
    hk.remove()
    return cap["h"][:, -1, :]  # [batch, hidden]


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--n", type=int, default=200)
    a = ap.parse_args()
    a.ckpt = a.ckpt.resolve()
    tok = tok_for(151936)
    m = load_model(a.ckpt)
    enable_cache(m)
    train, test = build_sets()
    perm = [0, 1, 2, 3]  # fixed canonical option order
    print(
        f"[data] train={len(train)} test={len(test)} train_digits<=28 test_digits>=29",
        flush=True,
    )

    tr = train[: a.n]
    te = test[: a.n]
    trp = [prompt(x, y, perm) for x, y, _ in tr]
    tep = [prompt(x, y, perm) for x, y, _ in te]
    Xtr = run(m, tok, trp)
    Xte = run(m, tok, tep)
    Ytr = torch.tensor([SUMS.index(s) for _, _, s in tr])
    Yte = torch.tensor([SUMS.index(s) for _, _, s in te])
    d = Xtr.shape[1]
    print(f"[hidden] d={d} Xtr={tuple(Xtr.shape)} Xte={tuple(Xte.shape)}", flush=True)
    mu = Xtr.mean(0, keepdim=True)
    A = Xtr - mu
    B = Xte - mu
    G = A.T @ A
    b = A.T @ Ytr.float()

    def evalw(W, X, Y):
        pred = (X @ W).argmax(-1)
        return int((pred == Y).sum())

    best = None
    for lam in [1e-3, 1e-2, 1e-1, 1, 10, 100]:
        W = torch.linalg.solve(G + lam * torch.eye(d), b)
        t = evalw(W, A, Ytr)
        e = evalw(W, B, Yte)
        print(
            f"[ridge lam={lam:g}] train={t}/{len(Ytr)} test={e}/{len(Yte)} ({100 * e / len(Yte):.1f}%)",
            flush=True,
        )
        if best is None or e > best[1]:
            best = (lam, t, e)
    lam = best[0]
    W = torch.linalg.solve(G + lam * torch.eye(d), b)
    # wrong-target control
    Ysh = Ytr.clone()
    random.Random(7).shuffle(Ysh.tolist())
    Ysh = torch.tensor(Ysh.tolist())
    Ww = torch.linalg.solve(G + lam * torch.eye(d), A.T @ Ysh.float())
    print(
        f"[WRONG-TARGET] test={evalw(Ww, B, Yte)}/{len(Yte)} ({100 * evalw(Ww, B, Yte) / len(Yte):.1f}%)",
        flush=True,
    )
    # random-direction control
    Wr = torch.randn(d, 4) * 0.01
    print(
        f"[RANDOM-DIR] test={evalw(Wr, B, Yte)}/{len(Yte)} ({100 * evalw(Wr, B, Yte) / len(Yte):.1f}%)",
        flush=True,
    )
    # permutation control: read on permuted-option test prompts
    perm2 = [2, 0, 3, 1]
    tep2 = [prompt(x, y, perm2) for x, y, _ in te]
    B2 = run(m, tok, tep2) - mu
    # labels shift under permutation: canonical sum index -> new option position
    Yte2 = torch.tensor([perm2.index(SUMS.index(s)) for _, _, s in te])
    print(
        f"[PERM-CONTROL] test(perm read)={evalw(W, B2, Yte2)}/{len(Yte2)} ({100 * evalw(W, B2, Yte2) / len(Yte2):.1f}%)",
        flush=True,
    )
    acc = 100 * best[2] / len(Yte)
    print(
        f"[VERDICT] chance=25% best_heldout={acc:.1f}% | {'LATENT LINEAR STRUCTURE' if acc > 45 else 'NO LINEAR LATENT STRUCTURE'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
