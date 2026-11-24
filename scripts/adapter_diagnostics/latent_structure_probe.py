#!/usr/bin/env python3
"""DECISIVE: is a+b linearly decodable from the BASE last-token hidden state?

If the base model already encodes arithmetic compositionally, a closed-form linear
read generalizes to held-out pairs (Qwen-like: latent ability exists).
If not, no linear LoRA write can generalize -> need nonlinear circuit (C8 PTP),
or the ability is simply absent (Gemma-like: honest dead-end).

Controls: digit-level holdout, wrong-target ridge, random-label ridge.
"""

from __future__ import annotations

# --- SM61 GPU AUTO-ROUTE (2026-11-23, reversible) ---
import os as _os, sys as _sys
from pathlib import Path as _Path
try:
    import torch as _torch
    if _os.environ.get("QUILLAN_SM61_DIRECT") != "1" and "+cu130" in _torch.__version__:
        _vpy = _Path(r"C:\02_QUILLAN\venv_sm61_test\Scripts\python.exe")
        _wrap = _Path(r"C:\02_QUILLAN\scripts\adapter_diagnostics\run_probe_gpu_sm61.py")
        if _vpy.exists() and _wrap.exists():
            _os.environ["QUILLAN_SM61_DIRECT"] = "1"
            _cmd = [str(_vpy), "-u", str(_wrap), str(_Path(__file__).resolve())] + _sys.argv[1:]
            _os.execv(str(_vpy), _cmd)
except Exception:
    pass
# ----------------------------------------------------
import sys, random
from pathlib import Path
import torch
import torch.nn.functional as F

ROOT = Path(r"C:\02_QUILLAN")
DIAG = ROOT / "scripts" / "adapter_diagnostics"
for p in (str(ROOT / "scripts"), str(ROOT / "03 - Training & Model"), str(DIAG)):
    if p not in sys.path:
        sys.path.insert(0, p)
from hard_math20 import LETTERS
from wsp_math_eval import tok_for, letter_ids, load_model


def fmt(a, b, opts):
    q = f"What is {a} + {b}?"
    # randomize option order deterministically by a+b seed so it isn't a positional cue
    o = opts[:]
    rng = random.Random(a * 100 + b)
    rng.shuffle(o)
    key = o.index(a + b)
    return f"Question: {q}\n" + "".join(
        f"{L}) {v}\n" for L, v in zip(LETTERS, o)
    ) + "Answer:", key


def capture(m, tok, prompt):
    cap = {}
    hk = m.norm.register_forward_pre_hook(
        lambda mod, inp: cap.__setitem__("h", inp[0].detach().float())
    )
    ids = tok.encode(prompt, add_special_tokens=False)
    with torch.inference_mode():
        m(torch.tensor([ids]))
    hk.remove()
    return cap["h"][0, -1]


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--ridge", type=float, default=1e-2)
    a = ap.parse_args()
    a.ckpt = a.ckpt.resolve()
    tok = tok_for(151936)
    lids = letter_ids(tok, 151936)
    m = load_model(a.ckpt)

    # target sums and their 4 options: buckets {20,30,40,50}
    sums = [20, 30, 40, 50]

    def opts_for(s):
        others = [v for v in sums if v != s]
        return [s] + others  # all 4 sums as options

    rng = random.Random(2026)
    # Build pairs for each sum, digits 1..40
    bysum = {s: [] for s in sums}
    seen = set()
    while any(len(bysum[s]) < a.n // 4 for s in sums):
        t = rng.choice(sums)
        aa = rng.randint(1, 40)
        bb = t - aa
        if bb < 1 or bb > 40:
            continue
        if (aa, bb) in seen:
            continue
        seen.add((aa, bb))
        bysum[t].append((aa, bb, t))

    # digit-level holdout: train digits 1..28, test uses digit >=29 (unseen in that range)
    train = [p for pl in bysum.values() for p in pl if p[0] <= 28 and p[1] <= 28]
    test = [p for pl in bysum.values() for p in pl if p[0] >= 29 or p[1] >= 29]
    rng.shuffle(train)
    rng.shuffle(test)
    train = train[: a.n]
    test = test[: a.n]
    print(
        f"[data] train={len(train)} test={len(test)} train-max-digit={max(max(p[0], p[1]) for p in train)} test-min-overlap-digit>=29",
        flush=True,
    )

    def build(rows):
        X = []
        Y = []
        for aa, bb, t in rows:
            opts = opts_for(t)
            prompt, key = fmt(aa, bb, opts)
            # bucket label = index of the TRUE sum in canonical sums list (not option position)
            Y.append(sums.index(t))
            X.append(capture(m, tok, prompt))
        return torch.stack(X), torch.tensor(Y)

    Xtr, Ytr = build(train)
    Xte, Yte = build(test)
    d = Xtr.shape[1]
    print(f"[hidden] d={d} |Xtr|={tuple(Xtr.shape)}", flush=True)

    def fit_ridge(X, Y, lam):
        return torch.linalg.solve(X.T @ X + lam * torch.eye(d), X.T @ Y)

    def acc(W, X, Y):
        pred = (X @ W).argmax(-1)
        return int((pred == Y).sum()), len(Y)

    # center using train mean
    mu = Xtr.mean(0, keepdim=True)
    Xtrc = Xtr - mu
    Xtec = Xte - mu
    best = None
    for lam in [1e-3, 1e-2, 1e-1, 1.0, 10.0]:
        W = fit_ridge(Xtrc, Ytr.float(), lam)
        tr = acc(W, Xtrc, Ytr)
        te = acc(W, Xtec, Yte)
        print(
            f"[ridge lam={lam:g}] train={tr[0]}/{tr[1]} test={te[0]}/{te[1]} ({100 * te[0] / te[1]:.1f}%)",
            flush=True,
        )
        if best is None or te[0] > best[1]:
            best = (lam, te[0], te[1])

    # WRONG-TARGET control: shuffle labels, refit at best lam -> should collapse to chance (~25%)
    lam = best[0]
    Ysh = Ytr.clone()
    random.Random(1).shuffle(Ysh.tolist())
    Ysh = torch.tensor(Ysh.tolist())
    Ww = fit_ridge(Xtrc, Ysh.float(), lam)
    wtr = acc(Ww, Xtrc, Ysh)
    wte = acc(Ww, Xtec, Yte)
    print(
        f"[WRONG-TARGET control] train={wtr[0]}/{wtr[1]} test={wte[0]}/{wte[1]} ({100 * wte[0] / wte[1]:.1f}%)",
        flush=True,
    )

    # RANDOM-DIRECTION control: random W, test
    Wr = torch.randn(d, 4) * 0.01
    rtr = acc(Wr, Xtrc, Ytr)
    rte = acc(Wr, Xtec, Yte)
    print(
        f"[RANDOM-DIR control] test={rte[0]}/{rte[1]} ({100 * rte[0] / rte[1]:.1f}%)",
        flush=True,
    )

    print(
        f"[VERDICT] chance=25% | best held-out={100 * best[1] / best[2]:.1f}% | {'LATENT STRUCTURE (linear read generalizes)' if best[1] / best[2] > 0.45 else 'NO LINEAR LATENT STRUCTURE (need nonlinear circuit / ability absent)'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
