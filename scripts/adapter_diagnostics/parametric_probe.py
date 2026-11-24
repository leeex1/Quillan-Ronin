#!/usr/bin/env python3
"""Proper parametric generalization test: disjoint train/test numeral pairs."""

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
from wsp_math_tune_v3 import enable_cache, seed_lora_basis


def gen_pair(rng, lo=1, hi=40):
    target = rng.choice([20, 30, 40, 50])  # FIX: only 4 options (A-D), key index < 4
    a = rng.randint(max(lo, target - hi), min(hi, target - 1))
    b = target - a
    return (a, b, target)


def fmt(pair, shuffle=None):
    a, b, t = pair
    opts = [20, 30, 40, 50]  # FIX: 4 options only, matches A-D scorer
    key = opts.index(t)
    if shuffle:
        order = shuffle
        opts = [opts[i] for i in order]
        key = order.index(key)
    q = f"What is {a} + {b}?"
    return f"Question: {q}\n" + "".join(
        f"{L}) {v}\n" for L, v in zip(LETTERS, opts)
    ) + "Answer:", key


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--train-n", type=int, default=80)
    ap.add_argument("--test-n", type=int, default=80)
    ap.add_argument("--margin", type=float, default=1.0)
    ap.add_argument("--lam", type=float, default=1e-4)
    a = ap.parse_args()
    a.ckpt = a.ckpt.resolve()
    tok = tok_for(151936)
    lids = letter_ids(tok, 151936)

    rng = random.Random(2026)
    allp = set()
    while len(allp) < a.train_n + a.test_n + 50:
        allp.add(gen_pair(rng))
    pairs = list(allp)
    rng.shuffle(pairs)
    train = pairs[: a.train_n]
    test = pairs[a.train_n : a.train_n + a.test_n]
    assert not (set(train) & set(test))
    print(f"[data] train={len(train)} test={len(test)} disjoint=True", flush=True)

    # add option-order shuffling to train so the write keys on computation not position
    tr_rows = []
    for i, p in enumerate(train):
        order = random.Random(7000 + i).sample(range(5), 5)  # 5 options now
        # use 4 options subset containing the target for A/B/C/D
        four = sorted(
            set(
                [p[2]]
                + random.Random(8000 + i).sample(
                    [v for v in [20, 30, 40, 50, 60] if v != p[2]], 3
                )
            )
        )
        rng2 = random.Random(9000 + i)
        rng2.shuffle(four)
        opts = four
        key = four.index(p[2])
        q = f"What is {p[0]} + {p[1]}?"
        tr_rows.append(
            (
                f"Question: {q}\n"
                + "".join(f"{L}) {v}\n" for L, v in zip(LETTERS, opts))
                + "Answer:",
                key,
            )
        )
    # test with FIXED order for clean held-out
    te_rows = [fmt(p)[0:1] + (fmt(p)[1],) for p in test]
    te_rows = [(fmt(p)[0], fmt(p)[1]) for p in test]

    m = load_model(a.ckpt)
    down = m.layers[-1].down_proj
    seed_lora_basis(m, [len(m.layers) - 1], 42)
    enable_cache(m)
    A = down.lora_A.detach().float()
    d = A.shape[0]
    E = m.embed_tokens.unpack_float().float()[lids]
    Epinv = E.T @ torch.linalg.inv(E @ E.T + 1e-6 * torch.eye(4))

    def score(rows):
        ok = 0
        with torch.inference_mode():
            for r in rows:
                ids = tok.encode(r[0], add_special_tokens=False)
                lg = m(torch.tensor([ids]))[0][0, -1, lids]
                ok += int(int(lg.argmax()) == r[1])
        return ok

    bt = score(tr_rows)
    be = score(te_rows)
    print(f"[base] train={bt}/{len(tr_rows)} test={be}/{len(te_rows)}", flush=True)

    # capture keys over shuffled train
    Xs = []
    for r in tr_rows:
        ids = tok.encode(r[0], add_special_tokens=False)
        cap = {}
        hd = down.register_forward_pre_hook(
            lambda mod, inp: cap.__setitem__("x", inp[0].detach().float())
        )
        with torch.inference_mode():
            m(torch.tensor([ids]))
        hd.remove()
        Xs.append(cap["x"][0, -1])
    K = torch.stack([F.linear(Xs[i], A) for i in range(len(tr_rows))])
    N = len(tr_rows)
    T = torch.full((N, 4), -a.margin)
    for i, r in enumerate(tr_rows):
        T[i, r[1]] = a.margin
    C = torch.linalg.solve(K.T @ K + a.lam * torch.eye(d), K.T @ T).T
    dB = Epinv @ C
    with torch.no_grad():
        down.lora_B.copy_(dB)

    tr = score(tr_rows)
    te = score(te_rows)
    print(f"[WSP] train={tr}/{len(tr_rows)} test={te}/{len(te_rows)}", flush=True)
    print(
        f"[VERDICT] chance=25% | held-out {'GENERALIZES' if te / len(te_rows) > 0.4 else 'MEMORIZES'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
