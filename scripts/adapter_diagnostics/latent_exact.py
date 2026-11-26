#!/usr/bin/env python3
"""Latent structure probe with the EXACT trained prompt format (non-padded),
length-grouped batching. Positive control = math40 (known 40/40 keyed write)."""

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


def prompt(a, b, perm=(0, 1, 2, 3)):
    opts = [SUMS[i] for i in perm]
    return (
        f"Question: What is {a} + {b}?\n"
        + "".join(f"{L}) {v}\n" for L, v in zip(LETTERS, opts))
        + "Answer:"
    )


def capture_batched(m, tok, prompts):
    cap = {}
    hk = m.norm.register_forward_pre_hook(
        lambda mod, inp: cap.__setitem__("h", inp[0].detach().float())
    )
    out = []
    # group by token length so no padding needed
    groups = {}
    for p in prompts:
        ids = tok.encode(p, add_special_tokens=False)
        groups.setdefault(len(ids), []).append(ids)
    # need to preserve original order
    idx_by_len = [[] for _ in prompts]
    for i, p in enumerate(prompts):
        idx_by_len[i] = len(tok.encode(p, add_special_tokens=False))
    results = [None] * len(prompts)
    # rebuild with indices
    gidx = {}
    for i, p in enumerate(prompts):
        L = len(tok.encode(p, add_special_tokens=False))
        gidx.setdefault(L, []).append((i, p))
    for L, items in gidx.items():
        ids = [tok.encode(p, add_special_tokens=False) for _, p in items]
        with torch.inference_mode():
            m(torch.tensor(ids))
        h = cap["h"][:, -1, :]
        for j, (orig_i, _) in enumerate(items):
            results[orig_i] = h[j]
    hk.remove()
    return torch.stack(results)


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
    rng = random.Random(2026)
    train = []
    test = []
    for s in SUMS:
        for aa in range(1, 40):
            bb = s - aa
            if bb < 1 or bb > 40:
                continue
            (train if aa <= 28 and bb <= 28 else test).append((aa, bb, s))
    rng.shuffle(train)
    rng.shuffle(test)
    tr = train[: a.n]
    te = test[: min(a.n, len(test))]
    print(f"[{a.ckpt.name}] data train={len(tr)} test={len(te)}", flush=True)
    Xtr = capture_batched(m, tok, [prompt(x, y) for x, y, _ in tr])
    Xte = capture_batched(m, tok, [prompt(x, y) for x, y, _ in te])
    Ytr = torch.tensor([SUMS.index(s) for _, _, s in tr])
    Yte = torch.tensor([SUMS.index(s) for _, _, s in te])
    d = Xtr.shape[1]
    A = Xtr - Xtr.mean(0)
    B = Xte - Xtr.mean(0)
    for lam in [0.01, 1.0, 10.0]:
        W = torch.linalg.solve(A.T @ A + lam * torch.eye(d), A.T @ Ytr.float())
        print(
            f"[{a.ckpt.name}] ridge lam={lam} train={int(((A @ W).argmax(-1) == Ytr).sum())}/{len(Ytr)} test={int(((B @ W).argmax(-1) == Yte).sum())}/{len(Yte)}",
            flush=True,
        )


for name in [
    "quillan-mini-rank64.safetensors",
    "quillan-mini-math40-rank64.safetensors",
]:
    main_clck = Path(r"C:\02_QUILLAN\exports") / name
    import sys as _s

    _s.argv = ["x", "--ckpt", str(main_clck), "--n", "200"]
    main()
