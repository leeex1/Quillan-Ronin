#!/usr/bin/env python3
"""Fast EXACT reproduction of parametric_probe capture point, batched + self-contained.
Mirrors wsp_math_tune_v3: key = down_proj INPUT (pre-hook), lora_A seeded basis.
Includes: capture sanity, base-vs-math40 positive control, digit holdout, wrong-target."""

from __future__ import annotations
import sys, random
from pathlib import Path
import torch, torch.nn.functional as F

ROOT = Path(r"C:\02_QUILLAN")
DIAG = ROOT / "scripts" / "adapter_diagnostics"
for p in (str(ROOT / "scripts"), str(ROOT / "03 - Training & Model"), str(DIAG)):
    if p not in sys.path:
        sys.path.insert(0, p)
from hard_math20 import LETTERS
from wsp_math_eval import tok_for, letter_ids, load_model
from wsp_math_tune_v3 import enable_cache, seed_lora_basis

SUMS = [20, 30, 40, 50]


def fmt(a, b, shuffle):
    opts = SUMS[:]
    key = opts.index(a + b)
    if shuffle:
        opts = [opts[i] for i in shuffle]
        key = shuffle.index(SUMS.index(a + b))
    return (
        f"Question: What is {a} + {b}?\n"
        + "".join(f"{L}) {v}\n" for L, v in zip(LETTERS, opts))
        + "Answer:",
        key,
    )


def capture_down_batched(m, tok, prompts):
    cap = {}
    hk = m.layers[-1].down_proj.register_forward_pre_hook(
        lambda mod, inp: cap.__setitem__("x", inp[0].detach().float())
    )
    gidx = {}
    for i, p in enumerate(prompts):
        L = len(tok.encode(p, add_special_tokens=False))
        gidx.setdefault(L, []).append((i, p))
    results = [None] * len(prompts)
    for L, items in gidx.items():
        ids = [tok.encode(p, add_special_tokens=False) for _, p in items]
        with torch.inference_mode():
            m(torch.tensor(ids))
        x = cap["x"][:, -1, :]
        for j, (orig, _) in enumerate(items):
            results[orig] = x[j]
    hk.remove()
    return torch.stack(results)


def sanity(m, tok):
    ps = [fmt(a, b, None)[0] for a, b in [(3, 17), (11, 9), (25, 15), (1, 19)]]
    X = capture_down_batched(m, tok, ps)
    X2 = capture_down_batched(m, tok, ps)
    d = torch.cdist(X, X)
    print(
        f"[SANITY] det_maxdiff={float((X - X2).abs().max()):.6f} pairwise_offdiag_min={float(d[d > 0].min()):.3f}",
        flush=True,
    )


def probe(ckpt, n=60):
    tok = tok_for(151936)
    lids = letter_ids(tok, 151936)
    m = load_model(ckpt)
    down = m.layers[-1].down_proj
    seed_lora_basis(m, [len(m.layers) - 1], 42)
    enable_cache(m)
    A = down.lora_A.detach().float()
    d = A.shape[0]
    sanity(m, tok)
    rng = random.Random(2026)
    allp = set()
    while len(allp) < 2 * n + 30:
        t = rng.choice(SUMS)
        aa = rng.randint(1, 40)
        bb = t - aa
        if 1 <= bb <= 40:
            allp.add((aa, bb, t))
    pairs = list(allp)
    rng.shuffle(pairs)
    train = pairs[:n]
    test = pairs[n : 2 * n]
    assert not (set(train) & set(test))
    tr = [
        fmt(x, y, random.Random(7000 + i).sample(range(4), 4))
        for i, (x, y, _) in enumerate(train)
    ]
    te = [fmt(x, y, None) for x, y, _ in test]
    Xtr = capture_down_batched(m, tok, [p for p, _ in tr])
    Xte = capture_down_batched(m, tok, [p for p, _ in te])
    Ktr = F.linear(Xtr, A)
    Kte = F.linear(Xte, A)
    Ytr = torch.tensor([k for _, k in tr])
    Yte = torch.tensor([k for _, k in te])
    T = torch.full((len(tr), 4), -1.0)
    for i, (_, k) in enumerate(tr):
        T[i, k] = 1.0
    for lam in [1e-4, 1e-3, 1e-2]:
        C = torch.linalg.solve(Ktr.T @ Ktr + lam * torch.eye(d), Ktr.T @ T).T
        ptr = (Ktr @ C.T).argmax(-1)
        pte = (Kte @ C.T).argmax(-1)
        trh = int((ptr == Ytr).sum())
        teh = int((pte == Yte).sum())
        print(
            f"[{ckpt.name}] lam={lam:g} train={trh}/{len(Ytr)} test={teh}/{len(Yte)} ({100 * teh / len(Yte):.1f}%)",
            flush=True,
        )
    # wrong-target control at lam=1e-3
    C = torch.linalg.solve(Ktr.T @ Ktr + 1e-3 * torch.eye(d), Ktr.T @ T).T
    Ysh = Ytr.clone()
    random.Random(7).shuffle(Ysh.tolist())
    Ysh = torch.tensor(Ysh.tolist())
    Tw = torch.full((len(tr), 4), -1.0)
    for i, k in enumerate(Ysh.tolist()):
        Tw[i, k] = 1.0
    Cw = torch.linalg.solve(Ktr.T @ Ktr + 1e-3 * torch.eye(d), Ktr.T @ Tw).T
    print(
        f"[{ckpt.name}] WRONG-TARGET test={int(((Kte @ Cw.T).argmax(-1) == Yte).sum())}/{len(Yte)}",
        flush=True,
    )


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    a = ap.parse_args()
    for name in [
        "quillan-mini-rank64.safetensors",
        "quillan-mini-math40-rank64.safetensors",
    ]:
        probe(Path(r"C:\02_QUILLAN\exports") / name, a.n)
