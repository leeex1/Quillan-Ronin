#!/usr/bin/env python3
"""Fast positive control: centroid + ridge on base vs math40 (no slow raw loop)."""

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
from latent_fast import prompt, build_sets, run, SUMS


def diag(ckpt):
    tok = tok_for(151936)
    m = load_model(ckpt)
    enable_cache(m)
    train, test = build_sets()
    tr = train[:80]
    te = test[:60]
    trp = [prompt(a, b, [0, 1, 2, 3]) for a, b, _ in tr]
    tep = [prompt(a, b, [0, 1, 2, 3]) for a, b, _ in te]
    Xtr = run(m, tok, trp)
    Xte = run(m, tok, tep)
    Ytr = torch.tensor([SUMS.index(s) for _, _, s in tr])
    Yte = torch.tensor([SUMS.index(s) for _, _, s in te])
    C = torch.stack([Xtr[Ytr == k].mean(0) for k in range(4)])
    print(
        f"[{ckpt.name}] centroid train={int(((Xtr @ C.T).argmax(-1) == Ytr).sum())}/{len(Ytr)} test={int(((Xte @ C.T).argmax(-1) == Yte).sum())}/{len(Yte)}",
        flush=True,
    )
    A = Xtr - Xtr.mean(0)
    B = Xte - Xtr.mean(0)
    d = A.shape[1]
    for lam in [0.01, 1.0]:
        W = torch.linalg.solve(A.T @ A + lam * torch.eye(d), A.T @ Ytr.float())
        print(
            f"[{ckpt.name}] ridge lam={lam} train={int(((A @ W).argmax(-1) == Ytr).sum())} test={int(((B @ W).argmax(-1) == Yte).sum())}",
            flush=True,
        )


for name in [
    "quillan-mini-rank64.safetensors",
    "quillan-mini-math40-rank64.safetensors",
]:
    diag(Path(r"C:\02_QUILLAN\exports") / name)
