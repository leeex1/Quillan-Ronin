#!/usr/bin/env python3
"""Falsify the probe harness: compare base vs math40 checkpoint on same latent read.
Also check hidden-state separation (centroid classifier) and raw model argmax."""

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
from wsp_math_eval import tok_for, letter_ids, load_model
from wsp_math_tune_v3 import enable_cache
from latent_fast import prompt, build_sets, run, SUMS


def diag(ckpt):
    tok = tok_for(151936)
    lids = letter_ids(tok, 151936)
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
    # centroid classifier (memorization sanity)
    C = torch.stack([Xtr[Ytr == k].mean(0) for k in range(4)])
    cent_tr = (Xtr @ C.T).argmax(-1)
    cent_te = (Xte @ C.T).argmax(-1)
    # ridge
    A = Xtr - Xtr.mean(0)
    B = Xte - Xtr.mean(0)
    d = A.shape[1]
    W = torch.linalg.solve(A.T @ A + 0.01 * torch.eye(d), A.T @ Ytr.float())
    rid_tr = (A @ W).argmax(-1)
    rid_te = (B @ W).argmax(-1)
    # raw model argmax over letter logits
    raw_tr = raw_te = 0
    for rows in [(trp, Ytr), (tep, Yte)]:
        ok = 0
        ids = [tok.encode(p, add_special_tokens=False) for p in rows[0]]
        with torch.inference_mode():
            for ii, y in zip(ids, rows[1]):
                lg = m(torch.tensor([ii]))[0][0, -1, lids]
                ok += int(int(lg.argmax()) == int(y))
        if rows[0] is trp:
            raw_tr = ok
        else:
            raw_te = ok
    print(
        f"[{ckpt.name}] centroid train={int((cent_tr == Ytr).sum())}/{len(Ytr)} test={int((cent_te == Yte).sum())}/{len(Yte)} | ridge train={int((rid_tr == Ytr).sum())} test={int((rid_te == Yte).sum())} | rawmodel train={raw_tr} test={raw_te}",
        flush=True,
    )


for name in [
    "quillan-mini-rank64.safetensors",
    "quillan-mini-math40-rank64.safetensors",
]:
    diag(Path(r"C:\02_QUILLAN\exports") / name)
