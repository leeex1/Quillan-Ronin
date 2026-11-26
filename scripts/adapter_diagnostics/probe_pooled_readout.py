#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""POOLED READOUT TEST — READ-ONLY, FROZEN CACHES.

final_norm last-token cos=1.0 (echo) but all-position mean=0.727 (signal).
Test: pool final_norm hidden across positions (mean), center by cross-prompt
mean mu, then apply the correct-fold head. If the two prose prompts become
distinguishable (A) and donor top-k stays >0 (C), the fix is a value-only
pooling operation change — NO source bypass, NO new params.

Compares pooling modes: last, mean-all, mean-no-last. LOO split.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List

ROOT = Path(r"C:\02_QUILLAN")
DIAG = ROOT / "scripts" / "adapter_diagnostics"
for p in (str(ROOT / "scripts"), str(ROOT / "03 - Training & Model"), str(DIAG)):
    if p not in sys.path:
        sys.path.insert(0, p)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

STAGE = "final_norm"
LAMS = [1e-2, 1e-1]
PROSE_A, PROSE_B = "The capital of France is", "Hello, my name is"
MODES = ["last", "mean_all", "mean_no_last"]


def pool(h: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "last":
        return h[-1:]
    if mode == "mean_all":
        return h.mean(dim=0, keepdim=True)
    if mode == "mean_no_last":
        return h[:-1].mean(dim=0, keepdim=True) if h.shape[0] > 1 else h[-1:]
    raise ValueError(mode)


def main() -> int:
    donor = torch.load(
        str(DIAG / "_acts" / "donor_acts.pt"), map_location="cpu", weights_only=False
    )
    custom = torch.load(
        str(DIAG / "_acts" / "custom_rescaled_acts.pt"),
        map_location="cpu",
        weights_only=False,
    )
    Wu = torch.load(
        str(ROOT / "exports" / "donor_cache_v23" / "qwen_0.5b_donors.pt"),
        map_location="cpu",
        weights_only=False,
    )["embed_tokens"].float()
    prompts = list(donor.keys())

    def H(p):
        return custom[p][STAGE][0].float()

    def donor_H(p):
        return donor[p]["final_hidden"][0].float()

    def D(p):
        return donor[p]["logits"][0].float()

    print("=" * 78)
    print("POOLED READOUT TEST (stage=final_norm, LOO, frozen caches)")
    print("=" * 78)

    for mode in MODES:
        for lam in LAMS:
            # fit R on train-prompt POOLED hidden -> donor pooled hidden
            Xs, Ys = [], []
            for p in prompts:
                hp = pool(H(p), mode)  # [1,896] or [T',896]
                yp = pool(donor_H(p), mode)
                T = min(hp.shape[0], yp.shape[0])
                Xs.append(hp[:T])
                Ys.append(yp[:T])
            X = torch.cat(Xs, 0)
            Y = torch.cat(Ys, 0)
            A = X.T @ X + lam * torch.eye(X.shape[1], dtype=X.dtype)
            R = torch.linalg.solve(A, X.T @ Y)
            W = Wu @ R.T
            mu = X.mean(dim=0, keepdim=True)

            # prose pair with mu from all prompts (approx; fine for mechanism test)
            la = ((pool(H(PROSE_A), mode) - mu) @ W.T)[0]
            lb = ((pool(H(PROSE_B), mode) - mu) @ W.T)[0]
            cos = float(F.cosine_similarity(la.unsqueeze(0), lb.unsqueeze(0)).item())
            t1a, t1b = int(la.argmax()), int(lb.argmax())
            a_pass = cos < 0.99 and t1a != t1b

            # gate C: donor top-k on all prompts (pooled custom vs donor last logit)
            c1 = c5 = 0
            for p in prompts:
                cl = ((pool(H(p), mode) - mu) @ W.T)[0]
                dl = D(p)[-1]
                c1 += int(
                    bool(
                        set(torch.topk(cl, 1).indices.tolist())
                        & set(torch.topk(dl, 1).indices.tolist())
                    )
                )
                c5 += len(
                    set(torch.topk(cl, 5).indices.tolist())
                    & set(torch.topk(dl, 5).indices.tolist())
                )
            m1, m5 = c1 / len(prompts), c5 / len(prompts)
            c_pass = m1 > 0 and m5 > 0
            print(
                f"  {mode:12} lam={lam:<6} A: cos={cos:.4f} t1={t1a}!={t1b} {'PASS' if a_pass else 'FAIL'} "
                f"| C: t1={m1:.3f} t5={m5:.3f} {'PASS' if c_pass else 'FAIL'}"
            )
    print("=" * 78)
    print("No files written.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
