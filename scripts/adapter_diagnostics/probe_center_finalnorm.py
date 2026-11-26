#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CENTERING TEST FOR PROMPT DISCRIMINATION — READ-ONLY, FROZEN CACHES.

Gate A failed: decomposition tap, cos 0.995, both prose prompts top1=' a'.
Hypothesis: subtract the CROSS-PROMPT MEAN mu from decomposition hidden before
the fold, removing the shared base-rate direction so the residual is prompt-specific.

Uses frozen caches (fast, no model load). Leave-one-prompt-out fitting:
  fit R and mu on 4 prompts, test centered logits = (h - mu) @ (Wu @ R.T).T
Compares centered vs uncentered on gate A (prose pair) and gate C (donor overlap).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List

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
LAMS = [1e-3, 1e-2, 1e-1, 1.0]
PROSE_A, PROSE_B = "The capital of France is", "Hello, my name is"


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

    def X(p):
        return custom[p][STAGE][0].float()

    def Y(p):
        return donor[p]["final_hidden"][0].float()

    def D(p):
        return donor[p]["logits"][0].float()

    print("=" * 78)
    print("CENTERING DISCRIMINATION TEST (frozen caches, LOO, stage=final_norm)")
    print("=" * 78)

    for lam in LAMS:
        print(f"\n### lam={lam} ###")
        acc = {"raw": [], "centered": []}
        prose_raw = {"cos": [], "differ": []}
        prose_cen = {"cos": [], "differ": []}
        c1_raw = c5_raw = c1_cen = c5_cen = 0

        for held in prompts:
            train = [p for p in prompts if p != held]
            Xtr = torch.cat([X(p) for p in train], 0)
            Ytr = torch.cat([Y(p) for p in train], 0)
            A = Xtr.T @ Xtr + lam * torch.eye(Xtr.shape[1], dtype=Xtr.dtype)
            R = torch.linalg.solve(A, Xtr.T @ Ytr)
            W = Wu @ R.T
            mu = Xtr.mean(dim=0, keepdim=True)  # cross-prompt mean (training only)

            x = X(held)
            dl = D(held)
            t = x.shape[0] - 1
            with torch.no_grad():
                lg_raw = (x[t : t + 1] @ W.T)[0]
                lg_cen = ((x[t : t + 1] - mu) @ W.T)[0]
            d = dl[t]
            o1r = int(
                bool(
                    set(torch.topk(lg_raw, 1).indices.tolist())
                    & set(torch.topk(d, 1).indices.tolist())
                )
            )
            o5r = len(
                set(torch.topk(lg_raw, 5).indices.tolist())
                & set(torch.topk(d, 5).indices.tolist())
            )
            o1c = int(
                bool(
                    set(torch.topk(lg_cen, 1).indices.tolist())
                    & set(torch.topk(d, 1).indices.tolist())
                )
            )
            o5c = len(
                set(torch.topk(lg_cen, 5).indices.tolist())
                & set(torch.topk(d, 5).indices.tolist())
            )
            c1_raw += o1r
            c5_raw += o5r
            c1_cen += o1c
            c5_cen += o5c

        # prose pair: need both prompts fitted with the SAME R/mu (fit on other 3)
        others = [p for p in prompts if p not in (PROSE_A, PROSE_B)]
        Xo = torch.cat([X(p) for p in others], 0)
        Yo = torch.cat([Y(p) for p in others], 0)
        A = Xo.T @ Xo + lam * torch.eye(Xo.shape[1], dtype=Xo.dtype)
        R = torch.linalg.solve(A, Xo.T @ Yo)
        W = Wu @ R.T
        mu = Xo.mean(dim=0, keepdim=True)
        la_raw = (X(PROSE_A)[-1:] @ W.T)[0]
        lb_raw = (X(PROSE_B)[-1:] @ W.T)[0]
        la_cen = ((X(PROSE_A)[-1:] - mu) @ W.T)[0]
        lb_cen = ((X(PROSE_B)[-1:] - mu) @ W.T)[0]
        cos_raw = float(
            F.cosine_similarity(la_raw.unsqueeze(0), lb_raw.unsqueeze(0)).item()
        )
        cos_cen = float(
            F.cosine_similarity(la_cen.unsqueeze(0), lb_cen.unsqueeze(0)).item()
        )
        t1_ar, t1_br = int(la_raw.argmax()), int(lb_raw.argmax())
        t1_ac, t1_bc = int(la_cen.argmax()), int(lb_cen.argmax())

        print(
            f"  PROSE RAW     cos={cos_raw:.4f} top1_A={t1_ar} top1_B={t1_br} differ={t1_ar != t1_br} "
            f"-> A {'PASS' if cos_raw < 0.99 and t1_ar != t1_br else 'FAIL'}"
        )
        print(
            f"  PROSE CENTERED cos={cos_cen:.4f} top1_A={t1_ac} top1_B={t1_bc} differ={t1_ac != t1_bc} "
            f"-> A {'PASS' if cos_cen < 0.99 and t1_ac != t1_bc else 'FAIL'}"
        )
        print(
            f"  GATE C raw:     mean t1={c1_raw / len(prompts):.4f} t5={c5_raw / len(prompts):.4f}"
        )
        print(
            f"  GATE C centered: mean t1={c1_cen / len(prompts):.4f} t5={c5_cen / len(prompts):.4f} "
            f"-> {'PASS' if c1_cen > 0 and c5_cen > 0 else 'FAIL'}"
        )

    print("\n" + "=" * 78)
    print("No files written.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
