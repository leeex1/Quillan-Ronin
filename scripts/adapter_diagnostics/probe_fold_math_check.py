#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""VERIFY FOLD MATH — READ-ONLY. Compare buggy vs correct ridge-fold logits.

Ridge R maps custom hidden -> donor hidden: h_d_hat = h_c @ R.
Donor logits: logits = h_d @ Wu.T   (Wu = embed_tokens [151936,896]).

CORRECT:  logits = (h_c @ R) @ Wu.T
BUGGY in probe_construct_*: W_eff = Wu @ R; logits = (h_c @ R) @ W_eff.T
                          = h_c @ R @ R.T @ Wu.T   <-- extra R.T factor!

Fold into txt_dec.weight: need logits = h_c @ W_eff.T, so
W_eff.T must = R @ Wu.T  =>  W_eff = Wu @ R.T  (NOT Wu @ R).

Compares both on leave-one-prompt-out over frozen caches. No files written.
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

STAGES = ["decomposition", "ingestion", "moe"]
LAMS = [1e-3, 1e-2, 1e-1]


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
    print("=" * 78)
    print("FOLD MATH CHECK: buggy (Wu@R, double-R) vs correct (Wu@R.T)")
    print("=" * 78)

    def X(p, stage):
        return custom[p][stage][0].float()

    def Y(p):
        return donor[p]["final_hidden"][0].float()

    def D(p):
        return donor[p]["logits"][0].float()

    def solve(Xs, Ys, lam):
        Xc = torch.cat(Xs, 0)
        Yc = torch.cat(Ys, 0)
        A = Xc.T @ Xc + lam * torch.eye(Xc.shape[1], dtype=Xc.dtype)
        return torch.linalg.solve(A, Xc.T @ Yc)

    for stage in STAGES:
        for lam in LAMS:
            buggy_cos, correct_cos = [], []
            buggy_t1 = correct_t1 = 0
            buggy_t5 = correct_t5 = 0
            for held in prompts:
                train = [p for p in prompts if p != held]
                R = solve([X(p, stage) for p in train], [Y(p) for p in train], lam)
                W_buggy = Wu @ R  # what the probe used
                W_correct = Wu @ R.T  # correct fold
                x = X(held, stage)
                dl = D(held)
                T = min(x.shape[0], dl.shape[0])
                for t in range(T):
                    hc = x[t : t + 1]
                    cl_bug = (hc @ R @ W_buggy.T)[0]  # double-R (probe's formula)
                    cl_cor = (hc @ W_correct.T)[0]  # correct: hc @ R @ Wu.T
                    d = dl[t]
                    buggy_cos.append(
                        float(
                            F.cosine_similarity(
                                cl_bug.unsqueeze(0), d.unsqueeze(0)
                            ).item()
                        )
                    )
                    correct_cos.append(
                        float(
                            F.cosine_similarity(
                                cl_cor.unsqueeze(0), d.unsqueeze(0)
                            ).item()
                        )
                    )
                    buggy_t1 += int(
                        bool(
                            set(torch.topk(cl_bug, 1).indices.tolist())
                            & set(torch.topk(d, 1).indices.tolist())
                        )
                    )
                    correct_t1 += int(
                        bool(
                            set(torch.topk(cl_cor, 1).indices.tolist())
                            & set(torch.topk(d, 1).indices.tolist())
                        )
                    )
                    buggy_t5 += len(
                        set(torch.topk(cl_bug, 5).indices.tolist())
                        & set(torch.topk(d, 5).indices.tolist())
                    )
                    correct_t5 += len(
                        set(torch.topk(cl_cor, 5).indices.tolist())
                        & set(torch.topk(d, 5).indices.tolist())
                    )
            n = len(buggy_cos)
            print(
                f"  {stage:<14} lam={lam:<6} BUGGY cos={sum(buggy_cos) / n:+.4f} t1={buggy_t1}/{n} t5={buggy_t5}/{n} "
                f"| CORRECT cos={sum(correct_cos) / n:+.4f} t1={correct_t1}/{n} t5={correct_t5}/{n}"
            )
    print("=" * 78)
    print("No files written.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
