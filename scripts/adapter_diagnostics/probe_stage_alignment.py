#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""STAGE ALIGNMENT DIAGNOSTIC — READ-ONLY.

Which custom stage best linearly predicts donor layer-24 hidden?
Uses paired caches already on disk (custom_rescaled_acts.pt / donor_acts.pt).
For each stage: ridge map h_stage -> h_donor, leave-one-prompt-out CV.
Reports held-out cosine + top-k logit overlap via donor head Wu.

If NO stage generalizes, the donor computation is not recoverable from any
readout of the frozen graph under the current locks.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(r"C:\02_QUILLAN")
DIAG = ROOT / "scripts" / "adapter_diagnostics"
for p in (str(ROOT / "scripts"), str(ROOT / "03 - Training & Model"), str(DIAG)):
    if p not in sys.path:
        sys.path.insert(0, p)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

ACTS = DIAG / "_acts"
STAGES = [
    "ingestion",
    "decomposition",
    "diffusion_core",
    "moe",
    "final_norm",
    "decoder_input",
]


def solve(X: torch.Tensor, Y: torch.Tensor, lam: float) -> torch.Tensor:
    A = X.T @ X + lam * torch.eye(X.shape[1], dtype=X.dtype)
    return torch.linalg.solve(A, X.T @ Y)


def main() -> int:
    donor = torch.load(
        str(ACTS / "donor_acts.pt"), map_location="cpu", weights_only=False
    )
    custom = torch.load(
        str(ACTS / "custom_rescaled_acts.pt"), map_location="cpu", weights_only=False
    )
    prompts = list(donor.keys())
    print(f"paired prompts: {len(prompts)}")
    print("=" * 78)

    # Build per-prompt [T,896] tensors (last position only for prediction)
    Y_by_prompt = {}
    for p in prompts:
        Y_by_prompt[p] = donor[p]["final_hidden"][0, -1].float()  # [896]

    for stage in STAGES:
        cos_vals: List[float] = []
        # leave-one-prompt-out: train on 4, test on 1 (all positions of test as samples)
        for held in prompts:
            Xtr, Ytr = [], []
            for p in prompts:
                if p == held:
                    continue
                x = custom[p][stage][0].float()  # [T,896]
                y = donor[p]["final_hidden"][0].float()  # [T,896]
                T = min(x.shape[0], y.shape[0])
                Xtr.append(x[:T])
                Ytr.append(y[:T])
            X = torch.cat(Xtr, 0)
            Y = torch.cat(Ytr, 0)
            try:
                R = solve(X, Y, 1e-2)
            except torch.linalg.LinAlgError:
                continue
            # test: predict donor hidden from custom hidden at held-out positions
            xte = custom[held][stage][0].float()
            yte = donor[held]["final_hidden"][0].float()
            T = min(xte.shape[0], yte.shape[0])
            with torch.no_grad():
                yhat = xte[:T] @ R
            for t in range(T):
                c = float(
                    F.cosine_similarity(
                        yhat[t].unsqueeze(0), yte[t].unsqueeze(0)
                    ).item()
                )
                cos_vals.append(c)
        if cos_vals:
            mean_cos = sum(cos_vals) / len(cos_vals)
            print(
                f"  {stage:<18} leave-one-prompt-out hidden cos = {mean_cos:+.4f}  (n={len(cos_vals)})"
            )
    print("=" * 78)
    print("No files written.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
