#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PERMUTATION CONTROL (C7 R1 / C25 R1) — READ-ONLY.

Is the +0.56/+0.34 decomposition->donor alignment REAL transfer, or just
shared input-embedding base rate? Fit the ridge map on TRAIN prompts under:
  (a) TRUE pairing   (custom[p] -> donor[p])
  (b) DERANGED pairing (custom[p] -> donor[perm[p]], no fixed points)
Score both on HELD-OUT prompts with TRUE pairing. If (a) does not beat (b),
the alignment is a base-rate artifact and T_R collapses.

Only frozen caches. No files written.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

ROOT = Path(r"C:\02_QUILLAN")
DIAG = ROOT / "scripts" / "adapter_diagnostics"
for p in (str(ROOT / "scripts"), str(ROOT / "03 - Training & Model"), str(DIAG)):
    if p not in sys.path:
        sys.path.insert(0, p)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

STAGE = "decomposition"
LAM = 1e-2


def main() -> int:
    donor = torch.load(
        str(DIAG / "_acts" / "donor_acts.pt"), map_location="cpu", weights_only=False
    )
    custom = torch.load(
        str(DIAG / "_acts" / "custom_rescaled_acts.pt"),
        map_location="cpu",
        weights_only=False,
    )
    prompts = list(donor.keys())
    assert len(prompts) >= 4
    print("=" * 74)
    print(f"PERMUTATION CONTROL stage={STAGE} lam={LAM} (read-only)")
    print("=" * 74)
    print("prompts:")
    for i, p in enumerate(prompts):
        print(f"  {i}: {p!r}")

    def X(p):
        return custom[p][STAGE][0].float()

    def Y(p):
        return donor[p]["final_hidden"][0].float()

    def solve(Xs, Ys):
        Xc = torch.cat(Xs, 0)
        Yc = torch.cat(Ys, 0)
        A = Xc.T @ Xc + LAM * torch.eye(Xc.shape[1], dtype=Xc.dtype)
        return torch.linalg.solve(A, Xc.T @ Yc)

    def score_on(R, test_prompts):
        vals = []
        for p in test_prompts:
            x, y = X(p), Y(p)
            T = min(x.shape[0], y.shape[0])
            with torch.no_grad():
                yh = x[:T] @ R
            vals += [
                float(F.cosine_similarity(yh[t].unsqueeze(0), y[t].unsqueeze(0)).item())
                for t in range(T)
            ]
        return sum(vals) / len(vals), min(vals), max(vals)

    # Leave-one-prompt-out CV for both true and deranged pairing
    rng = random.Random(20261121)
    true_scores, perm_scores = [], []
    for held in prompts:
        train = [p for p in prompts if p != held]
        # TRUE pairing
        R_true = solve([X(p) for p in train], [Y(p) for p in train])
        # DERANGED pairing: derange the target list (no fixed point)
        idx = list(range(len(train)))
        while True:
            rng.shuffle(idx)
            if all(idx[j] != j for j in range(len(idx))):
                break
        deranged = [Y(train[idx[j]]) for j in range(len(train))]
        R_perm = solve([X(p) for p in train], deranged)
        t_mean, _, _ = score_on(R_true, [held])
        p_mean, _, _ = score_on(R_perm, [held])
        true_scores.append(t_mean)
        perm_scores.append(p_mean)
        print(
            f"  held={held[:36]!r:39} true={t_mean:+.4f} deranged={p_mean:+.4f} delta={t_mean - p_mean:+.4f}"
        )

    tm = sum(true_scores) / len(true_scores)
    pm = sum(perm_scores) / len(perm_scores)
    print("-" * 74)
    print(f"MEAN true-pairing held-out cos = {tm:+.4f}")
    print(f"MEAN deranged-pairing held-out cos = {pm:+.4f}")
    print(f"DELTA (true - deranged) = {tm - pm:+.4f}")
    verdict = (
        "T_R SURVIVES PERMUTATION CONTROL"
        if tm - pm > 0.10
        else "T_R COLLAPSES — base-rate artifact"
    )
    print(f"VERDICT: {verdict}")
    print(
        "  criterion: delta > 0.10 (pre-registered by C25 R1 style; >C2 0.85 support)"
    )
    print("=" * 74)
    print("No files written.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
