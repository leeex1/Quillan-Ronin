#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Decode the pooled-readout logits (C17 semantic-fidelity check). READ-ONLY.

Reproduce mean_all + center + correct-fold logits from frozen caches and
decode top-5 with Qwen tokenizer. Compare pooled vs norm-matched random null.
If pooled gives prompt-correct continuations (Paris, names) >> random garbage,
the centroid carries real semantics (C17). If both garbage, C9 was right.
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
from transformers import AutoTokenizer  # noqa: E402

STAGE, LAM = "final_norm", 0.01
PROMPTS = [
    "The capital of France is",
    "2 + 2 =",
    "Hello, my name is",
    "Quillan-Ronin is a sovereign",
    "The quick brown fox jumps over the lazy dog and then",
]


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
    tok = AutoTokenizer.from_pretrained(
        r"C:\Users\Admin\.cache\huggingface\hub\models--Qwen--Qwen2.5-0.5B-Instruct\snapshots\7ae557604adf67be50417f59c2c2f167def9a775"
    )

    def H(p):
        return custom[p][STAGE][0].float().mean(dim=0)

    def Y(p):
        return donor[p]["final_hidden"][0].float().mean(dim=0)

    X = torch.stack([H(p) for p in PROMPTS], 0)
    Ys = torch.stack([Y(p) for p in PROMPTS], 0)
    A = X.T @ X + LAM * torch.eye(X.shape[1], dtype=X.dtype)
    R = torch.linalg.solve(A, X.T @ Ys)
    W = Wu @ R.T
    mu = X.mean(dim=0)

    # norm-matched random null (same Frobenius scale as W@ centered input)
    scale = (Ys.norm(dim=1).mean() / X.norm(dim=1).mean()).item()
    Rrand = torch.randn(896, 896) * (scale / (896**0.5))
    Wr = Wu @ Rrand.T

    print("=" * 82)
    print("POOLED-READOUT SEMANTIC DECODE (read-only, frozen caches)")
    print("=" * 82)
    for p in PROMPTS:
        hp = H(p) - mu
        with torch.no_grad():
            cl_pool = hp @ W.T
            cl_rand = hp @ Wr.T
        tp = [tok.decode([i]) for i in torch.topk(cl_pool, 5).indices.tolist()]
        tr = [tok.decode([i]) for i in torch.topk(cl_rand, 5).indices.tolist()]
        print(f"{p[:40]!r:42}")
        print(f"   POOLED -> {tp}")
        print(f"   RANDOM -> {tr}")
    print("=" * 82)
    print("No files written.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
