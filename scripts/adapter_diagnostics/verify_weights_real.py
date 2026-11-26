#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""VERIFY the mini checkpoint actually contains Qwen's learned weights (not random init).

Compare embed_tokens.weight_int8 (mini checkpoint) against Qwen 0.5b's real
token_embeddings, after the same int8 packing. If they match, the transplant
worked; if they're random, the build silently failed (C26's root-cause claim).
Read-only.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(r"C:\02_QUILLAN")
sys.path.insert(0, str(ROOT / "03 - Training & Model"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from safetensors import safe_open  # noqa: E402
from quillan_sovereign_bitnet import QuillanSovereignBitNetLM, PackedInt8Embedding  # noqa: E402

MINI = ROOT / "exports" / "quillan-mini-sovereign-bitnet.safetensors"
QWEN = Path(r"C:\02_QUILLAN\exports\donor_cache_v23\qwen_0.5b_donors.pt")


def main() -> int:
    print("=" * 78)
    print("VERIFY: does mini checkpoint hold REAL Qwen weights?")
    print("=" * 78)
    # mini embed
    with safe_open(str(MINI), framework="pt") as f:
        me_int8 = f.get_tensor("embed_tokens.weight_int8").float()
        me_scale = f.get_tensor("embed_tokens.weight_scale").float()
    mini_E = me_int8 * me_scale  # [vocab, hidden] approx
    print(f"mini embed: {tuple(mini_E.shape)} mean_abs={mini_E.abs().mean():.5f}")

    # Qwen real embed
    d = torch.load(str(QWEN), map_location="cpu", weights_only=False)
    qwen_E = d["embed_tokens"].float()  # [151936, 896]
    print(f"qwen embed: {tuple(qwen_E.shape)} mean_abs={qwen_E.abs().mean():.5f}")

    # Align scale: compare normalized cosine of rows
    n = min(mini_E.shape[0], qwen_E.shape[0])
    m = mini_E[:n]
    q = qwen_E[:n]
    cos = F.cosine_similarity(m.flatten(), q.flatten(), dim=0).item()
    print(f"global flatten cos(mini, qwen) = {cos:.6f}")
    # row-wise mean cos
    row_cos = F.cosine_similarity(m, q, dim=1).mean().item()
    print(f"row-wise mean cos = {row_cos:.6f}")
    # random baseline
    torch.manual_seed(0)
    rand = torch.randn_like(q)
    rand_cos = F.cosine_similarity(m, qwen_E[:n] * 0 + rand, dim=1).mean().item()
    print(f"row-wise cos vs RANDOM = {rand_cos:.6f}")

    verdict = (
        "REAL WEIGHTS"
        if row_cos > 0.5
        else ("RANDOM/INIT" if abs(row_cos) < 0.1 else "AMBIGUOUS")
    )
    print(f"\nVERDICT: {verdict} (row_cos={row_cos:.4f})")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
