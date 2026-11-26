#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""C18-SHEPHERD addendum: weight-level provenance + length-only collapse control.

Read-only. Uses:
  - exports\\donor_cache_v23\\qwen_0.5b_donors.pt  (exact donor tensors used at export)
  - exports\\quillan-mini-sovereign-bitnet158.safetensors (packed custom checkpoint)
  - _acts\\{donor,custom}_acts.pt (captured in transfer_verification.py)

Writes ONLY: addendum_confirm_results.json under this dir.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(r"C:\02_QUILLAN")
DIAG = ROOT / "scripts" / "adapter_diagnostics"
sys.path.insert(0, str(ROOT / "scripts"))
sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")

import torch
import torch.nn.functional as F

DONOR_PT = ROOT / "exports" / "donor_cache_v23" / "qwen_0.5b_donors.pt"
CKPT = ROOT / "exports" / "quillan-mini-sovereign-bitnet158.safetensors"
ACTS = DIAG / "_acts"

R: dict = {"owner": "c18-shepherd", "brief": 911, "read_only": True}

# ---- 1. exact donor tensor vs checkpoint embedding tables ----
donor = torch.load(str(DONOR_PT), map_location="cpu", weights_only=True)
d_emb = donor["embed_tokens"].float()
from safetensors import safe_open

with safe_open(str(CKPT), framework="pt", device="cpu") as f:
    keys = set(f.keys())
    c_txt_emb = (
        f.get_tensor("ingestion.txt_emb.weight").float()
        if "ingestion.txt_emb.weight" in keys
        else None
    )
    c_txt_dec = (
        f.get_tensor("txt_dec.weight").float() if "txt_dec.weight" in keys else None
    )
    c_pos = (
        f.get_tensor("ingestion.pos_embed").float()
        if "ingestion.pos_embed" in keys
        else None
    )


def cosw(a, b):
    """Row-wise (per-token) cosine averaged -- numerically stable vs one giant dot."""
    a = a.float().reshape(-1, a.shape[-1])
    b = b.float().reshape(-1, b.shape[-1])
    assert a.shape == b.shape, (a.shape, b.shape)
    c = F.cosine_similarity(a, b, dim=-1)
    return round(float(c.mean()), 8)


R["weight_level"] = {
    "donor_embed_tokens_shape": list(d_emb.shape),
    "custom_txt_emb_key_present": c_txt_emb is not None,
    "custom_txt_dec_key_present": c_txt_dec is not None,
    "cosine_donor_embed_vs_custom_txt_emb": cosw(d_emb, c_txt_emb)
    if c_txt_emb is not None
    else None,
    "cosine_donor_embed_vs_custom_txt_dec": cosw(d_emb, c_txt_dec)
    if c_txt_dec is not None
    else None,
    "donor_embed_mean_abs": round(float(d_emb.abs().mean()), 6),
    "custom_txt_emb_mean_abs": round(float(c_txt_emb.abs().mean()), 6)
    if c_txt_emb is not None
    else None,
    "custom_txt_emb_dtype_in_file": str(
        __import__("safetensors")
        .safe_open(str(CKPT), framework="pt", device="cpu")
        .get_slice("ingestion.txt_emb.weight")
        .get_dtype()
    )
    if c_txt_emb is not None
    else None,
}

# ---- 2. pos_embed scale + sinusoidal-vs-random discrimination ----
if c_pos is not None:
    L, D = c_pos.shape[1], c_pos.shape[2]
    pos = c_pos[0]
    # reconstruct a canonical sinusoidal table of same shape for comparison
    import numpy as np

    pe = np.zeros((L, D), dtype=np.float32)
    for i in range(L):
        for j in range(0, D, 2):
            w = 1.0 / (10000.0 ** (j / D))
            pe[i, j] = math.sin(i * w)
            if j + 1 < D:
                pe[i, j + 1] = math.cos(i * w)
    pe_t = torch.from_numpy(pe)
    R["pos_embed"] = {
        "shape": list(c_pos.shape),
        "mean_abs": round(float(pos.abs().mean()), 6),
        "max_abs": round(float(pos.abs().max()), 4),
        "std": round(float(pos.std()), 6),
        "cosine_vs_canonical_sinusoidal": cosw(pos, pe_t),
        "scale_ratio_pos_over_txt_emb": round(
            float(pos.abs().mean() / c_txt_emb.abs().mean()), 2
        ),
    }

# ---- 3. same-length different-content prompt collapse (custom logits) ----
c = torch.load(str(ACTS / "custom_acts.pt"), map_location="cpu", weights_only=False)
d = torch.load(str(ACTS / "donor_acts.pt"), map_location="cpu", weights_only=False)
A, B = "The capital of France is", "Hello, my name is"  # both 5 tokens
A2 = "2 + 2 ="  # 5 tokens in Qwen tokenizer
LONG = "The quick brown fox jumps over the lazy dog and then"  # 11 tokens
pairs = [(A, B), (A, A2), (A, LONG)]
out = {}
for x, y in pairs:
    la = c[x]["logits"][:, -1, :].float().reshape(1, -1)
    lb = c[y]["logits"][:, -1, :].float().reshape(1, -1)
    da = d[x]["logits"][:, -1, :].float().reshape(1, -1)
    db = d[y]["logits"][:, -1, :].float().reshape(1, -1)
    out[f"{x!r} vs {y!r}"] = {
        "token_lengths": [len(c[x]["ids"]), len(c[y]["ids"])],
        "custom_logit_cosine": round(float(F.cosine_similarity(la, lb)[0]), 6),
        "donor_logit_cosine": round(float(F.cosine_similarity(da, db)[0]), 6),
        "custom_max_abs_logit_diff": round(float((la - lb).abs().max()), 6),
    }
R["length_only_collapse_control"] = out

# ---- 4. is custom rank correlated with donor rank at all? (Spearman on last pos) ----
from scipy.stats import spearmanr

sp = []
for p in c:
    a = c[p]["logits"][0, -1, :].float().numpy()
    b = d[p]["logits"][0, -1, :].float().numpy()
    rho = spearmanr(a, b).statistic
    sp.append(round(float(rho), 6))
R["spearman_logit_rank_correlation_custom_vs_donor"] = {
    "per_prompt": dict(zip(list(c.keys()), sp)),
    "mean": round(sum(sp) / len(sp), 6),
}

op = DIAG / "addendum_confirm_results.json"
op.write_text(json.dumps(R, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps(R, indent=2, ensure_ascii=False))
print("WROTE", op)
