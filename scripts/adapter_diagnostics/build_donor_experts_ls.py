#!/usr/bin/env python3
"""IMPROVED donor expert bank: full least-squares + scale-normalized alignment.

Same structure as build_donor_experts.py but replaces orthogonal Procrustes
(cos 0.22-0.37) with a closed-form regularized least-squares map into Quillan's
host frame + per-donor scale normalization (user's '0.556/0.665 -> agreed
value' step). Qwen remains the host frame anchor; other donors are mapped in.

No backprop, no optimizer, no labels. All four donors incl. dequantized BF16
Falcon.
"""

from __future__ import annotations
import sys, json, shutil
from pathlib import Path

ROOT = Path(r"C:\02_QUILLAN")
BD = ROOT / "scripts" / "adapter_diagnostics"
for p in (str(ROOT / "scripts"), str(ROOT / "03 - Training & Model"), str(BD)):
    if p not in sys.path:
        sys.path.insert(0, p)

# Reuse everything from the canonical builder, then override alignment.
import build_donor_experts as B

_ORIGINAL_COMPUTE_ALIGNMENT = B.compute_alignment  # save before patching


def full_ls_align(donor_rows, host_rows, ridge=1e-3):
    A = donor_rows.float().T @ donor_rows.float()
    d = A.diagonal().mean().clamp_min(1e-8)
    A += ridge * d * __import__("torch").eye(A.shape[0])
    P = __import__("torch").linalg.solve(A, donor_rows.float().T @ host_rows.float())
    return P.float().contiguous()


# Monkeypatch: swap the orthogonal Procrustes call for full LS + scale norm.
import torch


def patched_compute_alignment(spec, host_rows, shared, d_ids, donor_map):
    if spec.name == "qwen":
        return _ORIGINAL_COMPUTE_ALIGNMENT(spec, host_rows, shared, d_ids, donor_map)
    donor_rows = B.read_embed_rows(spec.ckpt, "model.embed_tokens.weight", d_ids)
    P = full_ls_align(donor_rows, host_rows)
    # scale-normalize: make matched donor rows match host row magnitudes
    proj = donor_rows @ P
    s = host_rows.norm() / proj.norm().clamp_min(1e-8)
    P = P * s
    proj = donor_rows @ P
    cos = B.mean_pair_cosine(proj, host_rows)
    diag = {
        "n_shared_tokens": len(shared),
        "P_shape": [int(x) for x in P.shape],
        "align_cos_mean": float(cos),
        "align_cos_finite": True,
        "scale_normalized": float(s.item()),
        "identity_host_frame": False,
        "method": "full_ls_scale_norm",
    }
    del donor_rows
    return P, diag


B.compute_alignment = patched_compute_alignment

# Point outputs to new files so canonical bank/report are untouched.
B.OUT_FILE = ROOT / "exports" / "donor_experts_ls.safetensors"
B.REPORT_FILE = ROOT / "exports" / "donor_experts_ls.report.json"


def main():
    print(f"Building LS+scale-norm expert bank -> {B.OUT_FILE}", flush=True)
    # pass explicit args so B.main() does not override our output paths
    sys.argv = [
        sys.argv[0],
        "--ffn",
        "512",
        "--out",
        str(B.OUT_FILE),
        "--report",
        str(B.REPORT_FILE),
    ]
    B.main()


if __name__ == "__main__":
    main()
