#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Focused probe: why do q/k/v/o projections output exactly zero? (brief #911)"""

from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(r"C:\02_QUILLAN")
for p in (str(ROOT / "scripts"), str(ROOT / "03 - Training & Model")):
    if p not in sys.path:
        sys.path.insert(0, p)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
import torch
from probe_longctx_quality import build_real_model

model, meta = build_real_model()
print(f"params={meta['param_count']:,}")


def stats(name, t):
    t = t.detach().float()
    print(
        f"{name:<44} shape={str(tuple(t.shape)):<16} "
        f"mean_abs={t.abs().mean().item():.6e} max_abs={t.abs().max().item():.6e} "
        f"n_nonzero={(t != 0).sum().item()}/{t.numel()} all_zero={bool((t == 0).all())}"
    )


# Weight stats for the suspicious modules
with torch.no_grad():
    for path in (
        "diffusion_core.couil_attn.q_proj.weight",
        "diffusion_core.couil_attn.k_proj.weight",
        "diffusion_core.couil_attn.v_proj.weight",
        "diffusion_core.couil_attn.o_proj.weight",
        "diffusion_core.ffn.0.weight",
        "decomposition.vectors.Language.weight",
        "ingestion.q1_ingest.weight",
        "ingestion.txt_emb.weight",
        "ingestion.pos_embed",
    ):
        mod = dict(model.named_parameters())[path]
        stats("W " + path, mod.data)

    # Feed a nonzero random input directly into q_proj and inspect output
    torch.manual_seed(0)
    x = torch.randn(1, 5, 896)
    stats("INPUT random x", x)
    qp = model.diffusion_core.couil_attn.q_proj
    stats("q_proj.quantize_act", torch.tensor(float(qp.quantize_act)))
    stats("q_proj.quantize_weight", torch.tensor(float(qp.quantize_weight)))
    stats("q_proj.eggroll_active", torch.tensor(float(qp.eggroll_active)))
    y = qp(x)
    stats("q_proj(random x) OUTPUT", y)
    # Manually compute linear with raw weight
    stats("F.linear(x, raw_w)", torch.nn.functional.linear(x, qp.weight))
    # Check the quantizer
    from quillan_v8_saturated import _weight_quant

    wq = _weight_quant(qp.weight, qp.eps)
    stats("_weight_quant(w)", wq)
    stats("F.linear(x, wq)", torch.nn.functional.linear(x, wq))
