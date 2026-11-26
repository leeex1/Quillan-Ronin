#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build Quillan MINI with the locked dense-pull EvoMoE + swarm hybrid.

Differences from build_quillan_trio_bitnet.py (which produced the validated Qwen-backed mini):
  1. Expert council spaced across 4 layers (0, 8, 15, 23) instead of 2 (11, 22).
     -> 690M logical / ~569M actual params, inside the user's 0.6-0.7B target.
  2. New custom four-donor tokenizer (exports/quillan_tokenizer_v54_fourdonor) used for
     the host's calibration/validation text encoding where the model vocab aligns.
  3. Writes to a NEW checkpoint name so the validated artifact and gate sha are untouched.

All fusion math is reused verbatim from build_quillan_trio_bitnet (closed-form,
zero backprop). No gate thresholds changed. No WSP. No training.
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(r"C:\02_QUILLAN")
for p in (str(ROOT / "scripts"), str(ROOT / "03 - Training & Model")):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch

import build_quillan_trio_bitnet as B
import v23_common as c
from quillan_sovereign_bitnet import QuillanSovereignBitNetLM, QuillanSovereignConfig

EXPERT_LAYERS = (0, 8, 15, 23)
OUT_CKPT = c.EXPORT_ROOT / "quillan-mini-dense-council.safetensors"
OUT_REPORT = c.EXPORT_ROOT / "quillan-mini-dense-council-report.json"
TOK_DIR = c.EXPORT_ROOT / "quillan_tokenizer_v54_fourdonor"


def main() -> None:
    t0 = time.time()
    cfg = QuillanSovereignConfig.for_model("mini")
    cfg.moe_layer_indices = EXPERT_LAYERS
    cfg.model_name = "quillan-mini-dense-council"
    print("=" * 70)
    print(f"Quillan MINI dense-council build | expert layers {EXPERT_LAYERS}")
    print("=" * 70)

    calib_texts, val_texts = B.stage_b.build_calib_and_val_splits(c.CLEAN_DATA_PATH)
    host_model, tok = c.load_clean_host("mini", dtype=torch.float32)

    # NOTE: keep the host's native Qwen tokenizer for weight-space extraction —
    # token IDs 0..151642 are identical to our new custom tokenizer; the 5
    # sovereign specials are appended above the Qwen base and do not disturb the
    # embedding rows used here. New custom tokenizer is the runtime tokenizer.
    tok_path = TOK_DIR / "tokenizer.json"
    if tok_path.exists():
        print(
            f"[tokenizer] custom available at {TOK_DIR} (runtime); host uses aligned Qwen base for extraction"
        )

    base_val_nll = c.evaluate_texts_nll(host_model, tok, val_texts, max_len=160)
    print(f"[host] base val_nll={base_val_nll:.5f} ppl={math.exp(base_val_nll):.3f}")

    donor_meta, alpha, fused_val_nll, moe_seed_bank = B.align_and_fuse_4donors_float(
        "mini", host_model, tok, calib_texts, val_texts, base_val_nll
    )
    print(
        f"[fusion] alpha={alpha} base_nll={base_val_nll:.5f} fused_nll={fused_val_nll:.5f}"
    )

    sov = B.pack_host_into_sovereign(
        host_model, cfg, moe_seed_bank, tokenizer=tok, calib_texts=calib_texts
    )
    del host_model, moe_seed_bank
    import gc

    gc.collect()

    logical = sov.count_logical_parameters()
    actual = sum(p.numel() for p in sov.parameters()) + sum(
        b.numel() for b in sov.buffers()
    )
    print(
        f"[params] logical={logical:,} ({logical / 1e6:.0f}M) actual={actual:,} ({actual / 1e6:.0f}M)"
    )
    assert logical <= 1_000_000_000, "exceeded 1B logical ceiling"
    assert actual <= 1_000_000_000, "exceeded 1B actual ceiling"

    sov.save_packed_safetensors(OUT_CKPT)
    del sov
    gc.collect()

    reloaded = QuillanSovereignBitNetLM.load_packed_safetensors(OUT_CKPT, device="cpu")
    packed_nll = B.evaluate_sovereign_nll(reloaded, tok, val_texts[:12], max_len=160)
    report = {
        "model": "mini-dense-council",
        "expert_layers": list(EXPERT_LAYERS),
        "logical_params": logical,
        "actual_params": actual,
        "base_val_nll": round(base_val_nll, 5),
        "fused_val_nll": round(fused_val_nll, 5),
        "alpha": alpha,
        "packed_val_nll_12": round(packed_nll, 5),
        "packed_ppl_12": round(math.exp(packed_nll), 4),
        "tokenizer": str(TOK_DIR),
        "donors": donor_meta,
        "build_method": "closed-form fusion, zero backprop; expert council spaced 4 layers",
        "elapsed_s": round(time.time() - t0, 1),
    }
    OUT_REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("[done] " + json.dumps(report, indent=2))
    print(f"[done] checkpoint -> {OUT_CKPT}")


if __name__ == "__main__":
    main()
