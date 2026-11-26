#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build Quillan FLAGSHIP with the corrected four-donor expert bank.

Flagship config (16L/2048/ffn5120, Qwen-shaped Llama host):
  - 996M logical params at 1 expert layer — a 2nd expert layer would be 1050M,
    breaking the immutable <=1B ceiling. So we keep the single expert layer
    (base config moe_layer_indices=(14,)) as required by the budget.
  - Uses the SAME corrected four-donor expert bank as mini (all 4 donors real,
    Falcon dequantized BF16), fused with the flagship host via the proven
    closed-form pipeline. Zero backprop.
  - Writes NEW checkpoint names; canonical + gate untouched.
"""

from __future__ import annotations

import gc, json, math, sys, time
from pathlib import Path

ROOT = Path(r"C:\02_QUILLAN")
for p in (str(ROOT / "scripts"), str(ROOT / "03 - Training & Model")):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch
import build_quillan_trio_bitnet as B
import v23_common as c
from quillan_sovereign_bitnet import QuillanSovereignBitNetLM, QuillanSovereignConfig

OUT_CKPT = c.EXPORT_ROOT / "quillan-flagship-dense-council.safetensors"
OUT_REPORT = c.EXPORT_ROOT / "quillan-flagship-dense-council-report.json"


# RCI-5: align_and_fuse_4donors_float() creates its OWN cfg via for_model(), so
# monkeypatch the flagship preset so BOTH that function and this script see the
# same proven-mini shape. Otherwise the MoE seed bank uses flagship defaults
# (moe_ffn 256) while the model uses 1280 -> shape mismatch at _seed_council_moe.
_ORIG_FOR_MODEL = QuillanSovereignConfig.for_model.__func__


def _patched_for_model(cls, key: str):
    if key == "flagship":
        c = _ORIG_FOR_MODEL(cls, "mini")  # proven recipe as base
        c.model_name = "quillan-flagship-dense-council"
        c.moe_layer_indices = (0, 8, 15, 23)
        c.moe_ffn_dim = 1280
        return c
    return _ORIG_FOR_MODEL(cls, key)


QuillanSovereignConfig.for_model = classmethod(_patched_for_model)


def main() -> None:
    t0 = time.time()
    cfg = QuillanSovereignConfig.for_model("flagship")
    cfg.model_name = "quillan-flagship-dense-council"

    # RCI-4 (#1267): Qwen-1.5B host (head_dim=128) packed catastrophically
    # (PPL 37M, echo collapse J2/J3 FAIL). The PROVEN mini recipe (Qwen-0.5B host,
    # head_dim=64, 24L, 151936 vocab) passes 7/10. Scale that recipe via a
    # LARGER expert bank (moe_ffn 512 -> 1280) = 971M logical, still <=1B.
    # Genuine flagship: same Qwen family + unified tokenizer + 4 expert layers
    # at higher capacity than mini. Uses the SAME host as the passing mini.
    QWEN05 = Path(
        r"C:\Users\Admin\.cache\huggingface\hub\models--Qwen--Qwen2.5-0.5B-Instruct\snapshots\7ae557604adf67be50417f59c2c2f167def9a775"
    )
    cfg.vocab_size = 151936
    cfg.num_hidden_layers = 24
    cfg.hidden_size = 896
    cfg.intermediate_size = 4864
    cfg.num_attention_heads = 14
    cfg.num_key_value_heads = 2
    cfg.head_dim = 64  # PROVEN mini value
    cfg.moe_layer_indices = (0, 8, 15, 23)
    cfg.moe_ffn_dim = 1280  # larger than mini's 512 -> genuine flagship capacity
    c.MODEL_SPECS["flagship"]["base_hf"] = QWEN05
    # Fusion spec must match the new Qwen host (was llama_1b/16L for old flagship).
    c.MODEL_SPECS["flagship"]["self_donor"] = "qwen_0.5b"
    c.MODEL_SPECS["flagship"]["num_layers"] = 24
    c.MODEL_SPECS["flagship"]["target_layers"] = [0, 3, 7, 10, 13, 16, 20, 23]
    c.MODEL_SPECS["flagship"]["hidden_size"] = 896
    c.MODEL_SPECS["flagship"]["intermediate_size"] = 4864
    print("=" * 70)
    print(
        "Quillan FLAGSHIP RCI-4 | Qwen-0.5B host (proven) | 24L/896h | 4 experts moe_ffn=1280"
    )
    print("=" * 70)

    calib_texts, val_texts = B.stage_b.build_calib_and_val_splits(c.CLEAN_DATA_PATH)
    host_model, tok = c.load_clean_host("flagship", dtype=torch.float32)

    # prune host FFN (8960 -> 4864) with closed-form salience (no backprop)
    B.prune_host_swiglu_salience(host_model, tok, cfg.intermediate_size, calib_texts)

    base_val_nll = c.evaluate_texts_nll(host_model, tok, val_texts, max_len=160)
    print(f"[host] base val_nll={base_val_nll:.5f} ppl={math.exp(base_val_nll):.3f}")

    donor_meta, alpha, fused_val_nll, moe_seed_bank = B.align_and_fuse_4donors_float(
        "flagship", host_model, tok, calib_texts, val_texts, base_val_nll
    )
    print(
        f"[fusion] alpha={alpha} base_nll={base_val_nll:.5f} fused_nll={fused_val_nll:.5f}"
    )

    sov = B.pack_host_into_sovereign(
        host_model, cfg, moe_seed_bank, tokenizer=tok, calib_texts=calib_texts
    )
    del host_model, moe_seed_bank
    gc.collect()

    logical = sov.count_logical_parameters()
    actual = sum(p.numel() for p in sov.parameters()) + sum(
        b.numel() for b in sov.buffers()
    )
    print(
        f"[params] logical={logical:,} ({logical / 1e6:.0f}M) actual={actual:,} ({actual / 1e6:.0f}M)"
    )
    assert logical <= 1_000_000_000, f"flagship exceeded 1B logical ceiling: {logical}"

    sov.save_packed_safetensors(OUT_CKPT)
    del sov
    gc.collect()

    reloaded = QuillanSovereignBitNetLM.load_packed_safetensors(OUT_CKPT, device="cpu")
    packed_nll = B.evaluate_sovereign_nll(reloaded, tok, val_texts[:12], max_len=160)
    report = {
        "model": "flagship-dense-council",
        "expert_layers": list(cfg.moe_layer_indices),
        "logical_params": logical,
        "actual_params": actual,
        "base_val_nll": round(base_val_nll, 5),
        "fused_val_nll": round(fused_val_nll, 5),
        "alpha": alpha,
        "packed_val_nll_12": round(packed_nll, 5),
        "packed_ppl_12": round(math.exp(packed_nll), 4),
        "donors": donor_meta,
        "build_method": "closed-form 4-donor fusion, zero backprop, 1 expert layer (1B budget)",
        "elapsed_s": round(time.time() - t0, 1),
    }
    OUT_REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("[done] " + json.dumps(report, indent=2))
    print(f"[done] checkpoint -> {OUT_CKPT}")


if __name__ == "__main__":
    main()
