#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
Custom Quillan Architecture BitNet 1.58b Sovereign Export.

Populates QuillanRoninSovereign (03 - Training & Model/quillan_v8_saturated.py)
using QuillanCrossModelWeightAdapter (03 - Training & Model/quillan_weight_adapter.py)
from the 4-donor {-1, 0, 1} ternary-normalized pooled checkpoints for Flagship and Mini,
saves packed ternary .safetensors checkpoints, and verifies a forward pass.

Usage:
  python -u C:\02_QUILLAN\scripts\export_quillan_sovereign_bitnet.py --self-test
  python -u C:\02_QUILLAN\scripts\export_quillan_sovereign_bitnet.py --model flagship
  python -u C:\02_QUILLAN\scripts\export_quillan_sovereign_bitnet.py --model mini
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import sys
import tempfile
from pathlib import Path
from typing import Dict

import safetensors.torch
import torch
from transformers import AutoModelForCausalLM

CORE_MODEL_DIR = Path(r"C:\02_QUILLAN\03 - Training & Model")
sys.path.insert(0, str(CORE_MODEL_DIR))
sys.path.insert(0, r"C:\02_QUILLAN\scripts")

import v23_common as c  # noqa: E402
from quillan_v8_saturated import QuillanArchConfig, QuillanRoninSovereign  # noqa: E402
from quillan_weight_adapter import QuillanCrossModelWeightAdapter, _quantize_to_ternary  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("Export_Quillan_BitNet158")


def _pack_ternary_state_dict(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Quantizes 2D linear weights to BitNet {-1, 0, 1} int8 + float32 scale tensors."""
    packed: Dict[str, torch.Tensor] = {}
    for k, v in sd.items():
        t = v.detach().cpu().float()
        if t.ndim == 2 and ("weight" in k) and ("wte" not in k) and ("lm_head" not in k):
            w_q, scale = _quantize_to_ternary(t)
            packed[k] = w_q.to(torch.int8)
            packed[f"{k}.bitnet_scale"] = scale.to(torch.float32)
        else:
            packed[k] = t.to(torch.float16) if t.dtype == torch.float32 and t.ndim >= 2 else t
    return packed


def export_quillan_sovereign_bitnet(model_key: str) -> Path:
    """Adapts a merged v23 Flagship/Mini HF checkpoint into QuillanRoninSovereign BitNet 1.58b."""
    if model_key not in ("flagship", "mini"):
        raise ValueError("Custom Quillan architecture export applies to 'flagship' and 'mini' only.")

    spec = c.MODEL_SPECS[model_key]
    candidates = [
        c.EXPORT_ROOT / f"v23_staged_{model_key}_hf",
        spec["stagec_hf"],
        spec["stageb_hf"],
    ]
    src_dir = next((p for p in candidates if p.exists()), None)
    if src_dir is None:
        raise FileNotFoundError(f"No v23 HF checkpoint found for {model_key}: {candidates}")

    logger.info("Loading %s merged checkpoint from %s...", model_key, src_dir)
    hf_model = AutoModelForCausalLM.from_pretrained(str(src_dir), local_files_only=True, dtype=torch.float32)
    src_sd = {k: v.detach().cpu().float() for k, v in hf_model.state_dict().items()}
    vocab_size = int(hf_model.config.vocab_size)
    del hf_model
    gc.collect()

    hidden_dim = 1024 if model_key == "flagship" else 512
    ffn_dim = 2048 if model_key == "flagship" else 1024
    arch_name = "llama" if model_key == "flagship" else "qwen2"

    adapter = QuillanCrossModelWeightAdapter(
        target_hidden_dim=hidden_dim,
        target_ffn_dim=ffn_dim,
        target_vocab_size=vocab_size,
        num_experts=34,
        expert_rank=8,
        ternarize=True,
    )
    adapted_sd = adapter.adapt_state_dict(src_sd, source_arch=arch_name)
    del src_sd
    gc.collect()

    cfg = QuillanArchConfig(
        text_only=True,
        multimodal=False,
        hidden_dim=hidden_dim,
        ffn_dim=ffn_dim,
        vocab_size=vocab_size,
        num_experts=34,
        num_experts_active=4,
        top_k=4,
        use_lora=True,
        device="cpu",
        eggroll_rank=128 if model_key == "flagship" else 64,
    )
    q_model = QuillanRoninSovereign(cfg).eval()
    missing, unexpected = q_model.load_state_dict(adapted_sd, strict=False)
    logger.info("Loaded adapted weights into QuillanRoninSovereign (missing=%d, unexpected=%d)", len(missing), len(unexpected))

    with torch.no_grad():
        dummy_ids = torch.randint(0, min(1000, vocab_size), (1, 8), dtype=torch.long)
        out = q_model(dummy_ids)
        logits = out[0] if isinstance(out, tuple) else out
        assert torch.isfinite(logits).all(), "Non-finite logits in QuillanRoninSovereign forward pass!"

    out_name = f"quillan-{model_key}-sovereign-bitnet158.safetensors"
    out_path = c.EXPORT_ROOT / out_name
    packed = _pack_ternary_state_dict(q_model.state_dict())
    safetensors.torch.save_file(packed, str(out_path))
    logger.info("Saved packed BitNet 1.58b Quillan checkpoint -> %s (%d bytes)", out_path, out_path.stat().st_size)
    return out_path


def run_synthetic_unit_test() -> None:
    """
    Lightweight CPU self-test verifying QuillanCrossModelWeightAdapter -> QuillanRoninSovereign
    -> packed ternary {-1, 0, 1} safetensors save/reload -> finite forward pass.
    """
    torch.manual_seed(42)
    vocab_sz, h_dim, f_dim = 256, 64, 128
    synth_sd = {
        "model.embed_tokens.weight": torch.randn(vocab_sz, h_dim) * 0.02,
        "model.norm.weight": torch.ones(h_dim),
        "lm_head.weight": torch.randn(vocab_sz, h_dim) * 0.02,
        "model.layers.0.self_attn.q_proj.weight": torch.randn(h_dim, h_dim) * 0.02,
        "model.layers.0.mlp.down_proj.weight": torch.randn(h_dim, f_dim) * 0.02,
    }
    adapter = QuillanCrossModelWeightAdapter(
        target_hidden_dim=h_dim,
        target_ffn_dim=f_dim,
        target_vocab_size=vocab_sz,
        num_experts=34,
        expert_rank=8,
        ternarize=True,
    )
    adapted_sd = adapter.adapt_state_dict(synth_sd, source_arch="llama")
    cfg = QuillanArchConfig(
        text_only=True,
        hidden_dim=h_dim,
        ffn_dim=f_dim,
        vocab_size=vocab_sz,
        num_experts=34,
        eggroll_rank=16,
        device="cpu",
    )
    q_model = QuillanRoninSovereign(cfg).eval()
    q_model.load_state_dict(adapted_sd, strict=False)

    with torch.no_grad():
        out = q_model(torch.randint(0, vocab_sz, (1, 6), dtype=torch.long))
        logits = out[0] if isinstance(out, tuple) else out
        assert torch.isfinite(logits).all(), "Non-finite forward logits!"

    packed = _pack_ternary_state_dict(q_model.state_dict())
    with tempfile.TemporaryDirectory() as tmpd:
        st_file = Path(tmpd) / "test_bitnet158.safetensors"
        safetensors.torch.save_file(packed, str(st_file))
        reloaded = safetensors.torch.load_file(str(st_file))
        int8_keys = [k for k, v in reloaded.items() if v.dtype == torch.int8]
        assert int8_keys, "No packed int8 ternary weights saved!"
        vals = torch.unique(reloaded[int8_keys[0]])
        assert all(int(x) in (-1, 0, 1) for x in vals.tolist()), f"Non-ternary values: {vals}"

    print(
        f"QUILLAN_BITNET158_TEST PASS | logits_shape={tuple(logits.shape)} "
        f"packed_int8_tensors={len(int8_keys)} ternary_vals={vals.tolist()}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Export QuillanRoninSovereign BitNet 1.58b checkpoint")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--model", choices=["flagship", "mini"], default="flagship")
    args = parser.parse_args()
    if args.self_test:
        run_synthetic_unit_test()
        return
    out = export_quillan_sovereign_bitnet(args.model)
    print(json.dumps({"model": args.model, "safetensors": str(out), "bytes": out.stat().st_size}))


if __name__ == "__main__":
    main()
