#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
Custom Quillan Architecture BitNet 1.58b Sovereign Export.  [C10 HARDENED]

Populates ``QuillanRoninSovereign`` (03 - Training & Model/quillan_v8_saturated.py)
from a merged HF donor checkpoint using ``QuillanCrossModelWeightAdapter`` with:

  * target dims derived AT RUNTIME from ``QuillanRoninSovereign.state_dict()``;
  * source dims derived AT RUNTIME from the donor ``config.json`` (never 1024/2048
    guesses) so the full 1.2B donor is retained instead of being cropped away;
  * ``load_state_dict(..., strict=True)`` with zero missing / unexpected keys;
  * least-squares (absmean) ternary packing that PRESERVES gamma and never
    re-quantizes an already-ternary tensor with a fresh scale.

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
from pathlib import Path
from typing import Dict, Optional

import safetensors.torch
import torch
from transformers import AutoModelForCausalLM

CORE_MODEL_DIR = Path(r"C:\02_QUILLAN\03 - Training & Model")
sys.path.insert(0, r"C:\02_QUILLAN\scripts")
sys.path.insert(0, str(CORE_MODEL_DIR))

# Import our hardened modules BEFORE v23_common: that legacy helper prepends
# "...\03 - Training & Model\scripts" to sys.path, which contains an obsolete
# quillan_weight_adapter.py that would shadow the canonical one we just fixed.
from quillan_v8_saturated import QuillanArchConfig, QuillanRoninSovereign  # noqa: E402
from quillan_weight_adapter import (  # noqa: E402
    SourceDims,
    QuillanCrossModelWeightAdapter,
    _absmean_ternarize,
    _is_ternary,
)

import v23_common as c  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("Export_Quillan_BitNet158")

# 2D linear weights that carry pretrained knowledge. Images / routers / LoRA
# adapters / token tables are intentionally excluded from ternary packing.
_PACK_SKIP_TOKENS = (
    "txt_emb",
    "txt_dec",
    "pos_embed",
    "lora",
    "router",
    "sparse_mask_generator",
    "img_proj",
    "ingest_gate",
    "quillan_gate",
)


def _pack_ternary_state_dict(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Pack knowledge 2D weights as int8 {-1,0,1} + a float32 ``bitnet_scale``.

    Scale propagation invariant: if the source tensor is already ternary, its
    existing ``<key>.bitnet_scale`` is reused (or gamma=1.0 if none is stored).
    We NEVER recompute an absmean scale from an already-ternary tensor, which
    would shrink gamma to the non-zero fraction and corrupt the reconstruction.
    """
    packed: Dict[str, torch.Tensor] = {}
    for key, value in sd.items():
        if key.endswith(".bitnet_scale"):
            continue
        tensor = value.detach().cpu()
        is_packable = (
            tensor.ndim == 2
            and key.endswith("weight")
            and not any(token in key for token in _PACK_SKIP_TOKENS)
        )
        if is_packable:
            scale_key = f"{key}.bitnet_scale"
            if _is_ternary(tensor):
                # Already quantized: preserve the original gamma, no re-quant.
                gamma = (
                    sd[scale_key].detach().cpu().float()
                    if scale_key in sd
                    else torch.ones((), dtype=torch.float32)
                )
            else:
                ternary, gamma = _absmean_ternarize(tensor)
                tensor = ternary
            packed[key] = tensor.to(torch.int8).contiguous()
            packed[scale_key] = gamma.to(torch.float32).reshape(()).contiguous()
        else:
            packed[key] = (
                tensor.to(torch.float16)
                if tensor.is_floating_point() and tensor.ndim >= 2
                else tensor
            )
    return packed


def _source_dims_from_config(hf_config: object, arch: str) -> SourceDims:
    """Read authoritative donor dims from the HF config (no hardcoded guesses)."""
    hidden = int(getattr(hf_config, "hidden_size"))
    ffn = int(getattr(hf_config, "intermediate_size"))
    vocab = int(getattr(hf_config, "vocab_size"))
    layers = int(getattr(hf_config, "num_hidden_layers"))
    return SourceDims(hidden=hidden, ffn=ffn, vocab=vocab, num_layers=layers, arch=arch)


def export_quillan_sovereign_bitnet(
    model_key: str, out_name: Optional[str] = None, merge_embeddings: bool = False
) -> Path:
    """Adapt a merged v23 Flagship/Mini HF checkpoint into QuillanRoninSovereign BitNet 1.58b.

    C10 Phase-0/1 minimal addition: ``out_name`` lets a candidate be written to a
    NEW file (e.g. ``...-v2.safetensors``) so the canonical checkpoint is never
    overwritten. Default behaviour is unchanged when ``out_name`` is omitted.

    #956 addition: ``merge_embeddings=True`` replaces the single-donor embedding
    with the 3-donor TOKEN-STRING-aligned merge (qwen+llama+dragon) for BOTH
    ``txt_emb`` and ``txt_dec``. It REQUIRES an explicit ``out_name`` so the
    canonical checkpoint (sha 53E60D33...A5C) can never be overwritten.
    """
    if merge_embeddings and not out_name:
        raise ValueError(
            "#956 merge_embeddings requires an explicit NEW --out-name; "
            "the canonical checkpoint must never be overwritten."
        )
    if model_key not in ("flagship", "mini"):
        raise ValueError(
            "Custom Quillan architecture export applies to 'flagship' and 'mini' only."
        )

    spec = c.MODEL_SPECS[model_key]
    candidates = [
        c.EXPORT_ROOT / f"v23_staged_{model_key}_hf",
        spec["stagec_hf"],
        spec["stageb_hf"],
        spec["base_hf"],
    ]
    src_dir = next((p for p in candidates if Path(p).exists()), None)
    if src_dir is None:
        raise FileNotFoundError(
            f"No v23 HF checkpoint found for {model_key}: {candidates}"
        )

    logger.info("Loading %s merged checkpoint from %s...", model_key, src_dir)
    hf_model = AutoModelForCausalLM.from_pretrained(
        str(src_dir), local_files_only=True, dtype=torch.float32
    )
    hf_model.eval()
    src_sd: Dict[str, torch.Tensor] = {
        k: v.detach().cpu().float() for k, v in hf_model.state_dict().items()
    }
    dims = _source_dims_from_config(
        hf_model.config, str(getattr(hf_model.config, "model_type", "llama"))
    )
    logger.info(
        "Source config dims (AUTHORITATIVE): hidden=%d ffn=%d vocab=%d layers=%d arch=%s",
        dims.hidden,
        dims.ffn,
        dims.vocab,
        dims.num_layers,
        dims.arch,
    )
    del hf_model
    gc.collect()

    # The custom arch is built to the SOURCE dims so donor tensors are retained
    # rather than cropped to the legacy 1024/2048 target. Its depth is fixed
    # (single diffusion attention block + single MoE stage); N donor layers are
    # consolidated by the adapter's documented mean/cycle policy.
    cfg = QuillanArchConfig(
        text_only=True,
        multimodal=False,
        hidden_dim=dims.hidden,
        ffn_dim=dims.ffn,
        vocab_size=dims.vocab,
        num_experts=34,
        num_experts_active=4,
        top_k=4,
        use_lora=True,
        device="cpu",
        eggroll_rank=128 if model_key == "flagship" else 64,
    )
    q_model = QuillanRoninSovereign(cfg).eval()

    # #956: optionally replace the single-donor embedding with the 3-donor,
    # token-string-aligned merge. Legacy export is unchanged when the flag is off.
    embed_override = None
    if merge_embeddings:
        diag_dir = Path(__file__).resolve().parent / "adapter_diagnostics"
        if str(diag_dir) not in sys.path:
            sys.path.insert(0, str(diag_dir))
        from merge_donor_embeddings import (  # local import: heavy deps only when used
            merge_donor_embeddings as _merge_three_donors,
        )

        logger.info("#956 merging qwen+llama+dragon embeddings by TOKEN STRING...")
        merged = _merge_three_donors(verbose=True)
        embed_override = merged["float_weight"]  # float effective q*gamma
        logger.info(
            "#956 embedding coverage: %s",
            json.dumps(merged["coverage"], ensure_ascii=False),
        )
        del merged
        gc.collect()

    adapter = QuillanCrossModelWeightAdapter(
        target_hidden_dim=dims.hidden,
        target_ffn_dim=dims.ffn,
        target_vocab_size=dims.vocab,
        num_experts=34,
        expert_rank=8,
        ternarize=False,  # packing performs the single, gamma-preserving quantization
    )
    adapted_sd = adapter.adapt_state_dict(
        src_sd,
        source_arch=dims.arch,
        target_model=q_model,
        source_dims=dims,
        embedding_override=embed_override,
    )
    del src_sd
    gc.collect()

    missing, unexpected = q_model.load_state_dict(adapted_sd, strict=True)
    assert not missing and not unexpected, (
        f"STRICT LOAD FAILED: missing={len(missing)} unexpected={len(unexpected)}; "
        f"missing_sample={missing[:5]} unexpected_sample={unexpected[:5]}"
    )
    logger.info(
        "STRICT LOAD OK: missing=0 unexpected=0 (all %d keys covered)", len(adapted_sd)
    )

    with torch.no_grad():
        dummy_ids = torch.arange(8, dtype=torch.long).unsqueeze(0) % min(
            1000, dims.vocab
        )
        out = q_model(dummy_ids)
        logits = out[0] if isinstance(out, tuple) else out
        assert torch.isfinite(logits).all(), (
            "Non-finite logits in QuillanRoninSovereign forward pass!"
        )

    resolved_name = out_name or f"quillan-{model_key}-sovereign-bitnet158.safetensors"
    out_path = c.EXPORT_ROOT / resolved_name
    if out_name is not None and out_path.exists():
        raise FileExistsError(
            f"refusing to overwrite candidate output (use a new --out-name): {out_path}"
        )
    packed = _pack_ternary_state_dict(q_model.state_dict())
    safetensors.torch.save_file(packed, str(out_path))
    logger.info(
        "Saved packed BitNet 1.58b Quillan checkpoint -> %s (%d bytes)",
        out_path,
        out_path.stat().st_size,
    )
    return out_path


def run_synthetic_unit_test() -> None:
    """CPU self-test: strict transfer into the REAL custom arch + packed save/reload."""
    torch.manual_seed(42)
    vocab_sz, h_dim, f_dim, n_layers = 256, 64, 128, 3

    def det(shape: tuple[int, ...], offset: float) -> torch.Tensor:
        n = 1
        for s in shape:
            n *= s
        return torch.arange(n, dtype=torch.float32).reshape(shape) * 0.001 + offset

    synth_sd: Dict[str, torch.Tensor] = {
        "model.embed_tokens.weight": det((vocab_sz, h_dim), 0.5),
        "model.norm.weight": det((h_dim,), 2.0),
    }
    for layer in range(n_layers):
        synth_sd[f"model.layers.{layer}.input_layernorm.weight"] = det(
            (h_dim,), 1.0 + 0.01 * layer
        )
        synth_sd[f"model.layers.{layer}.post_attention_layernorm.weight"] = det(
            (h_dim,), 1.1 + 0.01 * layer
        )
        for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
            synth_sd[f"model.layers.{layer}.self_attn.{proj}.weight"] = det(
                (h_dim, h_dim), 0.1 * layer
            )
        for proj in ("gate_proj", "up_proj"):
            synth_sd[f"model.layers.{layer}.mlp.{proj}.weight"] = det(
                (f_dim, h_dim), 0.2 * layer
            )
        synth_sd[f"model.layers.{layer}.mlp.down_proj.weight"] = det(
            (h_dim, f_dim), 0.3 * layer
        )

    cfg = QuillanArchConfig(
        text_only=True,
        hidden_dim=h_dim,
        ffn_dim=f_dim,
        vocab_size=vocab_sz,
        num_experts=34,
        num_experts_active=4,
        top_k=4,
        eggroll_rank=16,
        device="cpu",
    )
    model = QuillanRoninSovereign(cfg).eval()
    dims = SourceDims(
        hidden=h_dim, ffn=f_dim, vocab=vocab_sz, num_layers=n_layers, arch="llama"
    )
    adapter = QuillanCrossModelWeightAdapter(num_experts=34, ternarize=False)
    adapted_sd = adapter.adapt_state_dict(
        synth_sd, source_arch="llama", target_model=model, source_dims=dims
    )
    missing, unexpected = model.load_state_dict(adapted_sd, strict=True)
    assert not missing and not unexpected, (
        f"strict load failed: {missing[:5]} / {unexpected[:5]}"
    )

    with torch.no_grad():
        out = model(torch.arange(6, dtype=torch.long).unsqueeze(0))
        logits = out[0] if isinstance(out, tuple) else out
        assert torch.isfinite(logits).all(), "Non-finite forward logits!"

    packed = _pack_ternary_state_dict(model.state_dict())
    int8_keys = [k for k, v in packed.items() if v.dtype == torch.int8]
    assert int8_keys, "No packed int8 ternary weights saved!"
    vals = torch.unique(packed[int8_keys[0]])
    assert all(int(x) in (-1, 0, 1) for x in vals.tolist()), (
        f"Non-ternary values: {vals}"
    )
    print(
        f"QUILLAN_BITNET158_TEST PASS | logits_shape={tuple(logits.shape)} "
        f"packed_int8_tensors={len(int8_keys)} ternary_vals={vals.tolist()}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export QuillanRoninSovereign BitNet 1.58b checkpoint"
    )
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--model", choices=["flagship", "mini"], default="flagship")
    parser.add_argument(
        "--out-name",
        default=None,
        help="write candidate to a NEW filename under exports\\ (never overwrite canonical)",
    )
    parser.add_argument(
        "--merge-embeddings",
        action="store_true",
        help="#956: merge qwen+llama+dragon embeddings by token string into txt_emb/txt_dec",
    )
    args = parser.parse_args()
    if args.self_test:
        run_synthetic_unit_test()
        return
    out = export_quillan_sovereign_bitnet(
        args.model, args.out_name, args.merge_embeddings
    )
    print(
        json.dumps(
            {"model": args.model, "safetensors": str(out), "bytes": out.stat().st_size}
        )
    )


if __name__ == "__main__":
    main()
