#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""Verification harness for the 815M 'Mighty Mouse' custom-arch checkpoint.

Non-mutating: loads the exported packed safetensors, dequantizes into a freshly
built QuillanRoninSovereign, checks strict key coverage, parameter count, finite
forward pass, donor-derived fraction (via the same adapter used at export time),
file size and SHA256. Prints a single JSON summary on stdout.
"""

from __future__ import annotations

import gc
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict

import safetensors.torch
import torch
from transformers import AutoModelForCausalLM

CORE_MODEL_DIR = Path(r"C:\02_QUILLAN\03 - Training & Model")
sys.path.insert(0, r"C:\02_QUILLAN\scripts")
sys.path.insert(0, str(CORE_MODEL_DIR))

from quillan_v8_saturated import QuillanArchConfig, QuillanRoninSovereign  # noqa: E402
from quillan_weight_adapter import (  # noqa: E402
    SourceDims,
    QuillanCrossModelWeightAdapter,
)
from export_quillan_sovereign_bitnet import _source_dims_from_config  # noqa: E402

MINI_SNAP = Path(
    r"C:\Users\Admin\.cache\huggingface\hub\models--Qwen--Qwen2.5-0.5B-Instruct"
    r"\snapshots\7ae557604adf67be50417f59c2c2f167def9a775"
)
CKPT = Path(r"C:\02_QUILLAN\exports\quillan-mini-sovereign-bitnet158.safetensors")

H, F, V, L = 896, 4864, 151936, 24


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    result: Dict[str, Any] = {}
    result["checkpoint"] = str(CKPT)
    result["exists"] = CKPT.exists()
    if not CKPT.exists():
        print(json.dumps(result))
        return
    result["bytes"] = CKPT.stat().st_size
    result["sha256"] = _sha256(CKPT)

    packed = safetensors.torch.load_file(str(CKPT))
    scales = {
        k[: -len(".bitnet_scale")]: v
        for k, v in packed.items()
        if k.endswith(".bitnet_scale")
    }
    raw = {k: v for k, v in packed.items() if not k.endswith(".bitnet_scale")}
    result["packed_tensor_count"] = len(packed)
    result["int8_tensors"] = sum(1 for v in raw.values() if v.dtype == torch.int8)
    result["bitnet_scale_tensors"] = len(scales)

    sd: Dict[str, torch.Tensor] = {}
    for k, v in raw.items():
        if v.dtype == torch.int8:
            sd[k] = v.float() * scales[k].float()
        else:
            sd[k] = v.float()

    cfg = QuillanArchConfig(
        text_only=True,
        multimodal=False,
        hidden_dim=H,
        ffn_dim=F,
        vocab_size=V,
        num_experts=34,
        num_experts_active=4,
        top_k=4,
        use_lora=True,
        device="cpu",
        eggroll_rank=64,
    )
    q = QuillanRoninSovereign(cfg).eval()
    expected_keys = set(q.state_dict().keys())
    result["model_state_dict_keys"] = len(expected_keys)
    result["checkpoint_base_keys"] = len(sd)
    result["key_set_match"] = set(sd) == expected_keys
    missing, unexpected = q.load_state_dict(sd, strict=True)
    result["strict_missing"] = len(missing)
    result["strict_unexpected"] = len(unexpected)

    result["total_params"] = sum(p.numel() for p in q.parameters())
    result["total_params_from_ckpt"] = sum(v.numel() for v in sd.values())

    with torch.no_grad():
        ids = torch.arange(8, dtype=torch.long).unsqueeze(0) % 1000
        out = q(ids)
    logits = out[0] if isinstance(out, tuple) else out
    result["logits_shape"] = list(logits.shape)
    result["finite_forward"] = bool(torch.isfinite(logits).all())

    del sd, packed, raw, scales
    gc.collect()

    # Re-derive the authoritative donor fraction using the exact export-time adapter.
    hf = AutoModelForCausalLM.from_pretrained(
        str(MINI_SNAP), local_files_only=True, dtype=torch.float32
    )
    src_sd = {k: v.detach().cpu().float() for k, v in hf.state_dict().items()}
    dims = _source_dims_from_config(hf.config, "qwen2")
    del hf
    gc.collect()
    adapter = QuillanCrossModelWeightAdapter(
        target_hidden_dim=H,
        target_ffn_dim=F,
        target_vocab_size=V,
        num_experts=34,
        expert_rank=8,
        ternarize=False,
    )
    adapter.adapt_state_dict(
        src_sd, source_arch=dims.arch, target_model=q, source_dims=dims
    )
    rep = adapter.last_report
    assert rep is not None
    result["donor_fraction"] = rep.donor_fraction
    result["donor_keys"] = len(rep.donor_keys)
    result["structural_keys"] = len(rep.structural_keys)
    result["strict_key_count"] = len(rep.donor_keys) + len(rep.structural_keys)
    result["source_dims"] = {
        "hidden": dims.hidden,
        "ffn": dims.ffn,
        "vocab": dims.vocab,
        "layers": dims.num_layers,
        "arch": dims.arch,
    }
    print("MIGHTY_MOUSE_VERIFY=" + json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
