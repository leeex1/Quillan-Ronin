#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
REAL self-test for donor -> QuillanRoninSovereign weight transfer.  [C10-CODEWEAVER]

Verifies, against the ACTUAL custom architecture state_dict():
  (a) deterministic tiny donor state_dict is built;
  (b) the adapter runs against that donor + a real target model;
  (c) strict key coverage: load_state_dict(strict=True) -> 0 missing, 0 unexpected;
  (d) >= MIN_DONOR_FRACTION of target parameter elements are derived from donor
      values (no zeros/random for knowledge tensors), and all key knowledge
      groups are donor-derived;
  (e) a forward pass yields finite logits;
  (f) two independent runs are bit-identical (determinism);
  (g) strict mode actually FAILS on an intentionally incomplete state_dict;
  (h) ternary packing preserves gamma and does not re-quantize ternary tensors.

Run:  python -u C:\02_QUILLAN\scripts\test_custom_arch_transfer.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Dict, Tuple

import torch

CORE_MODEL_DIR = Path(r"C:\02_QUILLAN\03 - Training & Model")
sys.path.insert(0, str(CORE_MODEL_DIR))
sys.path.insert(0, r"C:\02_QUILLAN\scripts")

from quillan_v8_saturated import QuillanArchConfig, QuillanRoninSovereign  # noqa: E402
from quillan_weight_adapter import (  # noqa: E402
    SourceDims,
    AdaptationReport,
    QuillanCrossModelWeightAdapter,
    _absmean_ternarize,
    _is_ternary,
)
from export_quillan_sovereign_bitnet import _pack_ternary_state_dict  # noqa: E402

# 34 experts (the real model setting) so the donor-derived MoE tensors dominate
# the fixed rank-256 EGGROLL/LoRA structural tensors: measured donor fraction is
# ~0.38 even at this tiny 64/128 hidden/ffn configuration. At production dims
# (2048/8192) the donor fraction is >0.95. The floor is set from measurement.
MIN_DONOR_FRACTION = 0.30

VOCAB, HIDDEN, FFN, LAYERS, EXPERTS = 256, 64, 128, 3, 34


def _det(shape: Tuple[int, ...], offset: float) -> torch.Tensor:
    """Deterministic, non-random donor tensor (affine ramp) -- no RNG at all."""
    count = math.prod(shape)
    return torch.arange(count, dtype=torch.float32).reshape(shape) * 0.001 + offset


def _build_donor() -> Dict[str, torch.Tensor]:
    """(a) Tiny deterministic Llama-style donor state_dict."""
    sd: Dict[str, torch.Tensor] = {
        "model.embed_tokens.weight": _det((VOCAB, HIDDEN), 0.5),
        "model.norm.weight": _det((HIDDEN,), 2.0),
    }
    for layer in range(LAYERS):
        sd[f"model.layers.{layer}.input_layernorm.weight"] = _det(
            (HIDDEN,), 1.0 + 0.01 * layer
        )
        sd[f"model.layers.{layer}.post_attention_layernorm.weight"] = _det(
            (HIDDEN,), 1.1 + 0.01 * layer
        )
        for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
            sd[f"model.layers.{layer}.self_attn.{proj}.weight"] = _det(
                (HIDDEN, HIDDEN), 0.1 * layer
            )
        for proj in ("gate_proj", "up_proj"):
            sd[f"model.layers.{layer}.mlp.{proj}.weight"] = _det(
                (FFN, HIDDEN), 0.2 * layer
            )
        sd[f"model.layers.{layer}.mlp.down_proj.weight"] = _det(
            (HIDDEN, FFN), 0.3 * layer
        )
    return sd


def _build_target() -> QuillanRoninSovereign:
    cfg = QuillanArchConfig(
        text_only=True,
        hidden_dim=HIDDEN,
        ffn_dim=FFN,
        vocab_size=VOCAB,
        num_experts=EXPERTS,
        num_experts_active=4,
        top_k=4,
        eggroll_rank=16,
        device="cpu",
    )
    return QuillanRoninSovereign(cfg).eval()


def _run_once() -> Tuple[Dict[str, torch.Tensor], AdaptationReport, torch.Tensor]:
    donor = _build_donor()
    target = _build_target()
    dims = SourceDims(
        hidden=HIDDEN, ffn=FFN, vocab=VOCAB, num_layers=LAYERS, arch="llama"
    )
    adapter = QuillanCrossModelWeightAdapter(num_experts=EXPERTS, ternarize=False)
    adapted = adapter.adapt_state_dict(
        donor, source_arch="llama", target_model=target, source_dims=dims
    )

    # (c) strict coverage -- fail loudly on any mismatch.
    missing, unexpected = target.load_state_dict(adapted, strict=True)
    assert not missing and not unexpected, (
        f"strict coverage failed: missing={missing[:5]} unexpected={unexpected[:5]}"
    )
    assert set(adapted) == set(target.state_dict()), (
        "key set differs from authoritative target state_dict"
    )

    report = adapter.last_report
    assert report is not None, "adapter did not publish a provenance report"

    # (d) donor-derived element fraction + explicit knowledge-group coverage.
    assert report.donor_fraction >= MIN_DONOR_FRACTION, (
        f"Only {report.donor_fraction:.3f} of target elements are donor-derived "
        f"(floor {MIN_DONOR_FRACTION})"
    )
    knowledge_keys = {
        "ingestion.txt_emb.weight",
        "txt_dec.weight",
        "moe.w1",
        "moe.wgate",
        "moe.w2",
        "diffusion_core.couil_attn.q_proj.weight",
        "diffusion_core.couil_attn.k_proj.weight",
        "diffusion_core.couil_attn.v_proj.weight",
        "diffusion_core.couil_attn.o_proj.weight",
        "diffusion_core.ffn.0.weight",
        "diffusion_core.ffn.2.weight",
        "decomposition.vectors.Language.weight",
        "quillan_finalizer.weight",
    }
    missing_knowledge = knowledge_keys - set(report.donor_keys)
    assert not missing_knowledge, (
        f"Knowledge tensors not donor-derived: {sorted(missing_knowledge)}"
    )

    # Every donor-derived tensor must actually contain donor magnitudes, i.e.
    # must not be all-zero (the naive-export failure mode this test guards).
    for key in report.donor_keys:
        assert bool(adapted[key].abs().sum().item() > 0.0), (
            f"Donor-derived tensor is all zeros: {key}"
        )

    # (e) finite forward logits.
    with torch.no_grad():
        out = target(torch.arange(6, dtype=torch.long).unsqueeze(0))
        logits = out[0] if isinstance(out, tuple) else out
    assert torch.isfinite(logits).all(), "forward pass produced non-finite logits"
    assert logits.ndim == 3, f"unexpected logits shape {tuple(logits.shape)}"

    # (g) strict mode must REJECT an incomplete state_dict.
    broken = dict(adapted)
    broken.pop("moe.w1")
    try:
        _build_target().load_state_dict(broken, strict=True)
    except RuntimeError:
        pass  # expected: strict mode detects the missing key
    else:
        raise AssertionError(
            "strict=True silently accepted a state_dict missing moe.w1"
        )

    # (h) ternary scale propagation: gamma preserved, ternary not re-quantized.
    dense = _det((32, 32), 0.7)
    q_dense, gamma_dense = _absmean_ternarize(dense)
    assert _is_ternary(q_dense), "ternarized dense tensor is not in {-1,0,1}"
    assert gamma_dense.item() > 0.0, "non-positive ternary scale"

    # Exactly-scaled ternary probe: gamma must round-trip the original values.
    base = torch.tensor([[1.0, -1.0, 1.0, -1.0], [-1.0, 1.0, -1.0, 1.0]])
    probe = base * 0.3
    q, gamma = _absmean_ternarize(probe)
    assert _is_ternary(q), "ternarized probe is not in {-1,0,1}"
    assert torch.allclose(probe, q * gamma, atol=1e-6), (
        "scale reconstruction not faithful"
    )
    assert torch.allclose(gamma, torch.tensor(0.3), atol=1e-6), (
        "gamma != least-squares scale 0.3"
    )

    # Demonstrate the guarded bug: re-quantizing an ALREADY ternary tensor with
    # a fresh scale yields gamma2=1.0, discarding the original 0.3 scale.
    _, gamma_requant = _absmean_ternarize(q)
    assert not torch.allclose(gamma_requant, gamma), (
        "expected fresh-scale re-quant to differ"
    )
    packed = _pack_ternary_state_dict(
        {"probe.weight": q, "probe.weight.bitnet_scale": gamma}
    )
    assert packed["probe.weight"].dtype == torch.int8, (
        "ternary weight not packed as int8"
    )
    assert torch.equal(packed["probe.weight.bitnet_scale"], gamma), (
        "gamma was recomputed on already-ternary tensor"
    )

    return adapted, report, logits


def main() -> None:
    torch.manual_seed(1234)
    first, report, logits_first = _run_once()
    # (f) determinism: a second independent run must be bit-identical.
    second, report2, logits_second = _run_once()
    assert set(first) == set(second), "non-deterministic key set"
    for key in first:
        assert torch.equal(first[key], second[key]), f"non-deterministic tensor: {key}"
    assert torch.equal(logits_first, logits_second), "non-deterministic logits"
    assert report.donor_keys == report2.donor_keys, "non-deterministic provenance"

    print(
        "CUSTOM_ARCH_TRANSFER_TEST PASS | "
        f"keys={len(first)} donor_keys={len(report.donor_keys)} "
        f"structural_keys={len(report.structural_keys)} "
        f"donor_fraction={report.donor_fraction:.4f} "
        f"donor_elems={report.donor_elements} total_elems={report.total_elements} "
        f"logits={tuple(logits_first.shape)} deterministic=True strict=True"
    )


if __name__ == "__main__":
    main()
