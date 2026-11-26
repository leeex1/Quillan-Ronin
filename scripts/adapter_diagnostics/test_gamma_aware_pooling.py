#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
REAL diagnostic for gamma-aware donor pooling (#925/#922 handoff, directives #930/#931/#933).

Loads tiny synthetic donors -- one FLOAT, one ALREADY TERNARY -- and asserts:
  1. gamma is preserved (non-zero, sensible) and never discarded;
  2. merged q is strictly in {-1, 0, +1};
  3. re-quantization is idempotent (ternarizing an already-ternary tensor with
     its supplied gamma is a no-op on BOTH code and scale);
  4. UNIQUE donor knowledge survives pooling; OVERLAPPING knowledge is
     normalized to one scale (#933);
  5. the emitted weight round-trips exactly through the REAL target
     ``_weight_quant`` (whole-matrix absmean, eps=0.01) -- no scale mismatch;
  6. adapter wiring: auto/forced BitNet path is active, legacy flag restores
     the old raw-float mean, and the run is deterministic.

Run:  python -u scripts\adapter_diagnostics\test_gamma_aware_pooling.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Tuple

import torch

REPO = Path(r"C:\02_QUILLAN")
MODEL_DIR = REPO / "03 - Training & Model"
SCRIPTS = REPO / "scripts"
for p in (str(MODEL_DIR), str(SCRIPTS)):
    if p not in sys.path:
        sys.path.insert(0, p)

from quillan_v8_saturated import _weight_quant  # noqa: E402
from quillan_weight_adapter import (  # noqa: E402
    QuillanCrossModelWeightAdapter,
    SourceDims,
    _absmean_ternarize,
    _is_ternary,
    effective_gamma,
    emit_bitnet_native,
    pool_ternary_pairs,
    to_ternary_pair,
)
from test_custom_arch_transfer import (  # noqa: E402  -- reuse existing tiny fixtures
    EXPERTS,
    HIDDEN,
    LAYERS,
    VOCAB,
    _build_donor,
    _build_target,
)

PASSED = 0


def check(name: str, ok: bool, detail: str = "") -> None:
    global PASSED
    assert ok, f"FAIL {name} :: {detail}"
    PASSED += 1
    print(f"PASS {name}" + (f" :: {detail}" if detail else ""))


def _synthetic_donors() -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One float donor, one already-ternary donor with an explicit gamma."""
    float_donor = torch.zeros((4, 8), dtype=torch.float32)
    float_donor[0, 0] = 1.0
    float_donor[0, 1] = 0.5  # unique to float donor
    float_donor[1, 2] = -0.8  # unique to float donor
    float_donor[2, 3] = 0.3  # unique to float donor
    float_donor[3, 0] = 0.6
    float_donor[3, 1] = -0.6  # overlap: both donors carry these

    ternary_donor = torch.zeros((4, 8), dtype=torch.float32)
    ternary_donor[3, 0] = 1.0
    ternary_donor[3, 1] = -1.0  # overlap, agrees with float donor
    ternary_donor[1, 5] = -1.0  # unique to already-ternary donor
    gamma_ternary = torch.tensor(0.30)
    return float_donor, ternary_donor, gamma_ternary


def test_primitives() -> None:
    w_float, q_tern, g_tern = _synthetic_donors()

    # (3) idempotency: an already-ternary tensor with its gamma is a NO-OP.
    q_back, g_back = to_ternary_pair(q_tern, gamma=g_tern)
    check(
        "idempotent_gamma_noop",
        torch.equal(q_back, q_tern) and torch.equal(g_back, g_tern),
        f"q_equal={torch.equal(q_back, q_tern)} gamma={g_back.item():.6g}",
    )

    # Donor-side conversion: float uses the reused per-channel absmean convention.
    q_float, g_float = to_ternary_pair(w_float, group_dim=-1)
    check(
        "float_donor_ternary",
        _is_ternary(q_float) and bool((g_float > 0).all()),
        f"gamma_per_channel={[round(float(x), 5) for x in g_float.flatten().tolist()]}",
    )

    pairs = [(q_float, g_float), (q_back, g_back)]
    q_pool, gamma_pool = pool_ternary_pairs(pairs)

    # (1)(2) gamma preserved, codes strictly ternary.
    check(
        "gamma_preserved_nonzero_sensible",
        bool(gamma_pool > 0)
        and bool(gamma_pool <= max(float(g_float.max()), float(g_tern))),
        f"gamma_pool={gamma_pool.item():.6g}",
    )
    check(
        "merged_q_ternary",
        _is_ternary(q_pool),
        f"unique={torch.unique(q_pool).tolist()}",
    )

    # (4) UNIQUE knowledge: nonzero where only ONE donor has a code.
    unique_float = (q_pool[0, 0].item() != 0) and (q_pool[2, 3].item() != 0)
    unique_tern = q_pool[1, 5].item() != 0
    check(
        "unique_knowledge_preserved",
        unique_float and unique_tern,
        f"float_only={unique_float} ternary_only={unique_tern} "
        f"q[0,0]={q_pool[0, 0].item():.0f} q[2,3]={q_pool[2, 3].item():.0f} "
        f"q[1,5]={q_pool[1, 5].item():.0f}",
    )
    check(
        "overlap_normalized",
        q_pool[3, 0].item() == 1 and q_pool[3, 1].item() == -1,
        f"q[3,0]={q_pool[3, 0].item():.0f} q[3,1]={q_pool[3, 1].item():.0f}",
    )

    # (5) emission: absmean == runtime scale; REAL _weight_quant is idempotent.
    eff = emit_bitnet_native(q_pool, gamma_pool)
    emitted_gamma = effective_gamma(gamma_pool)
    mean_abs = eff.abs().mean()
    check(
        "emitted_absmean_matches_runtime_scale",
        torch.allclose(mean_abs, emitted_gamma, atol=1e-6),
        f"mean|w|={mean_abs.item():.6g} gamma_emit={emitted_gamma.item():.6g}",
    )
    # The FORWARD value must equal q*gamma (full scale).  The stored tensor is
    # q*gamma/rho; this quantizer shrinks any sparse stored tensor by rho, so
    # literal `_weight_quant(w)==w` is impossible for rho<1 (backtrack #932).
    with torch.no_grad():
        fwd = _weight_quant(eff, eps=0.01)
    intended = q_pool * emitted_gamma
    check(
        "runtime_weight_quant_roundtrip",
        torch.allclose(fwd, intended, atol=1e-6),
        f"max_abs_diff={float((fwd - intended).abs().max()):.3e} "
        f"rho={float(q_pool.abs().mean()):.5f}",
    )
    q_rec, g_rec = _absmean_ternarize(eff)
    check(
        "exporter_recovers_q_gamma",
        torch.equal(q_rec, q_pool) and torch.allclose(g_rec, emitted_gamma, atol=1e-6),
        f"q_equal={torch.equal(q_rec, q_pool)} gamma={g_rec.item():.6g}",
    )

    # eps-floor guard: a sub-floor gamma still round-trips with NO mismatch.
    tiny_q = torch.tensor([[1.0, 0.0, -1.0, 0.0], [0.0, 1.0, 0.0, -1.0]])
    tiny_g = torch.tensor(1e-4)
    eff_small = emit_bitnet_native(tiny_q, tiny_g)
    with torch.no_grad():
        fwd_small = _weight_quant(eff_small, eps=0.01)
    check(
        "eps_floor_no_mismatch",
        torch.allclose(fwd_small, tiny_q * 0.01, atol=1e-6)
        and torch.allclose(eff_small.abs().mean(), torch.tensor(0.01), atol=1e-6),
        f"mean|w|={eff_small.abs().mean().item():.6g} "
        f"forward_matches_q*floor={torch.allclose(fwd_small, tiny_q * 0.01, atol=1e-6)}",
    )


def test_adapter_wiring() -> None:
    donor = _build_donor()
    dims = SourceDims(
        hidden=HIDDEN, ffn=128, vocab=VOCAB, num_layers=LAYERS, arch="llama"
    )

    target = _build_target()
    adapter = QuillanCrossModelWeightAdapter(
        num_experts=EXPERTS, bitnet_native_pool=True
    )
    adapted = adapter.adapt_state_dict(
        donor, source_arch="llama", target_model=target, source_dims=dims
    )
    missing, unexpected = target.load_state_dict(adapted, strict=True)
    check(
        "bitnet_strict_load",
        not missing and not unexpected,
        f"keys={len(adapted)} missing={len(missing)} unexpected={len(unexpected)}",
    )
    attn = adapted["diffusion_core.couil_attn.q_proj.weight"]
    check(
        "bitnet_pooled_tensor_is_ternary_scaled",
        len(torch.unique(attn)) <= 3,
        f"shape={tuple(attn.shape)} unique={torch.unique(attn).tolist()[:6]}",
    )
    check(
        "bitnet_gamma_recorded",
        bool(adapter.last_report and adapter.last_report.scales)
        and all(float(v) > 0 for v in adapter.last_report.scales.values()),
        f"scale_keys={len(adapter.last_report.scales) if adapter.last_report else 0}",
    )
    # forward = q*gamma = attn*rho (storage magnitude is gamma/rho).
    rho_attn = (attn != 0).float().mean()
    with torch.no_grad():
        check(
            "bitnet_attention_roundtrip",
            torch.allclose(_weight_quant(attn, eps=0.01), attn * rho_attn, atol=1e-5),
            f"rho={rho_attn.item():.5f}",
        )

    # Legacy flag restores exactly the old raw-float mean (no gamma records).
    legacy = QuillanCrossModelWeightAdapter(
        num_experts=EXPERTS, bitnet_native_pool=False
    )
    legacy_sd = legacy.adapt_state_dict(
        donor, source_arch="llama", target_model=_build_target(), source_dims=dims
    )
    legacy_attn = legacy_sd["diffusion_core.couil_attn.q_proj.weight"]
    check(
        "legacy_flag_restores_float_mean",
        len(torch.unique(legacy_attn)) > 3
        and bool(legacy.last_report and not legacy.last_report.scales),
        f"unique_count={len(torch.unique(legacy_attn))}",
    )

    # Determinism of the new path.
    again = QuillanCrossModelWeightAdapter(num_experts=EXPERTS, bitnet_native_pool=True)
    again_sd = again.adapt_state_dict(
        donor, source_arch="llama", target_model=_build_target(), source_dims=dims
    )
    check(
        "gamma_pool_deterministic",
        all(torch.equal(adapted[k], again_sd[k]) for k in adapted),
    )


def main() -> None:
    torch.manual_seed(1234)
    test_primitives()
    test_adapter_wiring()
    print(f"GAMMA_AWARE_POOLING_TEST PASS | checks={PASSED}")


if __name__ == "__main__":
    main()
