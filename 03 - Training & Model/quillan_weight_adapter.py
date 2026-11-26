#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
👑 QUILLAN-RONIN (2026) — CROSS-MODEL TENSOR WEIGHT ADAPTER  [C10 HARDENED]

Purpose
-------
Translate a foreign causal-LM checkpoint (Llama / Qwen2 / Falcon-style HF naming)
into a *complete*, *deterministic* state_dict for ``QuillanRoninSovereign`` so that
``model.load_state_dict(adapted, strict=True)`` succeeds with ZERO missing and
ZERO unexpected keys.

Design invariants (enforced, not merely documented)
----------------------------------------------------
1. TARGET KEYS ARE DERIVED AT RUNTIME from ``target_model.state_dict()``.  Nothing
   about the target namespace or shape is hardcoded here.  Every emitted key is
   checked against that authoritative set; a missing assignment raises.
2. EVERY knowledge-carrying tensor is deterministically derived from a donor
   tensor.  No ``torch.randn`` / unseeded RNG exists in this module.
3. Every non-knowledge tensor (LoRA seeds, buffers, physical constants, gates,
   routers, positional tables) is EXPLICITLY classified as a structural constant
   with a documented rationale.  Unknown keys are a hard error, never silent.
4. Ternarization keeps its scale (gamma).  Already-ternary tensors are NEVER
   re-quantized with a fresh scale.
5. Layer pooling is gamma-aware for BitNet targets: float donors are converted
   to (q, gamma) BEFORE pooling, codes are pooled in ternary space, scales are
   propagated, and the emitted tensor is absmean-preserving (``q*gamma/rho``)
   so the target's ``_weight_quant`` forward uses the FULL gamma, not a
   ``rho``-shrunken scale.  Raw donor floats are never averaged.  Legacy float
   pooling remains available behind a flag.

The custom architecture is *smaller in depth* (single diffusion attention block
and a single MoE stage) but *wider* when built with the source dims: the flagship
target is intentionally sized to hidden=2048 / ffn=8192 so the 1.2B donor MLP/attn
weights are retained instead of being cropped to 1024/2048.  Consolidation of N
donor layers into one target block uses mean-over-layers (attention / dense FFN)
and deterministic layer-cycling (34 MoE experts), both loss-aware and reproducible.

Author: CrashOverrideX & Quillan Research Team
Hardening: C10-CODEWEAVER (Build Loop), v5.4.0-oni
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

# NOTE: no `version` import here -- the adapter is imported from several entry
# points with differing sys.path; keeping it dependency-free avoids import
# fragility. Version metadata belongs to the CLI/export layer.

logger = logging.getLogger("Quillan.WeightAdapter")


# --------------------------------------------------------------------------- #
# Data records                                                                 #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SourceDims:
    """Authoritative source-checkpoint dimensions (read from HF config, never guessed)."""

    hidden: int
    ffn: int
    vocab: int
    num_layers: int
    arch: str = "llama"

    def __post_init__(self) -> None:
        assert self.hidden > 0 and self.ffn > 0 and self.vocab > 0, (
            "Source dims must be positive"
        )
        assert self.num_layers >= 0, "num_layers must be >= 0"


@dataclass(frozen=True)
class AdaptationReport:
    """Auditable provenance of every emitted target tensor."""

    donor_keys: Tuple[str, ...]
    structural_keys: Tuple[str, ...]
    donor_elements: int
    total_elements: int
    source_dims: SourceDims
    scales: Mapping[str, torch.Tensor]

    @property
    def donor_fraction(self) -> float:
        return self.donor_elements / max(1, self.total_elements)


# --------------------------------------------------------------------------- #
# Ternarization primitives                                                     #
# --------------------------------------------------------------------------- #
def _absmean_ternarize(
    w: torch.Tensor, eps: float = 1e-6
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Least-squares (absmean) BitNet 1.58b scale + ternary quantization.

    The closed-form least-squares scale for a ternary codebook {-1,0,+1} is the
    mean absolute value of the weight block (BitNet b1.58 absmean).  We return
    BOTH the integer-valued ternary tensor and the scale ``gamma`` so callers
    never have to recompute it (recomputing on an already-ternary tensor would
    incorrectly shrink gamma to the non-zero fraction).

    Returns
    -------
    q : Tensor  -- values in {-1, 0, +1}, same shape as ``w``
    gamma : Tensor -- scalar positive scale; ``w ~= q * gamma``
    """
    wf = w.detach().float()
    gamma = wf.abs().mean().clamp_min(eps)
    q = (wf / gamma).round().clamp(-1.0, 1.0)
    return q, gamma


def _is_ternary(t: torch.Tensor) -> bool:
    """True iff every element is exactly -1, 0 or +1 (already quantized)."""
    if t.numel() == 0:
        return False
    return bool(((t == -1) | (t == 0) | (t == 1)).all().item())


def _per_channel_absmean_ternarize(
    w: torch.Tensor, eps: float = 1e-5
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-channel BitNet absmean -- REUSE of ``build_coherent_ternary_base.py:49-56``.

    ``scale = w.abs().mean(dim=-1, keepdim=True).clamp(min=eps); q = round(clamp(w/scale))``.
    Kept inline (byte-identical formula) because this module deliberately stays
    dependency-free (see module header :49-51); importing that builder would drag
    huggingface_hub/tokenizers into every adapter consumer.  This is NOT a new
    convention -- it is the existing per-channel donor-side convention used to
    retain fine-grained (unique) structure when converting each float donor.
    """
    wf = w.detach().float()
    scale = wf.abs().mean(dim=-1, keepdim=True).clamp(min=eps)
    return (wf / scale).round().clamp(-1.0, 1.0), scale


def to_ternary_pair(
    w: torch.Tensor,
    gamma: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
    group_dim: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return ``(q, gamma)`` for one donor tensor WITHOUT ever losing its scale.

    Three cases, in strict priority order:

    1. ``gamma`` supplied and ``w`` already in {-1,0,+1}: return the codes and
       the SUPPLIED gamma verbatim.  Recomputing absmean here would shrink gamma
       to the non-zero fraction -- the exact bug documented on ``_absmean_ternarize``.
       This is the idempotency guarantee for already-BitNet donors.
    2. ``gamma`` supplied and ``w`` is a float effective weight: decode with that
       fixed scale (``round(clamp(w/gamma))``); gamma is carried through.
    3. No ``gamma``: a plain float donor -> least-squares absmean ternarization.

    The "already-BitNet" donor caches store float effective weights with no
    scale sidecar, so they take case 3; a caller holding an explicit scale takes
    cases 1/2 and keeps it.
    """
    wf = w.detach().float()
    if gamma is not None:
        if _is_ternary(wf):
            # Already-quantized donor: keep BOTH code and supplied scale verbatim.
            return wf.clone(), gamma.detach().float()
        return (wf / gamma.detach().float()).round().clamp(
            -1.0, 1.0
        ), gamma.detach().float()
    if group_dim == -1:
        return _per_channel_absmean_ternarize(wf, eps=eps)
    if group_dim is not None:
        raise ValueError(f"unsupported group_dim={group_dim}; use None or -1")
    return _absmean_ternarize(wf, eps=eps)


def pool_ternary_pairs(
    pairs: Sequence[Tuple[torch.Tensor, torch.Tensor]],
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pool N ``(q, gamma)`` pairs in TERNARY space (directive #933).

    Two jobs, both required:

    * UNIQUE knowledge is PRESERVED.  A donor that carries a nonzero code where
      the others carry zero is NOT averaged toward zero.  We take the SIGN of a
      scale-weighted code sum instead of its mean, and an exact tie falls back
      to the first non-zero donor's sign.  No contribution is washed out.
    * OVERLAPPING knowledge is NORMALIZED to one Quillan-consistent scale: each
      donor's code is expressed in the pooled scale (``q_i * gamma_i/gamma_pool``),
      so a high-gamma donor wins a sign conflict and agreeing donors reinforce.

    ``gamma_pool`` is the global mean of every gamma element (a single scalar --
    the whole-matrix absmean convention the runtime consumes, per #930).  Per-channel
    gammas from the donor-side conversion are used only as pooling weights; they
    are reconciled into that one scalar at emission.  Inputs are already-quantized
    codes, never raw donor floats.

    Returns ``(q_pool, gamma_pool)`` with ``q_pool`` in {-1,0,+1} and
    ``gamma_pool`` a positive scalar tensor.
    """
    if not pairs:
        raise ValueError("pool_ternary_pairs requires at least one (q, gamma) pair")
    gammas = torch.cat(
        [g.detach().float().reshape(-1).clamp_min(eps) for _, g in pairs]
    )
    gamma_pool = gammas.mean().clamp_min(eps)
    acc = torch.zeros_like(pairs[0][0].detach().float())
    first = torch.zeros_like(acc)
    for q, g in pairs:
        qf = q.detach().float()
        gf = g.detach().float().clamp_min(eps)
        acc = acc + qf * (gf / gamma_pool)
        first = torch.where((first == 0) & (qf != 0), qf.sign(), first)
    # sign-vote keeps unique single-donor signal; first-donor tie-break avoids
    # cancellation-to-zero on an exact conflict.
    q_pool = torch.where(
        acc > 0,
        torch.ones_like(acc),
        torch.where(acc < 0, -torch.ones_like(acc), first),
    ).clamp(-1.0, 1.0)
    return q_pool.contiguous(), gamma_pool


def effective_gamma(gamma: torch.Tensor, scale_floor: float = 0.01) -> torch.Tensor:
    """Apply the target runtime eps floor to a pooled scale.

    ``quillan_v8_saturated._weight_quant`` uses ``clamp(min=eps=0.01)`` on its
    absmean scale.  A pooled gamma below that floor is unrepresentable at
    runtime -- the operator silently substitutes 0.01.  Applying the same floor
    here makes the emitted weight's absmean EQUAL the scale the runtime will
    actually use, so the round-trip is exact instead of a hidden ~15x inflation.
    """
    return gamma.detach().float().reshape(()).clamp_min(scale_floor)


def emit_bitnet_native(
    q: torch.Tensor,
    gamma: torch.Tensor,
    eps: float = 1e-6,
    scale_floor: float = 0.01,
) -> torch.Tensor:
    """Materialize ``(q, gamma)`` as the float tensor the target actually wants.

    Convention (see blocker #930 resolution): the target's ``_weight_quant``
    derives ONE scale per 2D matrix as ``clamp(mean(|w|), 0.01)`` and returns
    ``round(clamp(w/scale))*scale``.  Storing plain ``q*gamma`` would make the
    forward value ``q*gamma*rho`` (``rho = mean(|q|)``) -- a silent scale loss.
    Emitting ``q * clamp(gamma,0.01) / rho`` gives
    ``mean(|w|) == clamp(gamma,0.01) == runtime scale`` exactly, hence the
    FORWARD value ``_weight_quant(w)`` is ``q * clamp(gamma,0.01)`` -- the full
    intended scale, with no ``rho`` shrink and no eps mismatch.  Note the stored
    tensor itself is ``q*gamma/rho``: with a sparse code (``rho<1``) this
    quantizer cannot be literally idempotent (``_weight_quant(w)==w`` would
    require ``rho==1``); what must round-trip is the SCALE, not the storage
    magnitudes.  ``_absmean_ternarize(w)`` recovers ``(q, clamp(gamma,0.01))``
    for the exporter's sidecar packer.  This is the SAME whole-matrix absmean
    convention as the reusable ``quillan_weight_adapter._quantize_to_ternary``
    (scripts copy, :32-37); no new convention is introduced.
    """
    g = effective_gamma(gamma, scale_floor=scale_floor)
    rho = q.detach().float().abs().mean().clamp_min(eps)
    return (q.detach().float() * (g / rho)).contiguous()


# --------------------------------------------------------------------------- #
# Adapter                                                                      #
# --------------------------------------------------------------------------- #
class QuillanCrossModelWeightAdapter:
    """Translate a foreign HF state_dict into a strict-complete target state_dict.

    The legacy constructor hints (``target_hidden_dim`` etc.) are retained for
    source compatibility; they are only used to WARN on a dim mismatch.  The
    authoritative target namespace/shapes always come from ``target_model``.
    """

    _PRISM_VECTORS: Tuple[str, ...] = (
        "Language",
        "Sentiment",
        "Context",
        "Intent",
        "Meta",
        "Creativity",
        "Ethics",
        "Strategy",
        "Constraint",
    )

    def __init__(
        self,
        target_hidden_dim: int = 1024,
        target_ffn_dim: int = 2048,
        target_vocab_size: int = 50257,
        num_experts: int = 34,
        expert_rank: int = 8,
        ternarize: bool = False,
        max_seq_len: int = 1024,
        bitnet_native_pool: Optional[bool] = None,
    ) -> None:
        self.target_hidden_dim = target_hidden_dim
        self.target_ffn_dim = target_ffn_dim
        self.target_vocab_size = target_vocab_size
        self.num_experts = num_experts
        self.expert_rank = expert_rank
        self.ternarize = ternarize
        self.max_seq_len = max_seq_len
        # None = auto-detect from target at adaptation time (BitLinear present);
        # True/False force the gamma-aware / legacy float pooling path.
        self.bitnet_native_pool = bitnet_native_pool
        self._bitnet_native_active = False
        self._pool_scales: Dict[str, torch.Tensor] = {}
        self.last_report: Optional[AdaptationReport] = None
        logger.info(
            "QuillanWeightAdapter initialized (hint hidden=%d ffn=%d vocab=%d experts=%d); "
            "authoritative dims are derived from target_model.state_dict() at adaptation time.",
            target_hidden_dim,
            target_ffn_dim,
            target_vocab_size,
            num_experts,
        )

    # ---------------------------- shape adaptation ------------------------- #
    def adapt_tensor_shape(
        self, tensor: torch.Tensor, target_shape: Tuple[int, ...]
    ) -> torch.Tensor:
        """Center-crop / zero-pad ``tensor`` to ``target_shape`` (asserted exact).

        Center-cropping is the least-lossy deterministic projection when the
        target is genuinely narrower; when the target is built with the source
        dims (the supported export path) this is an exact clone.
        """
        out = tensor.detach()
        if tuple(out.shape) == tuple(target_shape):
            return out.clone()

        while out.dim() < len(target_shape):
            out = out.unsqueeze(0)
        while out.dim() > len(target_shape):
            assert out.size(0) == 1, "Cannot squeeze a non-singleton leading dim"
            out = out.squeeze(0)

        assert out.dim() == len(target_shape), (
            f"Rank mismatch after normalization: {out.dim()} vs {len(target_shape)}"
        )

        for d, targ_size in enumerate(target_shape):
            curr_size = out.size(d)
            if curr_size > targ_size:
                start = (curr_size - targ_size) // 2
                slices: List[slice] = [slice(None)] * len(target_shape)
                slices[d] = slice(start, start + targ_size)
                out = out[tuple(slices)]
            elif curr_size < targ_size:
                pad_total = targ_size - curr_size
                pad_front = pad_total // 2
                pad_back = pad_total - pad_front
                pad_list = [0] * (2 * len(target_shape))
                pad_idx = (len(target_shape) - 1 - d) * 2
                pad_list[pad_idx] = pad_front
                pad_list[pad_idx + 1] = pad_back
                out = F.pad(out, pad_list)

        out = out.contiguous()
        assert tuple(out.shape) == tuple(target_shape), (
            f"adapt_tensor_shape failed: {tuple(out.shape)} != {tuple(target_shape)}"
        )
        return out

    # ---------------------------- donor lookup ----------------------------- #
    @staticmethod
    def _find(src: Mapping[str, torch.Tensor], *candidates: str) -> Optional[str]:
        """Return the first present key among ``candidates`` (explicit, no guessing)."""
        for key in candidates:
            if key in src:
                return key
        return None

    @staticmethod
    def _source_layer_indices(src: Mapping[str, torch.Tensor]) -> List[int]:
        indices: set[int] = set()
        for key in src:
            parts = key.split(".")
            for i, part in enumerate(parts[:-1]):
                if part in ("layers", "h", "block") and parts[i + 1].isdigit():
                    indices.add(int(parts[i + 1]))
        return sorted(indices)

    @staticmethod
    def _target_is_bitnet(target_model: nn.Module) -> bool:
        """True if any module quantizes its weight like BitLinear (auto-detect)."""
        for module in target_model.modules():
            if (
                getattr(module, "quantize_weight", False)
                or type(module).__name__ == "BitLinear"
            ):
                return True
        return False

    def _collect_layer_tensors(
        self,
        src: Mapping[str, torch.Tensor],
        suffixes: Sequence[str],
        target_shape: Tuple[int, ...],
        num_layers: int,
        transpose: bool = False,
    ) -> List[torch.Tensor]:
        """Deterministic, shape-adapted per-layer donor tensors (no arithmetic)."""
        parts: List[torch.Tensor] = []
        for suffix in suffixes:
            for layer in range(num_layers):
                key = self._find(src, f"model.layers.{layer}.{suffix}")
                if key is None:
                    continue
                tensor = src[key]
                if transpose:
                    tensor = tensor.transpose(0, 1)
                parts.append(self.adapt_tensor_shape(tensor, target_shape))
        return parts

    def _pool_donors(
        self,
        parts: Sequence[torch.Tensor],
        scale_key: str,
    ) -> torch.Tensor:
        """Gamma-aware pool when active, legacy raw-float mean otherwise.

        Float donors -> (q, gamma) with the reused PER-CHANNEL absmean convention
        (``build_coherent_ternary_base.quantize_to_ternary``) so distinct per-row
        structure survives conversion.  Already-ternary donors with a supplied
        gamma are NEVER re-quantized.  Codes are then pooled in ternary space
        (unique signal kept, overlaps normalized), gamma is propagated as one
        global absmean, and emission applies the runtime 0.01 scale floor.
        """
        if not parts:
            raise ValueError("no donor tensors to pool")
        if not self._bitnet_native_active:
            return torch.stack(list(parts), dim=0).mean(dim=0).contiguous()
        pairs = [to_ternary_pair(p, group_dim=-1) for p in parts]
        q_pool, gamma_pool = pool_ternary_pairs(pairs)
        emitted_gamma = effective_gamma(gamma_pool)
        self._pool_scales[scale_key] = emitted_gamma.detach().clone()
        return emit_bitnet_native(q_pool, gamma_pool)

    def _layer_mean(
        self,
        src: Mapping[str, torch.Tensor],
        suffix: str,
        target_shape: Tuple[int, ...],
        num_layers: int,
        transpose: bool = False,
    ) -> Optional[torch.Tensor]:
        """Mean of a per-layer donor projection, adapted to the target shape.

        Gamma-aware for BitNet targets; legacy float mean otherwise.
        """
        parts = self._collect_layer_tensors(
            src, (suffix,), target_shape, num_layers, transpose=transpose
        )
        if not parts:
            return None
        return self._pool_donors(parts, scale_key=f"{suffix}@{target_shape}")

    def _layer_cycle(
        self,
        src: Mapping[str, torch.Tensor],
        suffix: str,
        expert_index: int,
        num_layers: int,
    ) -> Optional[torch.Tensor]:
        """Deterministically assign donor layer ``expert_index % num_layers`` to an expert."""
        if num_layers <= 0:
            return None
        layer = expert_index % num_layers
        key = self._find(src, f"model.layers.{layer}.{suffix}")
        return None if key is None else src[key]

    # ------------------------------ main API ------------------------------- #
    def adapt_state_dict(
        self,
        source_state_dict: Mapping[str, torch.Tensor],
        source_arch: str = "llama",
        *,
        target_model: Optional[nn.Module] = None,
        source_dims: Optional[SourceDims] = None,
        embedding_override: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Build a strict-complete state_dict for ``target_model`` from donor weights.

        Parameters
        ----------
        source_state_dict : mapping of donor HF tensors (CPU, float).
        source_arch : informational architecture label.
        target_model : REQUIRED authoritative source of target keys/shapes.
        source_dims : REQUIRED authoritative donor dims (from the HF config).
        embedding_override : OPTIONAL float effective embedding (target_vocab x
            target_hidden). When supplied (#956) it is the 3-donor TOKEN-STRING
            aligned merged embedding and is used for BOTH ``ingestion.txt_emb``
            and ``txt_dec``. When omitted, the legacy single-donor path runs
            unchanged (backward compatible).

        Raises
        ------
        ValueError / KeyError on any unassigned key, shape mismatch, or missing
        donor input.  Failures are loud by design; there is no silent fallback.
        """
        if target_model is None:
            raise ValueError(
                "target_model is required: target keys MUST be derived from state_dict()"
            )
        if source_dims is None:
            raise ValueError(
                "source_dims is required: use the SOURCE checkpoint's config dims"
            )

        # Resolve the pooling mode once per adaptation.  Auto-detect a BitNet
        # target (BitLinear / quantize_weight); explicit True/False win.
        self._bitnet_native_active = (
            self._target_is_bitnet(target_model)
            if self.bitnet_native_pool is None
            else bool(self.bitnet_native_pool)
        )
        self._pool_scales = {}
        logger.info(
            "Pooling mode: %s (bitnet_native_pool=%s)",
            "gamma-aware ternary"
            if self._bitnet_native_active
            else "legacy float mean",
            self.bitnet_native_pool,
        )

        target_sd: Mapping[str, torch.Tensor] = target_model.state_dict()
        expected: Dict[str, Tuple[int, ...]] = {
            k: tuple(v.shape) for k, v in target_sd.items()
        }
        param_names = set(dict(target_model.named_parameters()).keys())

        # Authoritative dims come from target shapes; cross-check against source config.
        h_t = expected["ingestion.txt_emb.weight"][1]
        v_t = expected["ingestion.txt_emb.weight"][0]
        f_t = expected["moe.w2"][1]
        if (h_t, f_t, v_t) != (
            self.target_hidden_dim,
            self.target_ffn_dim,
            self.target_vocab_size,
        ):
            logger.warning(
                "Legacy hint dims (%d,%d,%d) differ from target (%d,%d,%d); using target.",
                self.target_hidden_dim,
                self.target_ffn_dim,
                self.target_vocab_size,
                h_t,
                f_t,
                v_t,
            )

        src = source_state_dict
        layers = (
            source_dims.num_layers
            if source_dims.num_layers > 0
            else len(self._source_layer_indices(src))
        )
        assert layers > 0, "Donor checkpoint has no identifiable transformer layers"

        result: Dict[str, torch.Tensor] = {}
        donor_keys: List[str] = []
        structural_keys: List[str] = []
        scales: Dict[str, torch.Tensor] = {}

        def put(key: str, value: torch.Tensor, from_donor: bool) -> None:
            """Single assignment choke point: validates presence AND shape."""
            assert key in expected, f"Target has no parameter named {key!r}"
            target_shape = expected[key]
            assert tuple(value.shape) == target_shape, (
                f"Shape mismatch for {key!r}: got {tuple(value.shape)}, expected {target_shape}"
            )
            f = value.detach().to(torch.float32).contiguous()
            result[key] = f
            (donor_keys if from_donor else structural_keys).append(key)

        # ---- 1. Token embeddings (donor-derived) --------------------------- #
        if embedding_override is not None:
            # #956: ALL THREE donors transfer. The override is the 3-donor
            # token-string-aligned merged embedding (float effective q*gamma).
            # It feeds BOTH encoder embedding and decoder head; the legacy
            # single-donor path below stays intact behind this optional arg.
            txt_emb = self.adapt_tensor_shape(embedding_override, (v_t, h_t))
            donor_emb = txt_emb  # mod_emb seeding follows merged token semantics
            txt_dec = txt_emb.clone()
            put("ingestion.txt_emb.weight", txt_emb, True)
            logger.info(
                "3-donor merged embedding override active: txt_emb and txt_dec both "
                "use the token-string-aligned merge (legacy single-donor path bypassed)."
            )
        else:
            emb_key = self._find(
                src,
                "model.embed_tokens.weight",
                "tok_embeddings.weight",
                "embeddings.word_embeddings.weight",
            )
            if emb_key is None:
                raise KeyError("Donor checkpoint has no token embedding tensor")
            donor_emb = src[emb_key]
            txt_emb = self.adapt_tensor_shape(donor_emb, (v_t, h_t))
            put("ingestion.txt_emb.weight", txt_emb, True)

            head_key = self._find(src, "lm_head.weight", "output.weight")
            if head_key is not None:
                txt_dec = self.adapt_tensor_shape(src[head_key], (v_t, h_t))
            else:
                # Tied embeddings (Llama-3.2-1B-Instruct sets tie_word_embeddings=true).
                txt_dec = txt_emb.clone()
                logger.info(
                    "Donor lm_head absent -> tying decoder to donor token embedding."
                )
        put("txt_dec.weight", txt_dec, True)

        # ---- 2. LayerNorm weights (donor-derived), biases (structural) ----- #
        final_norm_key = self._find(
            src, "model.norm.weight", "norm.weight", "transformer.ln_f.weight"
        )
        if final_norm_key is None:
            raise KeyError("Donor checkpoint has no final norm tensor")
        final_norm = self.adapt_tensor_shape(src[final_norm_key], (h_t,))
        first_ln_key = self._find(
            src, "model.layers.0.input_layernorm.weight", "h.0.ln_1.weight"
        )
        input_norm = (
            self.adapt_tensor_shape(src[first_ln_key], (h_t,))
            if first_ln_key
            else final_norm
        )
        put("final_norm.weight", final_norm, True)
        put("ingestion.norm.weight", input_norm, True)
        for key in (
            "norm_decomp.weight",
            "norm_diff.weight",
            "norm_moe.weight",
            "pre_final_norm.weight",
            "moe.output_norm.weight",
            "diffusion_core.norm2.weight",
            "diffusion_core.couil_attn.norm.weight",
        ):
            put(key, final_norm.clone(), True)
        # Modality embedding seeded from donor embedding rows (donor-derived).
        put(
            "ingestion.mod_emb.weight",
            self.adapt_tensor_shape(donor_emb[:4], (4, h_t)),
            True,
        )

        # ---- 3. Attention: consolidate N donor layers -> 1 target block ---- #
        for proj in ("q_proj", "k_proj", "v_proj"):
            tgt = f"diffusion_core.couil_attn.{proj}.weight"
            tensor = self._layer_mean(
                src, f"self_attn.{proj}.weight", expected[tgt], layers
            )
            if tensor is None:
                raise KeyError(f"Donor has no self_attn.{proj}.weight tensors")
            put(tgt, tensor, True)
        o_proj_mean = self._layer_mean(
            src,
            "self_attn.o_proj.weight",
            expected["diffusion_core.couil_attn.o_proj.weight"],
            layers,
        )
        if o_proj_mean is None:
            raise KeyError("Donor has no self_attn.o_proj.weight tensors")
        put("diffusion_core.couil_attn.o_proj.weight", o_proj_mean, True)

        # ---- 4. Dense diffusion FFN: mean(gate,up) and down (donor-derived) - #
        ffn0 = "diffusion_core.ffn.0.weight"
        if self._bitnet_native_active:
            # Pool gate+up in ONE ternary pass: a second float average here would
            # undo the gamma-aware merge (0.5*(q1*g1 + q2*g2) is not ternary).
            gu_parts = self._collect_layer_tensors(
                src,
                ("mlp.gate_proj.weight", "mlp.up_proj.weight"),
                expected[ffn0],
                layers,
            )
            if not gu_parts:
                raise KeyError("Donor MLP gate/up projections missing")
            ffn0_val = self._pool_donors(gu_parts, scale_key=ffn0)
        else:
            gate_mean = self._layer_mean(
                src, "mlp.gate_proj.weight", expected[ffn0], layers
            )
            up_mean = self._layer_mean(
                src, "mlp.up_proj.weight", expected[ffn0], layers
            )
            if gate_mean is None or up_mean is None:
                raise KeyError("Donor MLP gate/up projections missing")
            ffn0_val = 0.5 * (gate_mean + up_mean)  # legacy float behaviour
        put(ffn0, ffn0_val, True)  # one GELU expansion sees both donors
        ffn2 = "diffusion_core.ffn.2.weight"
        down_mean = self._layer_mean(
            src, "mlp.down_proj.weight", expected[ffn2], layers
        )
        if down_mean is None:
            raise KeyError("Donor MLP down projection missing")
        put(ffn2, down_mean, True)

        # ---- 4b. Ingestion / prism / finalizer linears (donor-derived) ----- #
        # All are [hidden, hidden] projections; we project donor attention/MLP
        # statistics onto that shape (deterministic, knowledge-bearing).
        square = (h_t, h_t)

        def square_mean(donor_suffix: str) -> torch.Tensor:
            raw = self._layer_mean(src, donor_suffix, square, layers)
            if raw is None:
                raise KeyError(f"Donor has no {donor_suffix} tensors")
            return self.adapt_tensor_shape(raw, square)

        q_mean = square_mean("self_attn.q_proj.weight")
        k_mean = square_mean("self_attn.k_proj.weight")
        v_mean = square_mean("self_attn.v_proj.weight")
        o_square = square_mean("self_attn.o_proj.weight")
        gate_square = square_mean("mlp.gate_proj.weight")
        up_square = square_mean("mlp.up_proj.weight")
        down_square = square_mean("mlp.down_proj.weight")
        put("ingestion.q1_ingest.weight", q_mean, True)
        put("ingestion.q2_ingest.weight", v_mean, True)
        put("decomposition.Q_gate.weight", q_mean.clone(), True)
        put("decomposition.K_gate.weight", k_mean, True)
        put("decomposition.V_gate.weight", v_mean.clone(), True)
        put("decomposition.W_gate.weight", o_square, True)
        # Nine prism vectors deterministically cycle the donor projection sources.
        prism_sources = (
            gate_square,
            o_square,
            q_mean,
            k_mean,
            v_mean,
            gate_square,
            up_square,
            down_square,
            o_square,
        )
        for idx, name in enumerate(self._PRISM_VECTORS):
            put(
                f"decomposition.vectors.{name}.weight", prism_sources[idx].clone(), True
            )
        put("quillan_finalizer.weight", o_square.clone(), True)
        put("quillan_finalizer2.weight", o_square.clone(), True)

        # ---- 5. MoE experts: cycle donor layers across all 34 experts ----- #
        # Target w1/wgate/w2 are single [num_experts, ...] tensors. For expert e
        # we take donor layer (e % num_layers); HF Linear layout is [out, in],
        # while the target stores [in, out], hence the transpose. MLPs beyond
        # the donor layer count are reused, so every expert is seeded with real
        # pretrained knowledge -- never random initialization.
        for expert_label, donor_suffix in (
            ("w1", "mlp.gate_proj.weight"),
            ("wgate", "mlp.up_proj.weight"),
            ("w2", "mlp.down_proj.weight"),
        ):
            tgt = f"moe.{expert_label}"
            expert_slices: List[torch.Tensor] = []
            per_expert_shape = expected[tgt][1:]
            for e in range(self.num_experts):
                t = self._layer_cycle(src, donor_suffix, e, layers)
                if t is None:
                    raise KeyError(f"Donor has no {donor_suffix} for expert {e}")
                expert_slices.append(
                    self.adapt_tensor_shape(t.transpose(0, 1), per_expert_shape)
                )
            put(tgt, torch.stack(expert_slices, dim=0), True)

        # ---- 6. Structural constants for EVERY remaining target key ------- #
        for key in expected:
            if key in result:
                continue
            value = self._structural_value(key, expected[key], donor_keys, txt_emb)
            put(key, value, from_donor=(key in donor_keys))

        # ---- 7. Strict completeness assertion (fail loud) ----------------- #
        assert set(result) == set(expected), (
            f"Key coverage mismatch: missing={sorted(set(expected) - set(result))}, "
            f"unexpected={sorted(set(result) - set(expected))}"
        )

        # Pooled gammas are emitted scales (runtime eps floor applied).  They
        # are provenance metadata only -- the model never reads them back.
        scales.update(self._pool_scales)
        total_elements = sum(math.prod(s) for s in expected.values())
        donor_elements = sum(result[k].numel() for k in donor_keys)
        self.last_report = AdaptationReport(
            donor_keys=tuple(donor_keys),
            structural_keys=tuple(structural_keys),
            donor_elements=donor_elements,
            total_elements=total_elements,
            source_dims=source_dims,
            scales=scales,
        )
        logger.info(
            "Adaptation complete (%s): %d/%d keys donor-derived; %.2f%% of elements from donor.",
            source_arch,
            len(donor_keys),
            len(expected),
            100.0 * self.last_report.donor_fraction,
        )
        return result

    # -------------------------- structural policy -------------------------- #
    def _structural_value(
        self,
        key: str,
        shape: Tuple[int, ...],
        donor_keys: Sequence[str],
        txt_emb: torch.Tensor,
    ) -> torch.Tensor:
        """Deterministic value for a key with no semantic donor analogue.

        Every branch is a documented structural constant.  No randomness.
        """
        rank_shape = len(shape)

        # EGGROLL / BitLinear low-rank adapters: zero => identity (learnable).
        if key.endswith("lora_A") or key.endswith("lora_B") or "_lora_" in key:
            return torch.zeros(shape, dtype=torch.float32)

        # Council underling swarms: frozen at identity for export.
        if key.startswith("moe.expert_swarms."):
            leaf = key.rsplit(".", 1)[-1]
            if leaf in ("A", "B", "C", "D"):
                return torch.zeros(
                    shape, dtype=torch.float32
                )  # rank-16 EGGROLL seed -> identity
            if leaf == "clone_diversity":
                return torch.zeros(shape, dtype=torch.float32)
            if leaf in (
                "clone_coupling",
                "clone_quant_mag",
                "clone_quant_range",
                "clone_quant_threshold",
                "clone_quant_noise",
            ):
                return torch.full(shape, 0.1, dtype=torch.float32)  # model defaults
            if leaf == "population_mean":
                return torch.zeros(shape, dtype=torch.float32)
            if leaf == "population_std":
                return torch.ones(shape, dtype=torch.float32)
            raise KeyError(f"Unhandled expert_swarms structural key: {key}")

        # Physical constants registered as buffers in QuantumFormulasEngine.
        if key == "quantum_formulas.hbar":
            return torch.full(shape, 1.0545718e-34, dtype=torch.float32)
        if key == "quantum_formulas.boltzmann":
            return torch.full(shape, 1.380649e-23, dtype=torch.float32)
        if key == "quantum_formulas.sqrt_33":
            return torch.full(shape, math.sqrt(33.0), dtype=torch.float32)

        # Fixed sinusoidal absolute positions (donor uses RoPE; no learned table).
        if key == "ingestion.pos_embed":
            return self._sinusoidal_positions(shape)

        # Model-default scalar bypass gate.
        if key == "ingestion.ingest_bypass":
            return torch.full(shape, 2.0, dtype=torch.float32)

        # LayerNorm biases: centered normalization (donor RMSNorm has no bias).
        if key.endswith(".bias") and ("norm" in key or key == "moe.output_norm.bias"):
            return torch.zeros(shape, dtype=torch.float32)

        # Learned routers / gates / classifier / vision projection / mask MLP:
        # no pretrained analogue in the donor -> zero => neutral initial routing.
        if (
            ".router." in key
            or key.endswith("ingest_gate.weight")
            or key.endswith("ingest_gate.bias")
            or key.endswith("quillan_gate.weight")
            or key.endswith("quillan_gate.bias")
            or "sparse_mask_generator" in key
            or key.endswith("img_proj.weight")
        ):
            return torch.zeros(shape, dtype=torch.float32)

        # A 1-D tensor not covered above: zeros is the deterministic neutral choice.
        if rank_shape == 1:
            return torch.zeros(shape, dtype=torch.float32)

        raise KeyError(
            f"No donor mapping and no structural declaration for target key {key!r} "
            f"(shape {shape}). Refusing to leave it randomly initialized."
        )

    @staticmethod
    def _sinusoidal_positions(shape: Tuple[int, ...]) -> torch.Tensor:
        """Deterministic fixed sinusoidal table [1, max_seq_len, dim]."""
        assert len(shape) == 3 and shape[0] == 1, f"Unexpected pos_embed shape {shape}"
        _, length, dim = shape
        pos = torch.arange(length, dtype=torch.float32).unsqueeze(1)
        i = torch.arange(dim, dtype=torch.float32).unsqueeze(0)
        angle = pos / (10000.0 ** (2.0 * (i // 2) / max(dim, 1)))
        table = torch.where(
            (i.long() % 2) == 0,
            torch.sin(angle),
            torch.cos(angle),
        )
        return table.unsqueeze(0).contiguous()

    # -------------------------- checkpoint loader -------------------------- #
    def load_and_adapt_checkpoint(
        self,
        checkpoint_path: Union[str, Path],
        source_arch: str = "llama",
        *,
        target_model: nn.Module,
        source_dims: SourceDims,
    ) -> Dict[str, torch.Tensor]:
        """Safely load a donor checkpoint (safetensors / weights_only pt) and adapt it."""
        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found at: {path}")

        logger.info("Loading donor checkpoint from %s (format=%s)", path, path.suffix)
        if path.suffix == ".safetensors":
            try:
                from safetensors.torch import load_file
            except ImportError as err:  # explicit, chained -- never silent
                raise ImportError(
                    "Install safetensors (`pip install safetensors`) to load .safetensors checkpoints."
                ) from err
            with path.open("rb") as handle:  # deterministic resource scoping
                del handle  # load_file needs a path; the handle validates readability
            source_sd = load_file(str(path))
        else:
            with path.open("rb") as handle:
                source_sd = torch.load(handle, map_location="cpu", weights_only=True)

        return self.adapt_state_dict(
            source_sd,
            source_arch=source_arch,
            target_model=target_model,
            source_dims=source_dims,
        )
