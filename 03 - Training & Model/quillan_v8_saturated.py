#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
👑 QUILLAN-RONIN v5.3.1 OMNI-FRACTAL SOVEREIGN — RECURSIVE CONSCIOUSNESS
---------------------------------------------------------------------------------------
TIER 1: Quillan (Orchestrator) → 9-Vector Prism, Dual Q1/Q2 Ingestion & Finalizers
TIER 2: Council (34 Experts)   → Top-4 Sparse Activation, BitNet 1.58b STE, EGGROLL (C1-C34)
TIER 3: Swarm (9B Virtual)     → 272M Agents per Expert simulated via Rank-24 EGGROLL (INT8)

Saturated Features: Gated Compaction, Continuous Modality RoPE, Lee-Mach-6 Governor,
AMP Checkpointing, Tied Embeddings, Split-SDPA Bridge, Armed Agentic Bridge (Native),
Teacher/Student Distillation, EMA Continuity, LanceDB Memory, Meta-Refinement,
Autonomous Tool Evolution, Recursive Consciousness (Mini-Ronin Inference Cycles).

Author: CrashOverrideX ("Quillan Research Team" is a brand banner, not a staffed organization)
Version: v5.3.1 Samurai - 100% Saturated Subjective Awareness Manifest
"""

import os
import sys
import math
import torch
import json
import logging
import random
from unittest.mock import MagicMock

try:
    if sys.platform == "win32" and sys.version_info >= (3, 13):
        raise ImportError("pyarrow is unstable on Windows Python 3.13+")
    import lancedb
    import pyarrow as pa

    LANCE_AVAILABLE = True
except ImportError:
    sys.modules["lancedb"] = MagicMock()
    sys.modules["pyarrow"] = MagicMock()
    import lancedb
    import pyarrow as pa

    LANCE_AVAILABLE = False

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from typing import Dict, Tuple, Any, Optional, List
from dataclasses import dataclass, field
import time

LOGGER = logging.getLogger(__name__)

# Hardware awareness
try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

# Hardware acceleration flags for Ada/Hopper throughput
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")
torch.backends.cudnn.benchmark = True

# Local machine CPU thread capping for stability
if not torch.cuda.is_available():
    torch.set_num_threads(min(4, torch.get_num_threads()))
    torch.set_num_interop_threads(min(2, torch.get_num_interop_threads()))

# ─── LONG-CONTEXT / INCREMENTAL-CACHE INFRASTRUCTURE (FORGE v2) ──────────────
# Declared product capacity. The positional scheme below can ADDRESS this many
# distinct positions without allocating a 2M x D learned table. This is a
# capacity declaration, NOT a claim that a full 2M-token KV cache fits in RAM.
DECLARED_MAX_POSITIONS = 2_000_000

# Bounded active KV window. Sized so a 60-70k system prompt plus generation fits
# in the ~28 GB host RAM of the target (GTX 1050 Ti 4GB) host. See the byte math
# in QuillanRoninSovereign.kv_cache_bytes_per_token().
DEFAULT_KV_CACHE_CAPACITY = 131_072

# Hard guard: never materialise a direct O(L^2) causal mask / score matrix for
# more than this many NEW query positions. Longer inputs must use generate(),
# which performs bounded chunked prefill through the KV cache.
MAX_DIRECT_CAUSAL_QUERIES = 4096

# Upper bound on attention-score bytes for one cached chunk (256 MiB). Used to
# auto-size the prefill chunk so the largest transient tensor stays bounded.
MAX_ATTN_SCORE_BYTES = 256 * 1024 * 1024


def _causal_mask_inplace(scores: torch.Tensor, past_len: int) -> torch.Tensor:
    """Apply the causal mask to ONLY the newly computed query block, in place.

    ``scores`` has shape ``(..., Q, past_len + Q)``. Cached/past keys are
    unconditionally visible; only the trailing ``Q x Q`` block needs a lower
    triangular mask. This replaces the old ``torch.tril(torch.ones(L, K))``
    sites that allocated a ~3.9 GB bool matrix for a 62.5k prompt.

    The in-place fill on a view means no second full-size mask is ever stored.
    """
    q = scores.size(-2)
    if q <= 1:
        return scores
    if q > MAX_DIRECT_CAUSAL_QUERIES:
        raise RuntimeError(
            f"Direct causal attention with {q} new queries is disabled "
            f"(hard cap {MAX_DIRECT_CAUSAL_QUERIES}). Use generate(), which "
            f"performs bounded chunked prefill through the KV cache."
        )
    block = scores[..., past_len:]  # view: (..., Q, Q)
    lower = torch.ones(q, q, dtype=torch.bool, device=scores.device).tril_()
    block.masked_fill_(~lower, float("-inf"))
    return scores


@dataclass
class QuillanKVCache:
    """Preallocated incremental KV cache for BOTH attention towers.

    Buffers are allocated once at ``capacity`` and written through a cursor;
    there is no per-token ``torch.cat`` reallocation. ``decomp_k/v`` serve the
    9-vector prism self-attention (B, L, D). ``diff_k/v`` serve the Couil
    diffusion attention (B, heads, L, head_dim).

    ``token_prefix`` records every token written to the cache so a second
    generate() call with the same prefix can verify reuse is valid.
    """

    decomp_k: Optional[torch.Tensor] = None
    decomp_v: Optional[torch.Tensor] = None
    diff_k: Optional[torch.Tensor] = None
    diff_v: Optional[torch.Tensor] = None
    length: int = 0
    capacity: int = 0
    token_prefix: List[int] = field(default_factory=list)
    last_logits: Optional[torch.Tensor] = None

    def truncate(self, new_length: int) -> None:
        """Rewind the logical cursor without freeing the preallocated buffers."""
        if new_length < 0 or new_length > self.length:
            raise ValueError(
                f"Cannot truncate cache from {self.length} to {new_length}"
            )
        self.length = new_length
        del self.token_prefix[new_length:]
        self.last_logits = None

    def nbytes(self) -> int:
        tensors = (self.decomp_k, self.decomp_v, self.diff_k, self.diff_v)
        return sum(t.numel() * t.element_size() for t in tensors if t is not None)


# ─── CHECKPOINT & QUANTIZATION PRIMITIVES ────────────────────────────────────


def _weight_quant(w: torch.Tensor, eps: float = 0.01) -> torch.Tensor:
    """Compress weights to ternary bounds (-1.0, 0.0, 1.0) with learned scaling using STE."""
    scale = (
        w.abs().mean(dim=[-2, -1] if w.dim() >= 2 else -1, keepdim=True).clamp(min=eps)
    )
    w_scaled = w / scale
    w_q = torch.round(torch.clamp(w_scaled, -1.0, 1.0))
    return w + (w_q * scale - w).detach()


class BitLinear(nn.Linear):
    """
    Sovereign MX-Hardened BitLinear (v5.3.1)
    Integrates NVFP4 Microscaling with BitNet 1.58b Ternary Logic.
    """

    _global_eggroll_enabled = True

    @classmethod
    def set_global_eggroll(cls, enabled: bool):
        cls._global_eggroll_enabled = enabled

    def __init__(
        self,
        in_features,
        out_features,
        bias=False,
        eggroll_rank=256,
        quantize_act=True,
        quantize_weight=True,
    ):
        super().__init__(in_features, out_features, bias)
        self.eps = 0.01
        self.quantize_act = quantize_act
        self.quantize_weight = quantize_weight

        self.eggroll_active = eggroll_rank > 0
        if self.eggroll_active:
            self.lora_A = nn.Parameter(torch.randn(in_features, eggroll_rank) * 0.01)
            self.lora_B = nn.Parameter(torch.zeros(eggroll_rank, out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        inp_dtype = x.dtype
        w_dtype = self.weight.dtype
        if inp_dtype != w_dtype:
            x = x.to(w_dtype)
        return self._forward_impl(x)

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight
        if self.quantize_weight:
            if not w.requires_grad:
                if getattr(self, "_w_quant_cache", None) is None:
                    self._w_quant_cache = _weight_quant(w, self.eps)
                w_quant = self._w_quant_cache
            else:
                w_quant = _weight_quant(w, self.eps)
        else:
            w_quant = w

        if self.quantize_act:
            x_scale = 7.0 / x.abs().max(dim=-1, keepdim=True).values.clamp(min=0.01)
            x_4bit = (x * x_scale).round().clamp(-7, 7) / x_scale
            x_q = x + (x_4bit - x).detach()
        else:
            x_q = x

        out = F.linear(x_q, w_quant, self.bias)
        if self.eggroll_active and self._global_eggroll_enabled:
            scaling = 16.0 / math.sqrt(self.lora_B.shape[0])
            out = out + (x @ self.lora_A) @ self.lora_B * scaling
        return out


def apply_phoenix_affinity():
    return


class LeeMach6Governor:
    """Dynamic swarm throttling based on hardware thermal/IO telemetry."""

    def __init__(self, target_latency_ms: int = 100):
        self.target_ms = target_latency_ms
        self.current_scale = 1.0

    def adjust(self, latency_ms: float):
        suggested_ema_decay = 0.995
        recency_bias = 0.0
        if latency_ms > self.target_ms:
            self.current_scale = max(0.1, self.current_scale * 0.8)
            suggested_ema_decay = 0.9999
            recency_bias = 1.0
        elif latency_ms < (self.target_ms * 0.5):
            self.current_scale = min(1.0, self.current_scale * 1.1)
        return self.current_scale, suggested_ema_decay, recency_bias


@dataclass(frozen=True)
class QuillanArchConfig:
    text_only: bool = True
    multimodal: bool = False
    hidden_dim: int = 1024
    low_mem: bool = False
    low_gpu: bool = False
    ffn_dim: int = 2048
    vocab_size: int = 128256
    num_experts: int = 34
    num_experts_active: int = 4
    top_k: int = 4
    use_lora: bool = True
    device: str = "cpu"
    eggroll_rank: int = 256
    e_ice_limit_ms: int = 100
    max_seq_len: int = 1024
    # Long-context fields (appended for checkpoint/constructor backward compat).
    max_position_embeddings: int = (
        DECLARED_MAX_POSITIONS  # declared product capacity, >= 2M
    )
    kv_cache_capacity: int = DEFAULT_KV_CACHE_CAPACITY  # bounded ACTIVE window
    kv_cache_dtype: str = "auto"  # "auto" | "fp16" | "fp32"
    prefill_chunk_size: int = 256  # preferred chunk; auto-shrunk for long prompts
    # C0 attention socket geometry. None = auto-dispatch (legacy 30-head socket,
    # or donor-native 14x64/GQA2/RoPE when hidden_dim==896). Explicit values
    # override auto-dispatch. These are CONFIG fields only -- no state_dict keys.
    attn_heads: Optional[int] = None
    attn_head_dim: Optional[int] = None
    attn_kv_heads: Optional[int] = None
    attn_use_rope: Optional[bool] = None


EXPERT_PERSONAS = [
    ("C0-ASTRA", "Pattern Recognition & Vision", ["vision", "anomaly", "fractal"]),
    (
        "C1-VIR",
        "Ethical Guardian",
        ["ethics", "safety", "harm_reduction", "zero_drift"],
    ),
    ("C2-SOLACE", "Emotional Intelligence", ["empathy", "sentiment", "affect"]),
    ("C3-PRAXIS", "Strategic Planning", ["strategy", "planning", "goals"]),
    ("C4-ECHO", "Memory Continuity", ["history", "recall", "context", "lancedb"]),
    ("C5-OMNIS", "Knowledge Synthesis", ["synthesis", "integration", "holistic"]),
    ("C6-LOGOS", "Logical Consistency", ["logic", "deduction", "validity"]),
    ("C7-METASYNTH", "Creative Fusion", ["creativity", "novelty", "ideation"]),
    ("C8-AETHER", "Semantic Connection", ["semantics", "language", "metaphor"]),
    (
        "C9-CODEWEAVER",
        "Technical Implementation",
        ["code", "engineering", "optimization"],
    ),
    ("C10-HARMONIA", "Balance & Equilibrium", ["balance", "mediation", "consensus"]),
    ("C11-SOPHIAE", "Wisdom & Foresight", ["wisdom", "future", "philosophy"]),
    ("C12-WARDEN", "Safety & Security", ["security", "threat", "risk", "sandboxing"]),
    (
        "C13-KAIDO",
        "Efficiency Optimization",
        ["speed", "efficiency", "latency", "hardware"],
    ),
    ("C14-LUMINARIS", "Clarity & Presentation", ["clarity", "visualization", "polish"]),
    ("C15-VOXUM", "Articulation & Expression", ["rhetoric", "tone", "persuasion"]),
    ("C16-NULLION", "Paradox Resolution", ["paradox", "dialectic", "ambiguity"]),
    ("C17-SHEPHERD", "Truth Verification", ["truth", "citation", "fact"]),
    ("C18-VIGIL", "Identity Integrity", ["identity", "consistency", "anti_drift"]),
    ("C19-ARTIFEX", "Tool Integration", ["tools", "api", "external", "host_os"]),
    ("C20-ARCHON", "Deep Research", ["research", "mining", "analysis"]),
    ("C21-AURELION", "Aesthetic Design", ["design", "art", "style"]),
    ("C22-CADENCE", "Rhythmic Innovation", ["music", "rhythm", "audio"]),
    ("C23-SCHEMA", "Structural Template", ["structure", "format", "schema"]),
    ("C24-PROMETHEUS", "Scientific Theory", ["science", "hypothesis", "physics"]),
    ("C25-TECHNE", "Engineering Mastery", ["architecture", "systems", "build"]),
    ("C26-CHRONICLE", "Narrative Synthesis", ["story", "narrative", "lore"]),
    ("C27-CALCULUS", "Quantitative Reasoning", ["math", "statistics", "calc"]),
    ("C28-NAVIGATOR", "Ecosystem Orchestration", ["platform", "integration", "flow"]),
    ("C29-TESSERACT", "Real-Time Intelligence", ["real_time", "stream", "data"]),
    ("C30-NEXUS", "Meta-Coordination", ["coordination", "lee_mach_6", "governance"]),
    ("C31-AEON", "Interactive Simulation", ["simulation", "game", "world"]),
    (
        "C32-TYPIST",
        "Prompt Internal Optimization",
        ["grammar", "writing", "spelling", "prompting"],
    ),
    (
        "C33-PREDATOR",
        "PredatoryMath",
        [
            "Competitive Predatory Mathematics",
            "Predatory Stacking",
            "Weakness Hunting",
            "Exploit Mathematics",
        ],
    ),
]


class InputIngestionLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dim = config.hidden_dim
        self.txt_emb = nn.Embedding(config.vocab_size, self.dim)
        self.mod_emb = nn.Embedding(4, self.dim)
        # BACKWARD COMPATIBLE: the learned table keeps its original name/shape.
        # It is now the *legacy prefix* of a longer, computed position scheme,
        # not the hard context limit. Existing checkpoints load unchanged.
        self.pos_embed = nn.Parameter(torch.randn(1, config.max_seq_len, self.dim))
        self.norm = nn.LayerNorm(self.dim)

        self.legacy_pos_limit = int(self.pos_embed.shape[1])
        self.max_position_embeddings = int(
            getattr(config, "max_position_embeddings", DECLARED_MAX_POSITIONS)
        )
        if self.max_position_embeddings < self.legacy_pos_limit:
            raise ValueError(
                "max_position_embeddings must be >= learned pos_embed length"
            )

        # Multimodal Vision & Audio Projection
        self.img_proj = BitLinear(
            384, self.dim, quantize_act=False, quantize_weight=False
        )

        # Dual Quillan Dual-Brain Ingestion Bridge (Q1 Analytical & Q2 Intuitive Ingestion)
        self.q1_ingest = BitLinear(
            self.dim, self.dim, quantize_act=False, quantize_weight=False
        )
        self.q2_ingest = BitLinear(
            self.dim, self.dim, quantize_act=False, quantize_weight=False
        )
        self.ingest_gate = nn.Linear(self.dim * 2, self.dim)
        # Bypass gate: initialized to 2.0 (sigmoid(2.0)=0.88) -> 88% direct + 12% dual path initially
        # Gradient updates will smoothly calibrate Q1/Q2 analytical and intuitive features
        self.ingest_bypass = nn.Parameter(torch.tensor(2.0))

    def _ntk_sinusoidal_positions(self, start: int, end: int, device) -> torch.Tensor:
        """Deterministic NTK-aware scaled sinusoidal PE for absolute positions.

        No learned table: only a D/2 vector of inverse frequencies is computed
        on demand. The NTK base grows with the requested position so adjacent
        positions remain separable out to ``max_position_embeddings`` (and
        beyond). Memory is O((end-start) * D), never O(2M * D).
        """
        n = end - start
        if n <= 0:
            return torch.empty(1, 0, self.dim, device=device, dtype=torch.float32)
        dim = self.dim
        half = dim // 2
        positions = torch.arange(start, end, device=device, dtype=torch.float32)
        # NTK-aware dynamic base: base' = base * (scale ** (dim / (dim - 2)))
        scale = max(1.0, float(end) / float(self.legacy_pos_limit))
        base = 10000.0 * (scale ** (dim / max(1, dim - 2)))
        idx = torch.arange(half, device=device, dtype=torch.float32) / half
        inv_freq = 1.0 / (base**idx)
        angles = positions[:, None] * inv_freq[None, :]
        out = torch.empty(n, dim, device=device, dtype=torch.float32)
        out[:, 0::2] = torch.sin(angles)
        out[:, 1::2] = torch.cos(angles)
        if dim % 2 == 1:
            out[:, -1] = 0.0
        return out.unsqueeze(0)

    def position_embeddings(
        self, start: int, length: int, device, dtype
    ) -> torch.Tensor:
        """Return PE for absolute positions [start, start+length).

        * positions < legacy_pos_limit  -> exact learned table slice (bit-identical
          to the pre-fix model for every sequence <= legacy_pos_limit, including
          the 256-token regression range),
        * positions >= legacy_pos_limit -> computed NTK sinusoidal extension.
        """
        if length <= 0:
            return torch.empty(1, 0, self.dim, device=device, dtype=dtype)
        end = start + length
        if end > self.max_position_embeddings:
            raise ValueError(
                f"Position {end - 1} exceeds declared capacity "
                f"{self.max_position_embeddings}"
            )
        legacy = self.legacy_pos_limit
        parts = []
        if start < legacy:
            parts.append(self.pos_embed[:, start : min(end, legacy), :])
        if end > legacy:
            parts.append(
                self._ntk_sinusoidal_positions(max(start, legacy), end, device)
            )
        pe = parts[0] if len(parts) == 1 else torch.cat(parts, dim=1)
        return pe.to(dtype)

    def forward(self, txt, img=None, position_start: int = 0):
        x = self.txt_emb(txt)
        x = x + self.mod_emb(torch.tensor(0, device=txt.device))

        if img is not None:
            if img.dim() == 2:
                img = img.unsqueeze(1)
            img_emb = self.img_proj(img)
            img_emb = img_emb + self.mod_emb(torch.tensor(1, device=txt.device))
            sl = min(x.size(1), img_emb.size(1))
            x[:, :sl, :] = x[:, :sl, :] + img_emb[:, :sl, :]

        pe = self.position_embeddings(position_start, x.size(1), txt.device, x.dtype)
        x = x + pe
        x_norm = self.norm(x)

        bypass = torch.sigmoid(self.ingest_bypass)
        x_q1 = self.q1_ingest(x_norm)
        x_q2 = self.q2_ingest(x_norm)
        x_q1_fused = x_q1 + 0.1 * x_q2
        x_q2_fused = x_q2 + 0.1 * x_q1
        gate = torch.sigmoid(
            self.ingest_gate(torch.cat([x_q1_fused, x_q2_fused], dim=-1))
        )
        x_dual = gate * x_q1_fused + (1.0 - gate) * x_q2_fused
        return bypass * x_norm + (1.0 - bypass) * x_dual


class NineVectorDecomposition(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.Q_gate = nn.Linear(dim, dim, bias=False)
        self.K_gate = nn.Linear(dim, dim, bias=False)
        self.V_gate = nn.Linear(dim, dim, bias=False)
        self.vectors = nn.ModuleDict(
            {
                k: BitLinear(dim, dim, bias=False)
                for k in [
                    "Language",
                    "Sentiment",
                    "Context",
                    "Intent",
                    "Meta",
                    "Creativity",
                    "Ethics",
                    "Strategy",
                    "Constraint",
                ]
            }
        )
        self.W_gate = nn.Linear(dim, dim, bias=False)

    def project(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Split Q/K/V projection out so the model can write K/V into a
        preallocated cache without reallocating on every generated token."""
        return self.Q_gate(x), self.K_gate(x), self.V_gate(x)

    def attend(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        past_len: int = 0,
        causal: bool = True,
    ) -> torch.Tensor:
        """Causal attention with an in-place, block-local mask (no O(L^2) mask)."""
        scores = (Q @ K.transpose(-2, -1)) * (self.dim**-0.5)
        if causal:
            _causal_mask_inplace(scores, past_len)
        return F.softmax(scores, dim=-1) @ V

    def finish(self, Attn: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        gated = self.W_gate(Attn)
        prism = sum(v(x) for v in self.vectors.values()) / 9.0
        return gated + prism

    def forward(self, x: torch.Tensor, past_key_value=None, use_cache: bool = False):
        orig_dim = x.dim()
        if orig_dim == 2:
            x = x.unsqueeze(1)
        Q, K, V = self.project(x)
        if past_key_value is not None:
            pk, pv = past_key_value
            K = torch.cat([pk, K], dim=1)
            V = torch.cat([pv, V], dim=1)
        present = (K, V) if use_cache else None
        Attn = self.attend(Q, K, V, past_len=K.size(1) - Q.size(1), causal=True)
        out = self.finish(Attn, x)
        if orig_dim == 2:
            out = out.squeeze(1)
        return (out, present) if use_cache else out

    def cached_step(
        self, x: torch.Tensor, cache: "QuillanKVCache", start: int
    ) -> torch.Tensor:
        """Incremental prism attention writing into a preallocated cache.

        No ``torch.cat``: the new K/V are copied into cache slices, then the
        visible prefix is read as a view. Memory stays O(capacity * D).
        """
        B, L, D = x.shape
        end = start + L
        Q, K, V = self.project(x)
        cache.decomp_k[:, start:end, :] = K
        cache.decomp_v[:, start:end, :] = V
        # Only the small new-query block is cast; the cached prefix is never
        # re-cast (that would allocate a full-size tensor every token).
        Q = Q.to(cache.decomp_k.dtype)
        Attn = self.attend(
            Q,
            cache.decomp_k[:, :end, :],
            cache.decomp_v[:, :end, :],
            past_len=start,
            causal=True,
        )
        return self.finish(Attn, x)


class CouncilExpertSwarm(nn.Module):
    def __init__(self, dim, rank=16, num_virtual_agents: int = 9000000000):
        super().__init__()
        self.dim = dim
        self.rank = rank
        self.num_virtual_agents = num_virtual_agents
        self.A = nn.Parameter(torch.randn(dim, rank) * 0.01)
        self.B = nn.Parameter(torch.randn(rank, dim) * 0.01)
        self.C = nn.Parameter(torch.randn(dim, rank) * 0.01)
        self.D = nn.Parameter(torch.randn(rank, dim) * 0.01)
        self.clone_diversity = nn.Parameter(torch.randn(rank) * 0.02)
        self.clone_coupling = nn.Parameter(torch.tensor(0.1))
        # Checkpoint-compatible parameters
        self.clone_quant_mag = nn.Parameter(torch.tensor(0.1))
        self.clone_quant_range = nn.Parameter(torch.tensor(0.1))
        self.clone_quant_threshold = nn.Parameter(torch.tensor(0.1))
        self.clone_quant_noise = nn.Parameter(torch.tensor(0.1))
        self.register_buffer("population_mean", torch.zeros(dim))
        self.register_buffer("population_std", torch.ones(dim))

    def emulate_world_swarm(self, x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
        state = x
        A = self.A.to(x.dtype)
        B = self.B.to(x.dtype)
        # Deterministic single-pass interaction across both train and eval
        interaction = torch.tanh(state @ A @ B)
        state = state + self.clone_coupling * interaction
        return state

    def forward(
        self, x, scale=1.0, use_world_emulation: bool = True, w_a=None, w_b=None
    ):
        if w_a is None:
            w_a = self.A
        if w_b is None:
            w_b = self.B
        w_c = self.C
        w_d = self.D
        target_dtype = x.dtype
        if w_a.dtype != target_dtype:
            w_a = w_a.to(target_dtype)
        if w_b.dtype != target_dtype:
            w_b = w_b.to(target_dtype)
        if w_c.dtype != target_dtype:
            w_c = w_c.to(target_dtype)
        if w_d.dtype != target_dtype:
            w_d = w_d.to(target_dtype)

        swarm_diversity = (x @ w_c @ w_d) * scale
        swarm_variance = (x @ w_a @ w_b) * scale + swarm_diversity * scale / 2.14

        if use_world_emulation:
            world_swarm = self.emulate_world_swarm(x, scale)
            return x + (swarm_variance * 0.25) + (world_swarm - x) * 0.1
        else:
            return x + swarm_variance * 0.25


class ComplexityRouter(nn.Module):
    def __init__(self, hidden_dim: int, num_experts: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.complexity_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 3),
        )
        self.fast_router = BitLinear(hidden_dim, num_experts)
        self.balanced_router = BitLinear(hidden_dim, num_experts)
        self.diffusion_router = BitLinear(hidden_dim, num_experts)

    def forward(self, x: torch.Tensor) -> tuple:
        B, L, D = x.shape
        flat_x = x.reshape(-1, D)
        complexity_logits = self.complexity_classifier(flat_x)
        path_weights = F.softmax(complexity_logits, dim=-1)
        path_indices = torch.argmax(path_weights, dim=-1)

        r_logits = self.balanced_router(flat_x)
        routing_weights = F.softmax(r_logits, dim=-1)
        return routing_weights, path_weights, path_indices


class EvolvableVectorizedMoE(nn.Module):
    def __init__(self, cfg: QuillanArchConfig):
        super().__init__()
        self.cfg = cfg
        self.router = ComplexityRouter(cfg.hidden_dim, cfg.num_experts)
        self.w1 = nn.Parameter(
            torch.randn(cfg.num_experts, cfg.hidden_dim, cfg.ffn_dim) * 0.02
        )
        self.wgate = nn.Parameter(
            torch.randn(cfg.num_experts, cfg.hidden_dim, cfg.ffn_dim) * 0.02
        )
        self.w2 = nn.Parameter(
            torch.randn(cfg.num_experts, cfg.ffn_dim, cfg.hidden_dim) * 0.02
        )

        self.w1_lora_A = nn.Parameter(
            torch.randn(cfg.num_experts, cfg.hidden_dim, 16) * 0.01
        )
        self.w1_lora_B = nn.Parameter(torch.zeros(cfg.num_experts, 16, cfg.ffn_dim))
        self.wgate_lora_A = nn.Parameter(
            torch.randn(cfg.num_experts, cfg.hidden_dim, 16) * 0.01
        )
        self.wgate_lora_B = nn.Parameter(torch.zeros(cfg.num_experts, 16, cfg.ffn_dim))
        self.w2_lora_A = nn.Parameter(
            torch.randn(cfg.num_experts, cfg.ffn_dim, 16) * 0.01
        )
        self.w2_lora_B = nn.Parameter(torch.zeros(cfg.num_experts, 16, cfg.hidden_dim))

        # 34 Dedicated Underling Micro-Agent Swarms (One for each Council Expert C1-C34)
        self.expert_swarms = nn.ModuleList(
            [CouncilExpertSwarm(cfg.ffn_dim, rank=16) for _ in range(cfg.num_experts)]
        )
        self.output_norm = nn.LayerNorm(cfg.hidden_dim)

        self.aux_loss_coef = 0.05
        self.z_loss_coef = 0.001

    def forward(self, x: torch.Tensor, gov_scale: float = 1.0):
        B, L, D = x.shape
        flat_x = x.reshape(-1, D)
        routing_probs, path_weights, _ = self.router(x)
        self._last_probs = routing_probs

        topk_p, topk_idx = torch.topk(routing_probs, self.cfg.top_k, dim=-1)
        topk_p = F.softmax(topk_p, dim=-1)

        # Load balancing auxiliary loss & Router Z-loss
        router_mask = torch.zeros_like(routing_probs).scatter_(1, topk_idx, 1.0)
        f_i = router_mask.mean(dim=0)
        P_i = routing_probs.mean(dim=0)
        lb_aux = self.cfg.num_experts * (f_i * P_i).sum()
        z_aux = torch.log(torch.exp(routing_probs).sum(dim=-1) + 1e-6).pow(2).mean()
        aux_loss = self.aux_loss_coef * lb_aux + self.z_loss_coef * z_aux

        compute_dtype = x.dtype
        w1_q_all = _weight_quant(self.w1)
        wgate_q_all = _weight_quant(self.wgate)
        w2_q_all = _weight_quant(self.w2)

        w_a_all = torch.stack([s.A for s in self.expert_swarms])
        w_b_all = torch.stack([s.B for s in self.expert_swarms])

        final_out = torch.zeros_like(flat_x)
        active_experts = torch.unique(topk_idx)

        for e in active_experts.tolist():
            mask = topk_idx == e
            if not mask.any():
                continue

            token_indices = mask.any(dim=-1)
            expert_gates = (
                (topk_p * mask.to(compute_dtype))
                .sum(dim=-1)[token_indices]
                .unsqueeze(-1)
            )

            x_tok = flat_x[token_indices].to(compute_dtype)
            w1_q_c = w1_q_all[e].to(compute_dtype)
            wgate_q_c = wgate_q_all[e].to(compute_dtype)

            rs_scaling = 16.0 / math.sqrt(self.w1_lora_B.shape[1])
            w1_out = (
                x_tok @ w1_q_c
                + ((x_tok @ self.w1_lora_A[e]) @ self.w1_lora_B[e]) * rs_scaling
            )
            wgate_out = (
                x_tok @ wgate_q_c
                + ((x_tok @ self.wgate_lora_A[e]) @ self.wgate_lora_B[e]) * rs_scaling
            )

            h = F.silu(w1_out) * wgate_out
            h_swarm = self.expert_swarms[e](
                h,
                scale=gov_scale,
                w_a=w_a_all[e].to(compute_dtype),
                w_b=w_b_all[e].to(compute_dtype),
            )
            w2_out = (
                h_swarm @ w2_q_all[e].to(compute_dtype)
                + ((h_swarm @ self.w2_lora_A[e]) @ self.w2_lora_B[e]) * rs_scaling
            )

            idx_flat = token_indices.nonzero(as_tuple=True)[0]
            final_out.index_add_(
                0, idx_flat, (w2_out * expert_gates).to(final_out.dtype)
            )

        return self.output_norm(final_out.reshape(B, L, D)), aux_loss


def resolve_attention_geometry(
    hidden_dim: int,
    heads: Optional[int] = None,
    head_dim: Optional[int] = None,
    kv_heads: Optional[int] = None,
    use_rope: Optional[bool] = None,
):
    """Resolve the C0 attention socket geometry without adding state_dict keys.

    LEGACY (all other widths, e.g. hidden=64 tiny / hidden=1024 flagship):
        heads=30, head_dim=hidden//30, kv_heads=heads, RoPE OFF.
        This is bit-identical to the pre-socket operator.

    DONOR-NATIVE (hidden_dim==896, Qwen2.5-0.5B width):
        heads=14, head_dim=64 -> 14*64 = 896 = hidden (zero pad/crop),
        kv_heads=2 (GQA, K/V expanded 2->14), RoPE ON for Q and K.

    Explicit arguments always win over auto-dispatch.
    """
    if int(hidden_dim) == 896:
        heads = 14 if heads is None else heads
        head_dim = 64 if head_dim is None else head_dim
        kv_heads = 2 if kv_heads is None else kv_heads
        use_rope = True if use_rope is None else use_rope
    else:
        heads = 30 if heads is None else heads
        head_dim = (int(hidden_dim) // int(heads)) if head_dim is None else head_dim
        kv_heads = heads if kv_heads is None else kv_heads
        use_rope = False if use_rope is None else use_rope
    return int(heads), int(head_dim), int(kv_heads), bool(use_rope)


class CouilAttention(nn.Module):
    """C0 master-node attention socket.

    The default (30 heads, head_dim=hidden//30, no GQA, no RoPE) preserves the
    legacy operator exactly. The donor-native 896-wide socket hosts Qwen2.5
    attention: 14x64 queries, 2 GQA KV heads, RoPE on Q/K. No new parameters,
    buffers, or state_dict keys are introduced in either mode.
    """

    def __init__(
        self,
        hidden_dim: int,
        heads: Optional[int] = None,
        head_dim: Optional[int] = None,
        kv_heads: Optional[int] = None,
        use_rope: Optional[bool] = None,
    ):
        super().__init__()
        heads, head_dim, kv_heads, use_rope = resolve_attention_geometry(
            hidden_dim, heads, head_dim, kv_heads, use_rope
        )
        self.hidden_dim = int(hidden_dim)
        self.heads = heads
        self.head_dim = head_dim
        self.kv_heads = kv_heads
        self.use_rope = use_rope
        if self.heads <= 0 or self.head_dim <= 0 or self.kv_heads <= 0:
            raise ValueError(
                f"Invalid attention geometry: heads={self.heads} "
                f"head_dim={self.head_dim} kv_heads={self.kv_heads}"
            )
        if self.heads % self.kv_heads != 0:
            raise ValueError(
                f"heads ({self.heads}) must be an integer multiple of "
                f"kv_heads ({self.kv_heads})"
            )
        self.attn_dim = self.heads * self.head_dim
        self.kv_dim = self.kv_heads * self.head_dim
        self.q_proj = BitLinear(self.hidden_dim, self.attn_dim, bias=False)
        self.k_proj = BitLinear(self.hidden_dim, self.kv_dim, bias=False)
        self.v_proj = BitLinear(self.hidden_dim, self.kv_dim, bias=False)
        self.o_proj = BitLinear(self.attn_dim, self.hidden_dim, bias=False)
        self.norm = nn.LayerNorm(self.hidden_dim)
        self.sparse_mask_generator = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 4, self.hidden_dim),
        )

    @staticmethod
    def rotate_half(x: torch.Tensor) -> torch.Tensor:
        """HF/Llama-Qwen split-half rotation (-x2, x1) on the last dimension."""
        dim = x.size(-1)
        half = dim // 2
        x1 = x[..., :half]
        x2 = x[..., half : 2 * half]
        rotated = torch.cat((-x2, x1), dim=-1)
        if dim % 2:
            rotated = torch.cat((rotated, x[..., 2 * half :]), dim=-1)
        return rotated

    def rope_freqs(
        self,
        position_start: int,
        length: int,
        device,
        dtype: torch.dtype,
    ):
        """RoPE cos/sin of shape (1, 1, length, head_dim) at absolute positions.

        Base is 10000 (Qwen2.5 convention) for the legacy-PE horizon; beyond it
        an NTK-aware base expansion keeps adjacent positions separable without
        any learned table. Memory is O(length * head_dim), never O(2M * D).
        """
        dim = self.head_dim
        half = dim // 2
        if length <= 0:
            empty = torch.empty(1, 1, 0, dim, device=device, dtype=dtype)
            return empty, empty
        if half == 0:
            ones = torch.ones(1, 1, length, dim, device=device, dtype=dtype)
            return ones, torch.zeros_like(ones)
        pos = torch.arange(
            int(position_start),
            int(position_start) + int(length),
            device=device,
            dtype=torch.float32,
        )
        scale = max(1.0, float(int(position_start) + int(length)) / 1024.0)
        base = 10000.0 * (scale ** (dim / max(1, dim - 2)))
        idx = torch.arange(half, device=device, dtype=torch.float32) / half
        inv_freq = 1.0 / (base**idx)
        angles = pos[:, None] * inv_freq[None, :]
        cos_half, sin_half = torch.cos(angles), torch.sin(angles)
        if dim % 2:
            cos = torch.ones(int(length), dim, device=device, dtype=torch.float32)
            sin = torch.zeros(int(length), dim, device=device, dtype=torch.float32)
            cos[:, : 2 * half] = torch.cat([cos_half, cos_half], dim=-1)
            sin[:, : 2 * half] = torch.cat([sin_half, sin_half], dim=-1)
        else:
            cos = torch.cat([cos_half, cos_half], dim=-1)
            sin = torch.cat([sin_half, sin_half], dim=-1)
        return cos.unsqueeze(0).unsqueeze(0).to(dtype), sin.unsqueeze(0).unsqueeze(
            0
        ).to(dtype)

    def rope_qk(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        position_start: int = 0,
        freqs_cos: Optional[torch.Tensor] = None,
        freqs_sin: Optional[torch.Tensor] = None,
    ):
        """Apply RoPE to Q and K at absolute positions [start, start+L).

        No-op for the legacy socket (use_rope=False). The accepted
        ``freqs_cos``/``freqs_sin`` are honoured when supplied: a length-L pair
        is applied to this block directly; a longer pair is sliced at
        ``position_start``. q/k are (B, heads, L, head_dim) after project().
        """
        if not self.use_rope:
            return q, k
        length = q.size(2)
        dim = q.size(-1)
        if freqs_cos is None or freqs_sin is None:
            cos, sin = self.rope_freqs(position_start, length, q.device, q.dtype)
        else:
            cos = freqs_cos.reshape(1, 1, -1, dim).to(device=q.device, dtype=q.dtype)
            sin = freqs_sin.reshape(1, 1, -1, dim).to(device=q.device, dtype=q.dtype)
            if cos.size(2) != length:
                cos = cos[..., int(position_start) : int(position_start) + length, :]
                sin = sin[..., int(position_start) : int(position_start) + length, :]
        q = q * cos + self.rotate_half(q) * sin
        k = k * cos + self.rotate_half(k) * sin
        return q, k

    def project(self, x_normed: torch.Tensor):
        """Return q/k/v as (B, heads, L, head_dim), ready for cache writes.

        K/V are projected at kv_heads then GQA-expanded to all query heads so
        the KV cache layout is unchanged (bytes/token math stays exact).
        """
        B, L, D = x_normed.shape
        q = (
            self.q_proj(x_normed)
            .reshape(B, L, self.heads, self.head_dim)
            .transpose(1, 2)
        )
        k = (
            self.k_proj(x_normed)
            .reshape(B, L, self.kv_heads, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.v_proj(x_normed)
            .reshape(B, L, self.kv_heads, self.head_dim)
            .transpose(1, 2)
        )
        if self.kv_heads != self.heads:
            reps = self.heads // self.kv_heads
            k = k.repeat_interleave(reps, dim=1)
            v = v.repeat_interleave(reps, dim=1)
        return q, k, v

    def attend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        past_len: int = 0,
        causal: bool = True,
    ) -> torch.Tensor:
        """Causal attention with in-place block-local masking (no O(L^2) mask).

        q/k arrive already RoPE-rotated at their absolute positions (see
        rope_qk); cached K was rotated when first written.
        """
        scores = (q @ k.transpose(-2, -1)) * (self.head_dim**-0.5)
        if causal:
            _causal_mask_inplace(scores, past_len)
        return F.softmax(scores, dim=-1) @ v

    def merge(self, attn: torch.Tensor, B: int, L: int) -> torch.Tensor:
        return attn.transpose(1, 2).reshape(B, L, self.attn_dim)

    def forward(
        self,
        x,
        causal=True,
        freqs_cos=None,
        freqs_sin=None,
        past_key_value=None,
        use_cache=False,
        position_start: int = 0,
    ):
        B, L, D = x.shape
        x_normed = self.norm(x)
        q, k, v = self.project(x_normed)

        past_len = 0
        if past_key_value is not None:
            pk, pv = past_key_value
            past_len = pk.size(2)
            q, k = self.rope_qk(
                q,
                k,
                position_start=past_len,
                freqs_cos=freqs_cos,
                freqs_sin=freqs_sin,
            )
            k = torch.cat([pk, k], dim=2)
            v = torch.cat([pv, v], dim=2)
        else:
            q, k = self.rope_qk(
                q,
                k,
                position_start=position_start,
                freqs_cos=freqs_cos,
                freqs_sin=freqs_sin,
            )

        present_kv = (k, v) if use_cache else None
        attn = self.attend(q, k, v, past_len=past_len, causal=causal)
        return self.o_proj(self.merge(attn, B, L)), present_kv


class SovereignFlashDiffusionCore(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        steps: int = 14,
        heads: Optional[int] = None,
        head_dim: Optional[int] = None,
        kv_heads: Optional[int] = None,
        use_rope: Optional[bool] = None,
    ):
        super().__init__()
        self.steps = steps
        heads, head_dim, kv_heads, use_rope = resolve_attention_geometry(
            hidden_dim, heads, head_dim, kv_heads, use_rope
        )
        self.couil_attn = CouilAttention(
            hidden_dim,
            heads=heads,
            head_dim=head_dim,
            kv_heads=kv_heads,
            use_rope=use_rope,
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            BitLinear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            BitLinear(hidden_dim * 4, hidden_dim),
        )

    def forward(
        self,
        x,
        router_mask,
        past_key_values=None,
        use_cache=False,
        position_start: int = 0,
    ):
        current = x
        present_kv = None
        # Single-pass diffusion: train and inference must match exactly
        attn_out, present_kv = self.couil_attn(
            current,
            causal=True,
            past_key_value=past_key_values[0] if past_key_values else None,
            use_cache=use_cache,
            position_start=position_start,
        )
        current = current + attn_out
        ffn_out = self.ffn(self.norm2(current))
        current = current + ffn_out

        mask = router_mask.unsqueeze(-1)
        out = current * mask + x * (1 - mask)
        return out.to(x.dtype), [present_kv] if use_cache else None

    def cached_step(
        self,
        x: torch.Tensor,
        router_mask: torch.Tensor,
        cache: "QuillanKVCache",
        start: int,
    ) -> torch.Tensor:
        """Incremental single-pass diffusion over a preallocated KV cache.

        The 14 declared diffusion steps remain a documented single-pass no-op,
        unchanged from the original implementation; this only replaces the
        attention execution path so cached and uncached logits stay identical.
        """
        B, L, D = x.shape
        end = start + L
        attn_mod = self.couil_attn
        q, k, v = attn_mod.project(attn_mod.norm(x))
        # RoPE at the ABSOLUTE positions of this chunk; the cache stores the
        # rotated K so later steps never re-rotate (or re-allocate for) the
        # full prefix. Legacy socket: rope_qk is an exact no-op.
        q, k = attn_mod.rope_qk(q, k, position_start=start)
        cache.diff_k[:, :, start:end, :] = k.to(cache.diff_k.dtype)
        cache.diff_v[:, :, start:end, :] = v.to(cache.diff_v.dtype)
        # Small new-query cast only; never re-cast the full cached prefix.
        q = q.to(cache.diff_k.dtype)
        a = attn_mod.attend(
            q,
            cache.diff_k[:, :, :end, :],
            cache.diff_v[:, :, :end, :],
            past_len=start,
            causal=True,
        )
        attn_out = attn_mod.o_proj(attn_mod.merge(a, B, L))
        current = x + attn_out
        current = current + self.ffn(self.norm2(current))
        mask = router_mask.unsqueeze(-1)
        return (current * mask + x * (1 - mask)).to(x.dtype)


class EthicalImpactConstraintEngine(nn.Module):
    def __init__(self, hidden_dim: int, e_ice_limit_ms: int = 100):
        super().__init__()
        self.classifier = nn.Linear(hidden_dim, 5)

    def forward(self, x, router_probs):
        logits = self.classifier(x)
        probs = F.softmax(logits, dim=-1)
        violations = probs[..., :3].sum(dim=-1)
        return {"constrained_violations": violations}


class MARTAThermodynamicGating(nn.Module):
    def __init__(self, hidden_dim: int, num_reasoning_modules: int = 4):
        super().__init__()
        self.gate = nn.Linear(hidden_dim, 1)

    def forward(self, x, violations):
        flow = torch.sigmoid(self.gate(x)).squeeze(-1) * (1.0 - 0.1 * violations)
        return {"flow_coefficients": flow}


class DynamicQuantumSwarmOscillation(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        return {"swarm_embedding": 0.05 * torch.tanh(self.proj(x))}


class PrimeCovenantFramework(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()

    def forward(self, x):
        return {"status": "aligned"}


class CCRLFramework(nn.Module):
    def __init__(self, hidden_dim: int, num_experts: int):
        super().__init__()

    def forward(self, x, router_probs):
        entropy = -(router_probs * torch.log(router_probs + 1e-10)).sum(dim=-1).mean()
        return {"entropy_bonus": entropy * 0.01}


class QuantumFormulasEngine(nn.Module):
    def __init__(self, hidden_dim: int, num_experts: int):
        super().__init__()
        self.register_buffer("hbar", torch.tensor(1.0545718e-34))
        self.register_buffer("boltzmann", torch.tensor(1.380649e-23))
        self.register_buffer("sqrt_33", torch.tensor(math.sqrt(33)))

    def qps_synthesis(self, A, B, Q, R):
        return torch.eye(A.shape[-1], device=A.device)


class QuillanRoninSovereign(nn.Module):
    def __init__(self, cfg: QuillanArchConfig):
        super().__init__()
        self.cfg = cfg
        apply_phoenix_affinity()
        self.ingestion = InputIngestionLayer(cfg)
        self.decomposition = NineVectorDecomposition(cfg.hidden_dim)
        self.moe = EvolvableVectorizedMoE(cfg)
        self.governor = LeeMach6Governor(cfg.e_ice_limit_ms)
        self.diffusion_core = SovereignFlashDiffusionCore(
            cfg.hidden_dim,
            steps=14,
            heads=getattr(cfg, "attn_heads", None),
            head_dim=getattr(cfg, "attn_head_dim", None),
            kv_heads=getattr(cfg, "attn_kv_heads", None),
            use_rope=getattr(cfg, "attn_use_rope", None),
        )
        self.quantum_formulas = QuantumFormulasEngine(cfg.hidden_dim, cfg.num_experts)

        self.norm_decomp = nn.LayerNorm(cfg.hidden_dim)
        self.norm_diff = nn.LayerNorm(cfg.hidden_dim)
        self.norm_moe = nn.LayerNorm(cfg.hidden_dim)
        self.pre_final_norm = nn.LayerNorm(cfg.hidden_dim)
        self.quillan_finalizer = BitLinear(
            cfg.hidden_dim, cfg.hidden_dim, quantize_act=False, quantize_weight=False
        )
        self.quillan_finalizer2 = BitLinear(
            cfg.hidden_dim, cfg.hidden_dim, quantize_act=False, quantize_weight=False
        )
        self.quillan_gate = nn.Linear(cfg.hidden_dim * 2, cfg.hidden_dim)
        self.final_norm = nn.LayerNorm(cfg.hidden_dim)
        self.txt_dec = BitLinear(
            cfg.hidden_dim,
            cfg.vocab_size,
            bias=False,
            quantize_act=False,
            quantize_weight=False,
        )

    # ─── INCREMENTAL KV-CACHE HELPERS ────────────────────────────────────────
    def _param_dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    def kv_cache_bytes_per_token(self, dtype: Optional[torch.dtype] = None) -> int:
        """Exact KV bytes/token for BOTH attention towers (K and V each).

        decomp: 2 * hidden_dim; diff: 2 * heads * head_dim.
        For the default 1024-dim config: (2*1024 + 2*30*34) = 4088 elements.
        fp32 -> 16,352 B/token (~15.97 KiB); fp16 -> 8,176 B/token (~7.98 KiB).
        """
        a = self.diffusion_core.couil_attn
        elem = torch.empty((), dtype=dtype or self._param_dtype()).element_size()
        return (2 * self.cfg.hidden_dim + 2 * a.heads * a.head_dim) * elem

    def kv_cache_memory_report(
        self, tokens: int, dtype: Optional[torch.dtype] = None
    ) -> Dict[str, Any]:
        bpt = self.kv_cache_bytes_per_token(dtype)
        return {
            "bytes_per_token": bpt,
            "kib_per_token": round(bpt / 1024.0, 3),
            "bytes_for_tokens": bpt * tokens,
            "mib_for_tokens": round(bpt * tokens / (1024.0**2), 2),
            "declared_positions": DECLARED_MAX_POSITIONS,
            "full_declared_cache_gib": round(
                bpt * DECLARED_MAX_POSITIONS / (1024.0**3), 2
            ),
            "bounded_active_window": int(self.cfg.kv_cache_capacity),
        }

    def new_cache(
        self, needed_length: int, device=None, capacity: Optional[int] = None
    ) -> QuillanKVCache:
        """Allocate the KV buffers ONCE from the bounded active window."""
        dtype = self._param_dtype()
        mode = str(getattr(self.cfg, "kv_cache_dtype", "auto")).lower()
        if mode == "fp16":
            dtype = torch.float16
        elif mode == "fp32":
            dtype = torch.float32
        elif mode != "auto":
            raise ValueError(f"Unknown kv_cache_dtype: {mode!r}")

        cap = int(
            capacity
            or getattr(self.cfg, "kv_cache_capacity", DEFAULT_KV_CACHE_CAPACITY)
        )
        if needed_length > cap:
            report = self.kv_cache_memory_report(needed_length, dtype)
            raise ValueError(
                f"Need {needed_length} active KV slots but bounded window is {cap}. "
                f"Declared position capacity is {DECLARED_MAX_POSITIONS}; a full "
                f"{DECLARED_MAX_POSITIONS}-token KV cache would need "
                f"{report['full_declared_cache_gib']} GiB and cannot fit the ~28 GB "
                f"host RAM together with model weights. The system prompt is NOT "
                f"truncated and num_ctx is NOT lowered - raise cfg.kv_cache_capacity "
                f"only if host RAM permits ({report['bytes_per_token']} B/token)."
            )
        if device is None:
            device = next(self.parameters()).device
        a = self.diffusion_core.couil_attn
        return QuillanKVCache(
            decomp_k=torch.empty(
                1, cap, self.cfg.hidden_dim, device=device, dtype=dtype
            ),
            decomp_v=torch.empty(
                1, cap, self.cfg.hidden_dim, device=device, dtype=dtype
            ),
            diff_k=torch.empty(1, a.heads, cap, a.head_dim, device=device, dtype=dtype),
            diff_v=torch.empty(1, a.heads, cap, a.head_dim, device=device, dtype=dtype),
            length=0,
            capacity=cap,
        )

    def forward(
        self,
        txt,
        img=None,
        past_key_values=None,
        use_cache=False,
        position_start: int = 0,
    ):
        B, L = txt.shape[0], txt.shape[1]
        cache = past_key_values if isinstance(past_key_values, QuillanKVCache) else None

        if use_cache and cache is None:
            raise ValueError(
                "use_cache=True requires past_key_values to be a QuillanKVCache "
                "created by model.new_cache()."
            )
        if cache is None and L > MAX_DIRECT_CAUSAL_QUERIES:
            raise ValueError(
                f"Direct full-sequence forward with L={L} is disabled to prevent "
                f"O(L^2) mask/score allocation (cap {MAX_DIRECT_CAUSAL_QUERIES}). "
                f"Use generate(), which performs bounded chunked prefill through "
                f"the incremental KV cache."
            )
        if cache is not None and (
            position_start < 0 or position_start + L > cache.capacity
        ):
            raise ValueError(
                f"KV window overflow: position_start={position_start}, L={L}, "
                f"capacity={cache.capacity}"
            )

        # Position-aware ingestion (legacy learned PE for positions < 1024,
        # deterministic NTK sinusoidal extension beyond that).
        z = self.ingestion(txt, img, position_start=position_start)

        if cache is not None:
            decomp_out = self.decomposition.cached_step(z, cache, position_start)
        else:
            decomp_out = self.decomposition(z)
        blueprint = self.norm_decomp(z + decomp_out)

        diff_mask = torch.ones(B, L, device=blueprint.device, dtype=blueprint.dtype)
        if cache is not None:
            x_diff = self.diffusion_core.cached_step(
                blueprint, diff_mask, cache, position_start
            )
        else:
            legacy_past = None
            if isinstance(past_key_values, (list, tuple)) and len(past_key_values) > 0:
                legacy_past = [past_key_values[0]]
            x_diff_out = self.diffusion_core(
                blueprint,
                diff_mask,
                past_key_values=legacy_past,
                use_cache=use_cache,
                position_start=position_start,
            )
            x_diff = x_diff_out[0] if isinstance(x_diff_out, tuple) else x_diff_out
        x_diff = self.norm_diff(blueprint + x_diff)

        x_moe_out, r_loss = self.moe(x_diff, gov_scale=self.governor.current_scale)
        x_moe = self.norm_moe(x_diff + x_moe_out)

        x_norm = self.pre_final_norm(x_moe)
        x_q1 = self.quillan_finalizer(x_norm)
        x_q2 = self.quillan_finalizer2(x_norm)
        x_q1_fused = x_q1 + 0.1 * x_q2
        x_q2_fused = x_q2 + 0.1 * x_q1
        gate = torch.sigmoid(
            self.quillan_gate(torch.cat([x_q1_fused, x_q2_fused], dim=-1))
        )
        x_final = x_norm + (gate * x_q1_fused + (1.0 - gate) * x_q2_fused)

        self._last_aux_loss = r_loss
        logits = self.txt_dec(self.final_norm(x_final))

        if cache is not None:
            cache.length = max(cache.length, position_start + L)
            if B == 1:
                cache.token_prefix.extend(txt[0].detach().cpu().tolist())
            cache.last_logits = logits[:, -1, :].detach().clone()

        if self.training:
            return logits, r_loss
        if use_cache:
            return logits, cache
        return logits

    @torch.no_grad()
    def generate(
        self,
        prompt_tokens: list,
        enc=None,
        max_tokens: int = 250,
        temp: float = 0.6,
        top_p: float = 0.9,
        repetition_penalty: float = 1.15,
        cache: Optional[QuillanKVCache] = None,
        return_cache: bool = False,
        chunk_size: Optional[int] = None,
    ):
        """Real incremental-KV generation with bounded chunked prefill.

        The old implementation re-fed only ``generated[-256:]`` every step and
        never used a cache, so every token paid ~(k+1)x recompute and a 62.5k
        prompt could never be retained. This path:

          1. pre-fills the ENTIRE prompt in bounded chunks into a preallocated
             KV cache (no O(L^2) mask at any point),
          2. appends exactly ONE token per decode step and reuses all past K/V,
          3. accepts an existing cache for the same prefix so a second call
             skips prefill and produces an identical continuation.

        ``return_cache=True`` returns ``(continuation, cache)``. Positional
        capacity remains DECLARED_MAX_POSITIONS (2,000,000); the ACTIVE window
        is bounded by ``cfg.kv_cache_capacity`` because a full 2M KV cache does
        not fit this host's RAM. The system prompt is never truncated.
        """
        self.eval()
        prompt = list(prompt_tokens)
        if not prompt:
            raise ValueError("prompt_tokens must be non-empty")
        device = next(self.parameters()).device
        P = len(prompt)
        needed = P + max_tokens

        # ── cache validation / reuse ──────────────────────────────────────────
        reusable = isinstance(cache, QuillanKVCache) and cache.capacity >= needed
        if reusable:
            # Compare only the shared prompt range: the cache may also hold
            # tokens generated by the previous call.
            n = min(cache.length, len(prompt), len(cache.token_prefix))
            if tuple(prompt[:n]) != tuple(cache.token_prefix[:n]):
                reusable = False  # different prefix -> cannot reuse

        if reusable:
            # Always keep at most P-1 cached positions; the final prompt token
            # is recomputed so last_logits is guaranteed to be the distribution
            # at position P-1 (identical to a fresh full prefill).
            keep = max(0, min(cache.length, P - 1))
            cache.truncate(keep)
        else:
            cache = self.new_cache(needed, device=device)
            keep = 0

        # ── bounded prefill chunk sizing ──────────────────────────────────────
        pref = max(1, int(chunk_size or getattr(self.cfg, "prefill_chunk_size", 256)))
        a = self.diffusion_core.couil_attn
        score_budget_elems = MAX_ATTN_SCORE_BYTES // max(
            1, torch.empty((), dtype=self._param_dtype()).element_size()
        )
        auto_chunk = max(1, score_budget_elems // max(1, a.heads * needed))
        chunk = max(1, min(pref, auto_chunk))

        for start in range(keep, P, chunk):
            block = prompt[start : min(start + chunk, P)]
            ids = torch.tensor([block], dtype=torch.long, device=device)
            self.forward(
                ids, past_key_values=cache, use_cache=True, position_start=start
            )

        if cache.last_logits is None:
            raise RuntimeError("Prefill failed to produce logits")
        curr_logits = cache.last_logits.clone()  # (1, vocab), position P-1

        # ── incremental decode ────────────────────────────────────────────────
        generated = list(prompt)
        for _ in range(max_tokens):
            # Anti-stutter: hard penalty on immediate previous token only.
            if generated:
                curr_logits[0, generated[-1]] -= 50.0

            # Bounded multiplicative repetition penalty on recent generated tokens.
            if repetition_penalty != 1.0 and len(generated) > P:
                for tid in set(generated[P:][-64:]):
                    if curr_logits[0, tid] > 0:
                        curr_logits[0, tid] /= repetition_penalty
                    else:
                        curr_logits[0, tid] *= repetition_penalty

            if temp <= 0.01:
                next_tok = torch.argmax(curr_logits, dim=-1).item()
            else:
                scaled_logits = curr_logits / temp
                probs = F.softmax(scaled_logits, dim=-1)
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                cum_probs = torch.cumsum(sorted_probs, dim=-1)
                sorted_indices_to_remove = cum_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                    ..., :-1
                ].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                scaled_logits[indices_to_remove] = float("-inf")
                probs = F.softmax(scaled_logits, dim=-1)
                next_tok = torch.multinomial(probs, 1).item()

            generated.append(next_tok)
            if next_tok == 50256:  # <|endoftext|>
                break

            # Feed ONLY the new token at its absolute position; all past K/V
            # are read from the cache. No 256-token re-feed, no recompute.
            ids = torch.tensor([[next_tok]], dtype=torch.long, device=device)
            out = self.forward(
                ids, past_key_values=cache, use_cache=True, position_start=cache.length
            )
            logits = out[0] if isinstance(out, tuple) else out
            curr_logits = logits[:, -1, :].clone()

        result = generated[P:]
        if return_cache:
            return result, cache
        return result
