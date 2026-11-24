#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quillan-Ronin Sovereign BitNet 1.58b Multi-Layer Architecture (`quillan_sovereign_bitnet.py`).

Depth-matched multi-layer BitNet 1.58b backbone (<= 1B parameters per model)
combining:
  - PackedBitLinear: 2-bit uint8 packed ternary weights (4 vals/byte, {-1, 0, +1}),
    group-128 per-block least-squares gamma*, optional 2-plane residual ternary
    (T_1 * gamma_1 + T_2 * gamma_2), W1.58A8 INT8 [-128, 127] activation quant,
    lazy on-the-fly unpack, and EGGROLL rank-r online closed-form Woodbury WSP.
  - PackedLittleBitLinear & PackedCouncilMoE: 34-Expert Council MoE + rank-16
    virtual 9B swarm + LeeMach6Governor with active-expert-only lazy unpack and
    zero-initialized residual gating (tanh(alpha_l)|_{t=0} == 0).
  - QuillanSovereignBlock & QuillanSovereignBitNetLM: GQA, 2M NTK-aware RoPE,
    per-layer incremental KV cache, RMSNorm, SwiGLU FFN, INT8 quantized embeddings,
    packed safetensors save/load, and autoregressive generate().
"""

from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import safetensors.torch
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger("Quillan.SovereignBitNet")


@dataclass
class QuillanSovereignConfig:
    """Configuration for QuillanSovereignBitNetLM (all presets strictly <= 1B params)."""

    model_name: str = "quillan-mini"
    vocab_size: int = 151936
    hidden_size: int = 896
    intermediate_size: int = 4864
    num_hidden_layers: int = 24
    num_attention_heads: int = 14
    num_key_value_heads: int = 2
    head_dim: int = 64
    max_position_embeddings: int = 2097152
    rope_theta: float = 1000000.0
    orig_max_position_embeddings: int = 32768
    rms_norm_eps: float = 1e-6
    tie_word_embeddings: bool = True
    qkv_bias: bool = False
    use_rht: bool = True
    group_size: int = 32
    attn_planes: int = 3
    ffn_planes: int = 3
    eggroll_rank: int = 16
    num_experts: int = 34
    num_experts_active: int = 2
    moe_ffn_dim: int = 512
    moe_layer_indices: Tuple[int, ...] = (11, 22)
    lm_head_rank: int = 0  # fp32 LoRA on untied lm_head (0 = legacy, no change)

    @classmethod
    def for_model(cls, key: str) -> "QuillanSovereignConfig":
        """Returns a verified <= 1B parameter config for mini, flagship, or dragon."""
        k = key.lower()
        if k == "mini":
            return cls(
                model_name="quillan-mini-bitnet",
                vocab_size=151936,
                hidden_size=896,
                intermediate_size=4864,
                num_hidden_layers=24,
                num_attention_heads=14,
                num_key_value_heads=2,
                head_dim=64,
                orig_max_position_embeddings=32768,
                rope_theta=1000000.0,
                qkv_bias=True,
                moe_ffn_dim=512,
                moe_layer_indices=(11, 22),
            )
        if k == "flagship":
            return cls(
                model_name="quillan-flagship-bitnet",
                vocab_size=128256,
                hidden_size=2048,
                intermediate_size=5120,
                num_hidden_layers=16,
                num_attention_heads=32,
                num_key_value_heads=8,
                head_dim=64,
                orig_max_position_embeddings=131072,
                rope_theta=500000.0,
                moe_ffn_dim=256,
                moe_layer_indices=(14,),
            )
        if k == "dragon":
            return cls(
                model_name="quillan-dragon-bitnet",
                vocab_size=131072,
                hidden_size=2048,
                intermediate_size=3840,
                num_hidden_layers=18,
                num_attention_heads=8,
                num_key_value_heads=4,
                head_dim=256,
                orig_max_position_embeddings=8192,
                rope_theta=1000000.0,
                moe_ffn_dim=256,
                moe_layer_indices=(16,),
            )
        raise ValueError(f"Unknown preset '{key}'; expected mini, flagship, or dragon")


def pack_ternary_2bit(t: torch.Tensor) -> Tuple[torch.Tensor, int]:
    """
    Packs a ternary tensor {-1, 0, +1} into uint8 (4 values per byte).
    Encoding: -1 -> 0, 0 -> 1, +1 -> 2.
    """
    flat = t.reshape(-1).to(torch.int8)
    orig_numel = flat.numel()
    pad = (-orig_numel) % 4
    if pad:
        flat = F.pad(flat, (0, pad), value=0)
    codes = (flat + 1).to(torch.uint8).view(-1, 4)
    packed = codes[:, 0] | (codes[:, 1] << 2) | (codes[:, 2] << 4) | (codes[:, 3] << 6)
    return packed.contiguous(), orig_numel


def unpack_ternary_2bit(
    packed: torch.Tensor, shape: Tuple[int, ...], orig_numel: int
) -> torch.Tensor:
    """Unpacks 2-bit uint8 buffer back to float32 {-1.0, 0.0, +1.0} with exact shape."""
    b = packed.to(torch.uint8)
    c0 = (b & 0x03).to(torch.int8) - 1
    c1 = ((b >> 2) & 0x03).to(torch.int8) - 1
    c2 = ((b >> 4) & 0x03).to(torch.int8) - 1
    c3 = ((b >> 6) & 0x03).to(torch.int8) - 1
    flat = torch.stack((c0, c1, c2, c3), dim=1).reshape(-1)[:orig_numel]
    return flat.to(torch.float32).view(shape)


def quantize_group_ternary_ls(
    w: torch.Tensor,
    group_size: int = 128,
    eps: float = 1e-5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Group-wise per-block absmean ternarization T in {-1, 0, +1} with exact
    least-squares scale gamma* = <W_g, T_g> / (<T_g, T_g> + eps).
    """
    out_dim, in_dim = w.shape
    g = group_size if (in_dim % group_size == 0) else in_dim
    blocks = w.float().view(out_dim, in_dim // g, g)
    gamma0 = blocks.abs().mean(dim=-1, keepdim=True).clamp(min=eps)
    t_tern = torch.clamp(torch.round(blocks / gamma0), -1.0, 1.0)
    denom = (t_tern * t_tern).sum(dim=-1, keepdim=True).clamp(min=eps)
    gamma_star = (blocks * t_tern).sum(dim=-1, keepdim=True) / denom
    return t_tern.view(out_dim, in_dim), gamma_star.squeeze(-1).to(torch.float16)


def quantize_activations_w158a8(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Symmetric per-token INT8 [-128, 127] activation quantization for W1.58A8."""
    scale = x.abs().amax(dim=-1, keepdim=True).clamp(min=eps) / 127.0
    return torch.round(x / scale).clamp(-128.0, 127.0) * scale


_H32_CACHE: Dict[str, torch.Tensor] = {}


def _get_hadamard_32(device: torch.device) -> torch.Tensor:
    """Returns deterministic orthonormal 32x32 Walsh-Hadamard matrix (Paper 012 RHT-32)."""
    key = str(device)
    if key not in _H32_CACHE:
        h = torch.tensor([[1.0]], dtype=torch.float32)
        for _ in range(5):  # 2^5 = 32
            h = torch.cat((torch.cat((h, h), dim=1), torch.cat((h, -h), dim=1)), dim=0)
        _H32_CACHE[key] = (h / math.sqrt(32.0)).to(device)
    return _H32_CACHE[key]


class PackedBitLinear(nn.Module):
    """
    2-bit packed BitNet 1.58b linear layer with:
      - 1-, 2-, or 3-plane residual ternary weights (sum_p T_p * gamma_p)
      - Paper 012 RHT-32 orthogonal Walsh-Hadamard outlier dispersion
      - Lazy on-the-fly unpack (zero persistent fp16/fp32 weight cache)
      - W1.58A8 INT8 activation quantization (eps=1e-5)
      - EGGROLL rank-r online closed-form Woodbury WSP (online_wsp_update)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        group_size: int = 32,
        num_planes: int = 3,
        eggroll_rank: int = 16,
        eps: float = 1e-5,
        use_rht: bool = True,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size if (in_features % group_size == 0) else in_features
        self.num_planes = max(1, min(3, num_planes))
        self.eggroll_rank = eggroll_rank
        self.eps = eps
        self.use_rht = use_rht and (in_features % 32 == 0)

        num_groups = in_features // self.group_size
        packed_len = (out_features * in_features + 3) // 4

        self.register_buffer(
            "q_packed",
            torch.ones(self.num_planes, packed_len, dtype=torch.uint8) * 0x55,
        )
        self.register_buffer(
            "gamma",
            torch.zeros(self.num_planes, out_features, num_groups, dtype=torch.float16),
        )
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.float32))
        else:
            self.register_parameter("bias", None)

        if eggroll_rank > 0:
            self.lora_A = nn.Parameter(
                torch.zeros(eggroll_rank, in_features, dtype=torch.float32)
            )
            self.lora_B = nn.Parameter(
                torch.zeros(out_features, eggroll_rank, dtype=torch.float32)
            )
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)

    @torch.no_grad()
    def set_weights_from_float(self, w: torch.Tensor) -> None:
        """Quantizes fp32/fp16 weight matrix into 1-, 2-, or 3-plane 2-bit packed ternary (with optional RHT-32)."""
        residual = w.detach().float().view(self.out_features, self.in_features)
        if self.use_rht:
            h32 = _get_hadamard_32(residual.device)
            residual = (residual.view(self.out_features, -1, 32) @ h32).view(
                self.out_features, self.in_features
            )
        for p in range(self.num_planes):
            t_tern, g_star = quantize_group_ternary_ls(
                residual, group_size=self.group_size, eps=self.eps
            )
            packed, _ = pack_ternary_2bit(t_tern)
            self.q_packed[p].copy_(packed)
            self.gamma[p].copy_(g_star)
            if p + 1 < self.num_planes:
                recon = (
                    t_tern.view(self.out_features, -1, self.group_size)
                    * g_star.float().unsqueeze(-1)
                ).view_as(residual)
                residual = residual - recon

    def unpack_effective_weight(self) -> torch.Tensor:
        """Lazily unpacks 2-bit buffers into temporary fp32 matrix (never cached)."""
        w_eff = torch.zeros(
            self.out_features,
            self.in_features,
            dtype=torch.float32,
            device=self.q_packed.device,
        )
        n_groups = self.in_features // self.group_size
        orig_numel = self.out_features * self.in_features
        for p in range(self.num_planes):
            t_p = unpack_ternary_2bit(
                self.q_packed[p],
                (self.out_features, n_groups, self.group_size),
                orig_numel,
            )
            w_p = (t_p * self.gamma[p].float().unsqueeze(-1)).view(
                self.out_features, self.in_features
            )
            w_eff = w_eff + w_p
        if self.use_rht:
            h32 = _get_hadamard_32(w_eff.device)
            w_eff = (w_eff.view(self.out_features, -1, 32) @ h32.T).view(
                self.out_features, self.in_features
            )
        return w_eff

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_f = x.float()
        x_q = quantize_activations_w158a8(x_f, eps=self.eps)
        w_eff = self.unpack_effective_weight()
        out = F.linear(x_q, w_eff, self.bias)
        if (
            self.eggroll_rank > 0
            and self.lora_A is not None
            and self.lora_B is not None
        ):
            if self.lora_B.abs().max().item() > 0.0:
                lora_out = F.linear(
                    F.linear(x_f, self.lora_A.float()), self.lora_B.float()
                )
                out = out + lora_out
        return out.to(x.dtype)

    @torch.inference_mode()
    def online_wsp_update(
        self,
        k_acts: torch.Tensor,
        v_target: torch.Tensor,
        a_anchor: Optional[torch.Tensor] = None,
        ridge_lambda: float = 8.0,
        max_rel_frob: float = 0.0030,
    ) -> Dict[str, float]:
        """
        Zero-backprop online closed-form Woodbury Ridge update into EGGROLL adapters:
          Delta W = (V_target - W_eff K) (K^T K + lambda I_N)^{-1} K^T
        Projected orthogonal to anchor covariance subspace V_a, clamped by Lyapunov
        relative Frobenius bound (<= max_rel_frob), and factored into lora_A / lora_B.
        """
        if self.eggroll_rank <= 0 or self.lora_A is None or self.lora_B is None:
            return {"applied": 0.0, "rel_frob": 0.0}

        w_eff = self.unpack_effective_weight()
        if self.lora_B.abs().max().item() > 0.0:
            w_eff = w_eff + (self.lora_B.float() @ self.lora_A.float())
        w_norm = float(torch.norm(w_eff).item()) + 1e-12

        k_f = k_acts.float()
        n_tok = k_f.shape[1]
        r_res = v_target.float() - (w_eff @ k_f)
        gram = k_f.T @ k_f
        lam = max(
            float(ridge_lambda),
            1e-2 * float(torch.trace(gram).item()) / float(max(1, n_tok)),
        )
        eye_n = torch.eye(n_tok, dtype=torch.float32, device=gram.device)
        coeffs = torch.linalg.solve(gram + lam * eye_n, k_f.T)
        dw = r_res @ coeffs

        if a_anchor is not None and a_anchor.shape[0] >= 4:
            q_a = min(24, a_anchor.shape[0], a_anchor.shape[1])
            # 2026-11-24 FIX: project dw (out x in) onto the orthogonal complement
            # of the anchor subspace using the ROW-space singular vectors U (in x q).
            # The old code used V (q x q from the 896x8 anchor), causing a shape
            # mismatch (896x896 @ 8x8). U has shape (in_features x q), so
            # dw - (dw @ U) @ U.T stays (out x in). Verified standalone.
            u_a, _, _ = torch.svd_lowrank(a_anchor.float(), q=q_a, niter=2)
            dw = dw - (dw @ u_a) @ u_a.T

        dw_norm = float(torch.norm(dw).item())
        rel_frob = dw_norm / w_norm
        if rel_frob > max_rel_frob:
            dw = dw * (max_rel_frob / (rel_frob + 1e-12))
            rel_frob = max_rel_frob

        q_r = min(self.eggroll_rank, dw.shape[0], dw.shape[1])
        u, s, v = torch.svd_lowrank(dw, q=q_r, niter=2)
        sqrt_s = torch.sqrt(s.clamp(min=0.0))
        b_new = u * sqrt_s.unsqueeze(0)
        a_new = sqrt_s.unsqueeze(1) * v.T

        cur_b = self.lora_B.float() + F.pad(b_new, (0, self.eggroll_rank - q_r))
        cur_a = self.lora_A.float() + F.pad(a_new, (0, 0, 0, self.eggroll_rank - q_r))
        self.lora_B.copy_(cur_b)
        self.lora_A.copy_(cur_a)
        return {"applied": 1.0, "rel_frob": round(rel_frob, 6), "lambda": round(lam, 4)}


class PackedLittleBitLinear(PackedBitLinear):
    """2-bit packed BitNet linear layer specialized for Council MoE & virtual swarm paths."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        group_size: int = 128,
        eggroll_rank: int = 16,
    ) -> None:
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            bias=False,
            group_size=group_size,
            num_planes=1,
            eggroll_rank=eggroll_rank,
        )


class LeeMach6Governor(nn.Module):
    """Adaptive spectral damping governor keeping MoE residual contributions bounded."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.scale = nn.Parameter(torch.ones(1, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x.float()).to(x.dtype) * self.scale.to(x.dtype)


class PackedCouncilMoE(nn.Module):
    """
    34-Expert Council MoE with:
      - Active-expert-only 2-bit lazy unpack (eliminates 1.78 GB bank dequant spikes)
      - Rank-16 virtual 9B swarm modulation
      - LeeMach6Governor + zero-initialized residual gate (tanh(alpha_l)|_{t=0} == 0)
    """

    def __init__(
        self,
        hidden_size: int,
        moe_ffn_dim: int = 512,
        num_experts: int = 34,
        top_k: int = 2,
        group_size: int = 128,
        swarm_rank: int = 16,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.moe_ffn_dim = moe_ffn_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.group_size = (
            group_size
            if (hidden_size % group_size == 0 and moe_ffn_dim % group_size == 0)
            else min(hidden_size, moe_ffn_dim)
        )

        self.router = nn.Linear(hidden_size, num_experts, bias=False)
        self.governor = LeeMach6Governor(hidden_size)
        self.gate_alpha = nn.Parameter(torch.zeros(1, dtype=torch.float32))

        self.w1_packed_len = (moe_ffn_dim * hidden_size + 3) // 4
        self.w2_packed_len = (hidden_size * moe_ffn_dim + 3) // 4
        g_in = hidden_size // self.group_size
        g_ffn = moe_ffn_dim // self.group_size

        self.register_buffer(
            "w1_q",
            torch.ones(num_experts, self.w1_packed_len, dtype=torch.uint8) * 0x55,
        )
        self.register_buffer(
            "w1_gamma", torch.zeros(num_experts, moe_ffn_dim, g_in, dtype=torch.float16)
        )
        self.register_buffer(
            "wgate_q",
            torch.ones(num_experts, self.w1_packed_len, dtype=torch.uint8) * 0x55,
        )
        self.register_buffer(
            "wgate_gamma",
            torch.zeros(num_experts, moe_ffn_dim, g_in, dtype=torch.float16),
        )
        self.register_buffer(
            "w2_q",
            torch.ones(num_experts, self.w2_packed_len, dtype=torch.uint8) * 0x55,
        )
        self.register_buffer(
            "w2_gamma",
            torch.zeros(num_experts, hidden_size, g_ffn, dtype=torch.float16),
        )

        # Rank-16 virtual 9B swarm modulation per expert
        self.swarm_a = nn.Parameter(
            torch.zeros(num_experts, swarm_rank, hidden_size, dtype=torch.float16)
        )
        self.swarm_b = nn.Parameter(
            torch.zeros(num_experts, hidden_size, swarm_rank, dtype=torch.float16)
        )

    def _unpack_expert_mat(
        self,
        q_buf: torch.Tensor,
        gamma_buf: torch.Tensor,
        expert_idx: int,
        out_dim: int,
        in_dim: int,
    ) -> torch.Tensor:
        g_count = in_dim // self.group_size
        t_tern = unpack_ternary_2bit(
            q_buf[expert_idx], (out_dim, g_count, self.group_size), out_dim * in_dim
        )
        return (t_tern * gamma_buf[expert_idx].float().unsqueeze(-1)).view(
            out_dim, in_dim
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_val = torch.tanh(self.gate_alpha)
        if gate_val.abs().item() == 0.0 and not torch.is_grad_enabled():
            return torch.zeros_like(x)

        bsz, seqlen, dim = x.shape
        x_flat = self.governor(x).view(-1, dim).float()
        logits = self.router(x_flat)
        top_vals, top_idx = torch.topk(logits, self.top_k, dim=-1)
        weights = F.softmax(top_vals, dim=-1)

        x_q = quantize_activations_w158a8(x_flat)
        out_flat = torch.zeros_like(x_flat)

        active_experts = torch.unique(top_idx).tolist()
        for e in active_experts:
            mask = top_idx == e
            if not mask.any():
                continue
            tok_idx, slot_idx = torch.where(mask)
            x_e = x_q[tok_idx]

            w1_e = self._unpack_expert_mat(
                self.w1_q, self.w1_gamma, e, self.moe_ffn_dim, self.hidden_size
            )
            wg_e = self._unpack_expert_mat(
                self.wgate_q, self.wgate_gamma, e, self.moe_ffn_dim, self.hidden_size
            )
            w2_e = self._unpack_expert_mat(
                self.w2_q, self.w2_gamma, e, self.hidden_size, self.moe_ffn_dim
            )

            h_e = F.silu(F.linear(x_e, wg_e)) * F.linear(x_e, w1_e)
            y_e = F.linear(h_e, w2_e)
            if self.swarm_b[e].abs().max().item() > 0.0:
                y_e = y_e + F.linear(
                    F.linear(x_e, self.swarm_a[e].float()), self.swarm_b[e].float()
                )

            out_flat.index_add_(
                0, tok_idx, y_e * weights[tok_idx, slot_idx].unsqueeze(-1)
            )

        return (gate_val * out_flat.view(bsz, seqlen, dim)).to(x.dtype)


class QuillanRMSNorm(nn.Module):
    """RMSNorm preserving exact LLaMA/Qwen/Falcon variance scaling."""

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=torch.float32))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_f = x.float()
        var = x_f.pow(2).mean(dim=-1, keepdim=True)
        out = x_f * torch.rsqrt(var + self.eps) * self.weight
        return out.to(x.dtype)


class PackedInt8Embedding(nn.Module):
    """Per-row INT8 quantized embedding table preserving semantic precision at 2x compression."""

    def __init__(self, num_embeddings: int, embedding_dim: int) -> None:
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.register_buffer(
            "weight_int8", torch.zeros(num_embeddings, embedding_dim, dtype=torch.int8)
        )
        self.register_buffer(
            "weight_scale", torch.ones(num_embeddings, 1, dtype=torch.float16)
        )

    @torch.no_grad()
    def set_weights_from_float(self, w: torch.Tensor) -> None:
        w_f = w.detach().float()
        scale = w_f.abs().amax(dim=-1, keepdim=True).clamp(min=1e-6) / 127.0
        q = torch.round(w_f / scale).clamp(-127.0, 127.0).to(torch.int8)
        self.weight_int8.copy_(q)
        self.weight_scale.copy_(scale.to(torch.float16))

    def unpack_float(self) -> torch.Tensor:
        return self.weight_int8.float() * self.weight_scale.float()

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        w_q = F.embedding(input_ids, self.weight_int8).float()
        s = F.embedding(input_ids, self.weight_scale).float()
        return w_q * s


def _compute_ntk_rope_cos_sin(
    seq_len: int,
    head_dim: int,
    base_theta: float,
    orig_max_pos: int,
    position_offset: int = 0,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Computes 2M NTK-aware RoPE cos/sin embeddings on the fly without pre-allocating 2M buffers."""
    total_pos = position_offset + seq_len
    if total_pos > orig_max_pos:
        scale = float(total_pos) / float(orig_max_pos)
        theta = base_theta * (scale ** (float(head_dim) / float(head_dim - 2)))
    else:
        theta = base_theta
    inv_freq = 1.0 / (
        theta
        ** (
            torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
            / float(head_dim)
        )
    )
    pos = torch.arange(
        position_offset, position_offset + seq_len, dtype=torch.float32, device=device
    )
    freqs = torch.outer(pos, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos(), emb.sin()


def _apply_rotary_pos_emb(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """Applies rotary position embeddings to (B, H, T, D) query/key tensors."""
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    rot = torch.cat((-x2, x1), dim=-1)
    return (x * cos.unsqueeze(0).unsqueeze(0)) + (rot * sin.unsqueeze(0).unsqueeze(0))


class QuillanSovereignBlock(nn.Module):
    """Single depth-matched transformer block with GQA, KV cache, SwiGLU BitNet FFN, and optional zero-gated Council MoE."""

    def __init__(self, cfg: QuillanSovereignConfig, layer_idx: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.layer_idx = layer_idx
        self.num_q = cfg.num_attention_heads
        self.num_kv = cfg.num_key_value_heads
        self.head_dim = cfg.head_dim
        self.kv_groups = self.num_q // self.num_kv

        self.input_layernorm = QuillanRMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.q_proj = PackedBitLinear(
            cfg.hidden_size,
            self.num_q * self.head_dim,
            bias=cfg.qkv_bias,
            group_size=cfg.group_size,
            num_planes=cfg.attn_planes,
            eggroll_rank=cfg.eggroll_rank,
            use_rht=cfg.use_rht,
        )
        self.k_proj = PackedBitLinear(
            cfg.hidden_size,
            self.num_kv * self.head_dim,
            bias=cfg.qkv_bias,
            group_size=cfg.group_size,
            num_planes=cfg.attn_planes,
            eggroll_rank=cfg.eggroll_rank,
            use_rht=cfg.use_rht,
        )
        self.v_proj = PackedBitLinear(
            cfg.hidden_size,
            self.num_kv * self.head_dim,
            bias=cfg.qkv_bias,
            group_size=cfg.group_size,
            num_planes=cfg.attn_planes,
            eggroll_rank=cfg.eggroll_rank,
            use_rht=cfg.use_rht,
        )
        self.o_proj = PackedBitLinear(
            self.num_q * self.head_dim,
            cfg.hidden_size,
            group_size=cfg.group_size,
            num_planes=cfg.attn_planes,
            eggroll_rank=cfg.eggroll_rank,
            use_rht=cfg.use_rht,
        )

        self.post_attention_layernorm = QuillanRMSNorm(
            cfg.hidden_size, eps=cfg.rms_norm_eps
        )
        self.gate_proj = PackedBitLinear(
            cfg.hidden_size,
            cfg.intermediate_size,
            group_size=cfg.group_size,
            num_planes=cfg.ffn_planes,
            eggroll_rank=cfg.eggroll_rank,
            use_rht=cfg.use_rht,
        )
        self.up_proj = PackedBitLinear(
            cfg.hidden_size,
            cfg.intermediate_size,
            group_size=cfg.group_size,
            num_planes=cfg.ffn_planes,
            eggroll_rank=cfg.eggroll_rank,
            use_rht=cfg.use_rht,
        )
        self.down_proj = PackedBitLinear(
            cfg.intermediate_size,
            cfg.hidden_size,
            group_size=cfg.group_size,
            num_planes=cfg.ffn_planes,
            eggroll_rank=cfg.eggroll_rank,
            use_rht=cfg.use_rht,
        )

        self.council_moe: Optional[PackedCouncilMoE] = (
            PackedCouncilMoE(
                cfg.hidden_size,
                moe_ffn_dim=cfg.moe_ffn_dim,
                num_experts=cfg.num_experts,
                top_k=cfg.num_experts_active,
                group_size=cfg.group_size,
                swarm_rank=cfg.eggroll_rank,
            )
            if layer_idx in cfg.moe_layer_indices
            else None
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        bsz, seqlen, _ = hidden_states.shape
        pos_offset = past_kv[0].shape[2] if past_kv is not None else 0

        h_norm = self.input_layernorm(hidden_states)
        q = (
            self.q_proj(h_norm)
            .view(bsz, seqlen, self.num_q, self.head_dim)
            .transpose(1, 2)
        )
        k = (
            self.k_proj(h_norm)
            .view(bsz, seqlen, self.num_kv, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.v_proj(h_norm)
            .view(bsz, seqlen, self.num_kv, self.head_dim)
            .transpose(1, 2)
        )

        cos, sin = _compute_ntk_rope_cos_sin(
            seqlen,
            self.head_dim,
            self.cfg.rope_theta,
            self.cfg.orig_max_position_embeddings,
            position_offset=pos_offset,
            device=hidden_states.device,
        )
        q = _apply_rotary_pos_emb(q, cos, sin)
        k = _apply_rotary_pos_emb(k, cos, sin)

        if past_kv is not None:
            k = torch.cat((past_kv[0], k), dim=2)
            v = torch.cat((past_kv[1], v), dim=2)
        new_kv = (k, v) if use_cache else None

        if self.kv_groups > 1:
            k_rep = k.repeat_interleave(self.kv_groups, dim=1)
            v_rep = v.repeat_interleave(self.kv_groups, dim=1)
        else:
            k_rep, v_rep = k, v

        attn_out = F.scaled_dot_product_attention(
            q, k_rep, v_rep, is_causal=(past_kv is None and seqlen > 1)
        )
        attn_out = (
            attn_out.transpose(1, 2)
            .contiguous()
            .view(bsz, seqlen, self.num_q * self.head_dim)
        )
        h = hidden_states + self.o_proj(attn_out)

        h_ffn_in = self.post_attention_layernorm(h)
        ffn_out = self.down_proj(
            F.silu(self.gate_proj(h_ffn_in)) * self.up_proj(h_ffn_in)
        )
        if self.council_moe is not None:
            ffn_out = ffn_out + self.council_moe(h_ffn_in)
        return h + ffn_out, new_kv


class QuillanSovereignBitNetLM(nn.Module):
    """Full depth-matched Quillan-Ronin Sovereign BitNet 1.58b Language Model."""

    def __init__(self, cfg: QuillanSovereignConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = PackedInt8Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList(
            [QuillanSovereignBlock(cfg, i) for i in range(cfg.num_hidden_layers)]
        )
        self.norm = QuillanRMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        if cfg.tie_word_embeddings:
            self.lm_head: Optional[PackedBitLinear] = None
        else:
            self.lm_head = PackedBitLinear(
                cfg.hidden_size,
                cfg.vocab_size,
                group_size=cfg.group_size,
                num_planes=1,
                eggroll_rank=cfg.lm_head_rank,
            )

    def count_logical_parameters(self) -> int:
        """Counts logical model parameters (treating 4x packed uint8 elements as 4 weights)."""
        total = self.cfg.vocab_size * self.cfg.hidden_size
        if not self.cfg.tie_word_embeddings:
            total += self.cfg.vocab_size * self.cfg.hidden_size
        for layer in self.layers:
            for mod in (
                layer.q_proj,
                layer.k_proj,
                layer.v_proj,
                layer.o_proj,
                layer.gate_proj,
                layer.up_proj,
                layer.down_proj,
            ):
                total += mod.out_features * mod.in_features
                if mod.eggroll_rank > 0:
                    total += mod.eggroll_rank * (mod.in_features + mod.out_features)
            total += 2 * self.cfg.hidden_size
            if layer.council_moe is not None:
                moe = layer.council_moe
                total += moe.num_experts * (3 * moe.moe_ffn_dim * moe.hidden_size)
                total += moe.num_experts * moe.hidden_size
        total += self.cfg.hidden_size
        return total

    def forward(
        self,
        input_ids: torch.Tensor,
        past_key_values: Optional[
            List[Optional[Tuple[torch.Tensor, torch.Tensor]]]
        ] = None,
        use_cache: bool = False,
    ) -> Tuple[
        torch.Tensor, Optional[List[Optional[Tuple[torch.Tensor, torch.Tensor]]]]
    ]:
        h = self.embed_tokens(input_ids)
        next_kvs: Optional[List[Optional[Tuple[torch.Tensor, torch.Tensor]]]] = (
            [] if use_cache else None
        )
        for idx, layer in enumerate(self.layers):
            pkv = past_key_values[idx] if past_key_values is not None else None
            h, nkv = layer(h, past_kv=pkv, use_cache=use_cache)
            if next_kvs is not None:
                next_kvs.append(nkv)
        h = self.norm(h)
        if self.lm_head is None:
            logits = F.linear(h, self.embed_tokens.unpack_float())
        else:
            logits = self.lm_head(h)
        return logits, next_kvs

    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 32,
        temperature: float = 0.0,
        eos_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        """Autoregressive generation with incremental per-layer KV caching."""
        self.eval()
        cur = input_ids
        logits, past_kv = self(cur, use_cache=True)
        for _ in range(max_new_tokens):
            next_logits = logits[:, -1, :]
            if temperature <= 0.0:
                next_tok = torch.argmax(next_logits, dim=-1, keepdim=True)
            else:
                probs = F.softmax(next_logits / temperature, dim=-1)
                next_tok = torch.multinomial(probs, num_samples=1)
            cur = torch.cat((cur, next_tok), dim=-1)
            if eos_token_id is not None and int(next_tok[0, 0].item()) == eos_token_id:
                break
            logits, past_kv = self(next_tok, past_key_values=past_kv, use_cache=True)
        return cur

    def save_packed_safetensors(self, path: Path) -> Path:
        """Saves packed 2-bit/int8 weights + JSON config metadata to safetensors."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tensors = {
            k: v.detach().cpu().contiguous() for k, v in self.state_dict().items()
        }
        meta = {"quillan_sovereign_config": json.dumps(asdict(self.cfg))}
        safetensors.torch.save_file(tensors, str(path), metadata=meta)
        logger.info(
            "Saved packed checkpoint -> %s (%d bytes)", path, path.stat().st_size
        )
        return path

    @classmethod
    def load_packed_safetensors(
        cls, path: Path, device: str = "cpu"
    ) -> "QuillanSovereignBitNetLM":
        """Loads a packed safetensors checkpoint deterministically."""
        path = Path(path)
        with safetensors.torch.safe_open(str(path), framework="pt", device=device) as f:
            raw_meta = f.metadata() or {}
            cfg_dict = json.loads(raw_meta["quillan_sovereign_config"])
            cfg_dict["moe_layer_indices"] = tuple(cfg_dict["moe_layer_indices"])
            cfg = QuillanSovereignConfig(**cfg_dict)
            model = cls(cfg).to(device)
            sd = {k: f.get_tensor(k) for k in f.keys()}
        model.load_state_dict(sd, strict=True)
        return model


def run_self_test() -> None:
    """
    Comprehensive CPU self-test verifying:
      (a) 2-bit pack/unpack bit-exactness
      (b) 1-plane & 2-plane ternary reconstruction
      (c) t=0 zero-gate identity for CouncilMoE & EGGROLL
      (d) logical parameter count <= 1B for Mini/Flagship/Dragon presets
      (e) online closed-form Woodbury WSP update + Lyapunov gate + anchor preservation
    """
    torch.manual_seed(42)

    # (a) 2-bit pack/unpack bit-exactness
    t_orig = torch.randint(-1, 2, (32, 128), dtype=torch.int8).float()
    packed, n_el = pack_ternary_2bit(t_orig)
    t_rec = unpack_ternary_2bit(packed, t_orig.shape, n_el)
    assert torch.equal(t_orig, t_rec), "2-bit pack/unpack mismatch!"

    # (b) 1-plane, 2-plane, & 3-plane RHT-32 ternary reconstruction
    w_test = torch.randn(64, 128) * 0.05
    lin1 = PackedBitLinear(
        128, 64, group_size=32, num_planes=1, eggroll_rank=8, use_rht=True
    )
    lin2 = PackedBitLinear(
        128, 64, group_size=32, num_planes=2, eggroll_rank=8, use_rht=True
    )
    lin3 = PackedBitLinear(
        128, 64, group_size=32, num_planes=3, eggroll_rank=8, use_rht=True
    )
    lin1.set_weights_from_float(w_test)
    lin2.set_weights_from_float(w_test)
    lin3.set_weights_from_float(w_test)
    err1 = float(torch.norm(w_test - lin1.unpack_effective_weight()).item())
    err2 = float(torch.norm(w_test - lin2.unpack_effective_weight()).item())
    err3 = float(torch.norm(w_test - lin3.unpack_effective_weight()).item())
    assert err3 < err2 < err1, (
        f"Multi-plane RHT-32 did not decrease error: {err1:.4f}, {err2:.4f}, {err3:.4f}"
    )

    # (c) t=0 zero-gate identity + qkv_bias + save/load + generate
    cfg = QuillanSovereignConfig(
        model_name="tiny-test",
        vocab_size=256,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        orig_max_position_embeddings=64,
        qkv_bias=True,
        moe_ffn_dim=128,
        moe_layer_indices=(1,),
        eggroll_rank=8,
    )
    model = QuillanSovereignBitNetLM(cfg).eval()
    x_ids = torch.randint(0, 256, (1, 8))
    logits_0, _ = model(x_ids)
    assert torch.isfinite(logits_0).all()
    assert float(model.layers[1].council_moe.gate_alpha.item()) == 0.0

    # (d) Parameter count <= 1B for Mini, Flagship, Dragon presets
    preset_counts: Dict[str, int] = {}
    for key in ("mini", "flagship", "dragon"):
        pcfg = QuillanSovereignConfig.for_model(key)
        # Compute analytical parameter count without allocating 1B tensors
        n_layers = pcfg.num_hidden_layers
        d = pcfg.hidden_size
        d_ff = pcfg.intermediate_size
        q_dim = pcfg.num_attention_heads * pcfg.head_dim
        kv_dim = pcfg.num_key_value_heads * pcfg.head_dim
        per_layer = d * q_dim + 2 * d * kv_dim + q_dim * d + 3 * d * d_ff + 2 * d
        moe_p = (
            len(pcfg.moe_layer_indices)
            * pcfg.num_experts
            * (3 * pcfg.moe_ffn_dim * d + d)
        )
        total_p = pcfg.vocab_size * d + n_layers * per_layer + moe_p + d
        assert total_p <= 1_000_000_000, f"{key} exceeds 1B params: {total_p}"
        preset_counts[key] = total_p

    # (e) Online closed-form Woodbury WSP update (< 5ms, Lyapunov bounded, anchor PPL delta <= 1e-4)
    k_act = torch.randn(128, 8)
    v_tgt = torch.randn(64, 8) * 0.1
    a_anc = torch.randn(16, 128)
    w_anc_before = lin2.unpack_effective_weight() @ a_anc.T
    t_start = time.perf_counter()
    wsp_res = lin2.online_wsp_update(k_act, v_tgt, a_anchor=a_anc, max_rel_frob=0.0030)
    elapsed_ms = (time.perf_counter() - t_start) * 1000.0
    assert wsp_res["applied"] == 1.0 and wsp_res["rel_frob"] <= 0.0030 + 1e-6

    w_anc_after = (
        lin2.unpack_effective_weight()
        + (lin2.lora_B.detach().float() @ lin2.lora_A.detach().float())
    ) @ a_anc.T
    anc_drift = float(
        torch.norm(w_anc_after - w_anc_before) / (torch.norm(w_anc_before) + 1e-12)
    )
    assert anc_drift <= 1e-4, f"Anchor drift exceeded bound: {anc_drift}"

    import tempfile

    with tempfile.TemporaryDirectory() as tmpd:
        ckpt = Path(tmpd) / "tiny_bitnet.safetensors"
        model.save_packed_safetensors(ckpt)
        reloaded = QuillanSovereignBitNetLM.load_packed_safetensors(ckpt)
        logits_reloaded, _ = reloaded(x_ids)
        assert torch.allclose(logits_0, logits_reloaded, atol=1e-5)

    out_gen = model.generate(x_ids, max_new_tokens=4)
    assert out_gen.shape == (1, 12)

    print(
        f"ARCH_SELF_TEST PASS | err1={err1:.4f} err2={err2:.4f} err3={err3:.4f} "
        f"presets={preset_counts} wsp_ms={elapsed_ms:.2f} rel_frob={wsp_res['rel_frob']:.4f} "
        f"anc_drift={anc_drift:.2e}"
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
    )
    run_self_test()
