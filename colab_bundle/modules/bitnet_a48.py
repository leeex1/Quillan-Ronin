#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BITNET a4.8: 4-BIT ACTIVATIONS FOR 1-BIT LLMS
=============================================
Reference:
  Hongyu Wang, Shuming Ma, Furu Wei (Microsoft Research & UCAS, 2024)
  "BitNet a4.8: 4-bit Activations for 1-bit LLMs"
  arXiv:2411.04965v1 [cs.CL] (10 pp)

Core Architectural Invariants:
  1. 1.58-bit Weights:
     - Ternary quantization Q_w(W) = alpha * RoundClip(W / (alpha + eps), -1, 1), alpha = mean(|W|).
  2. Hybrid Quantization & Sparsification Strategy:
     - Inputs to Attention (Q, K, V) and FFN (Up, Gate):
       Gaussian-like distributions with minimal outliers -> 4-bit absmean quantization:
       Q_INT4(X) = (beta / sqrt(7)) * RoundClip(sqrt(7) * X / (beta + eps), -8, 7), beta = mean(|X|).
     - Attention Output Projection (W_out):
       Sharp zero-peaked distribution with outliers -> Top-K (50%) sparsification + INT8:
       Y = (Q_INT8(X) ⊙ M) * Q_w(W)^T, M = Topk_50%( |X| ).
     - Feed-Forward Down Projection (W_down):
       Squared ReLU GLU: ReLU2GLU(X) = (X W_up^T) ⊙ (ReLU(X W_gate^T))^2.
       Achieves >80% activation sparsity, keeping non-zero intermediate states at INT8.
  3. FP4 MinMax Quantizer (E2M1 Format):
     - Alternative floating-point mode providing wider dynamic range for long-tailed distributions.
  4. Efficiency Profile:
     - Activates ~55% of parameters per forward pass while enabling 4-bit tensor-core GEMM.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


# =====================================================================
# 1. Quantization & Sparsification Operators
# =====================================================================

def round_clip(x: torch.Tensor, a: float, b: float) -> torch.Tensor:
    """RoundClip(X, a, b) = min(max(round(X), a), b) with STE."""
    rounded = torch.round(x)
    clipped = torch.clamp(rounded, a, b)
    # STE: straight-through estimator for backward pass
    return x + (clipped - x).detach()


class BitNetWeightQuantizer:
    """Ternary weight quantizer Q_w(W) per Eq. (2)."""

    @staticmethod
    def quantize(w: torch.Tensor, eps: float = 1e-5) -> Tuple[torch.Tensor, torch.Tensor]:
        alpha = torch.mean(torch.abs(w))
        scale = alpha + eps
        w_norm = w / scale
        w_q = round_clip(w_norm, -1.0, 1.0)
        return alpha * w_q, alpha


class ActivationINT4Quantizer:
    """4-bit activation quantizer Q_INT4(X) per Eq. (7)."""

    @staticmethod
    def quantize(x: torch.Tensor, eps: float = 1e-5) -> Tuple[torch.Tensor, torch.Tensor]:
        beta = torch.mean(torch.abs(x), dim=-1, keepdim=True)
        scale = beta + eps
        c = math.sqrt(7.0)
        x_scaled = (c / scale) * x
        x_q = round_clip(x_scaled, -8.0, 7.0)
        x_quant = (beta / c) * x_q
        return x_quant, beta


class ActivationINT8Quantizer:
    """8-bit activation quantizer Q_INT8(X) per Eq. (3)."""

    @staticmethod
    def quantize(x: torch.Tensor, eps: float = 1e-5) -> Tuple[torch.Tensor, torch.Tensor]:
        gamma, _ = torch.max(torch.abs(x), dim=-1, keepdim=True)
        scale = gamma + eps
        x_scaled = (127.0 / scale) * x
        x_q = round_clip(x_scaled, -128.0, 127.0)
        x_quant = (gamma / 127.0) * x_q
        return x_quant, gamma


class ActivationFP4Quantizer:
    """
    MinMax FP4 (E2M1) Quantizer per Eq. (8) & (9).
    E=2, M=1 -> 4-bit floating point with 1 sign bit, 2 exponent bits, 1 mantissa bit.
    """

    @staticmethod
    def quantize(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
        m_bits = 1
        e_bits = 2
        x_abs = torch.abs(x)
        x_max, _ = torch.max(x_abs, dim=-1, keepdim=True)
        x_max = torch.clamp(x_max, min=eps)

        # b = log2(2 - 2^-M / |X|max) + 2^E - 1
        b_bias = torch.log2(torch.tensor(2.0 - 2.0 ** (-m_bits), device=x.device) / x_max) + (2.0 ** e_bits - 1.0)
        # gamma = 2^max(floor(log2|X| + b), 1)
        # To avoid log2(0), clamp x_abs
        log_x = torch.log2(torch.clamp(x_abs, min=eps))
        gamma = torch.pow(2.0, torch.clamp(torch.floor(log_x + b_bias), min=1.0))

        scale = 2.0 ** (m_bits + b_bias)
        quant_scale = scale / (gamma + eps)
        x_scaled = quant_scale * x
        x_q = torch.round(x_scaled)
        # FP4 range: [-8, 7]
        x_q = torch.clamp(x_q, -8.0, 7.0)
        x_quant = (gamma / (scale + eps)) * x_q
        return x + (x_quant - x).detach()


class TopKSparsifier:
    """Top-K sparsification M = Topk(|X|) per Eq. (1)."""

    @staticmethod
    def apply(x: torch.Tensor, k_ratio: float = 0.5) -> Tuple[torch.Tensor, torch.Tensor]:
        d = x.shape[-1]
        k = max(1, int(round(d * k_ratio)))
        x_abs = torch.abs(x)
        topk_vals, _ = torch.topk(x_abs, k, dim=-1, largest=True)
        kth_val = topk_vals[..., -1:].detach()
        mask = (x_abs >= kth_val).float()
        # STE for mask
        return x * mask, mask


# =====================================================================
# 2. BitNet a4.8 Linear Projections
# =====================================================================

class BitLinearA48(nn.Module):
    """
    BitNet a4.8 Linear layer with ternary weights and configurable activation path:
      - 'int4': 4-bit absmean activation quantization (used for Q, K, V, Up, Gate).
      - 'sparse_int8': Top-K (50%) sparsification followed by 8-bit quantization (Attention Out).
      - 'relu2_int8': 8-bit quantization over squared-ReLU activations (FFN Down).
      - 'fp4': E2M1 FP4 quantization.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        activation_mode: str = "int4",
        topk_ratio: float = 0.5,
        bias: bool = False
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.activation_mode = activation_mode
        self.topk_ratio = topk_ratio
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * (1.0 / math.sqrt(in_features)))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        # 1. Activation quantization / sparsification
        sparsity = 0.0
        if self.activation_mode == "int4":
            x_act, _ = ActivationINT4Quantizer.quantize(x)
        elif self.activation_mode == "sparse_int8":
            x_sparse, mask = TopKSparsifier.apply(x, k_ratio=self.topk_ratio)
            x_act, _ = ActivationINT8Quantizer.quantize(x_sparse)
            sparsity = 1.0 - (mask.sum() / mask.numel()).item()
        elif self.activation_mode == "relu2_int8":
            x_act, _ = ActivationINT8Quantizer.quantize(x)
            sparsity = (x == 0.0).float().mean().item()
        elif self.activation_mode == "fp4":
            x_act = ActivationFP4Quantizer.quantize(x)
        else:
            x_act = x

        # 2. Ternary weight quantization
        w_quant, alpha = BitNetWeightQuantizer.quantize(self.weight)

        # 3. GEMM
        out = F.linear(x_act, w_quant, self.bias)
        meta = {"sparsity": sparsity, "alpha": alpha.item()}
        return out, meta


# =====================================================================
# 3. BitNet a4.8 Attention & Feed-Forward Network Blocks
# =====================================================================

class BitNetA48MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention using BitNet a4.8 hybrid quantization:
      - Q, K, V projections use INT4 activations and ternary weights.
      - Attention Out projection uses Top-K 50% sparsification + INT8 activations and ternary weights.
    """

    def __init__(self, d_model: int = 64, num_heads: int = 4, topk_ratio: float = 0.5):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.q_proj = BitLinearA48(d_model, d_model, activation_mode="int4")
        self.k_proj = BitLinearA48(d_model, d_model, activation_mode="int4")
        self.v_proj = BitLinearA48(d_model, d_model, activation_mode="int4")
        self.out_proj = BitLinearA48(d_model, d_model, activation_mode="sparse_int8", topk_ratio=topk_ratio)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        b, s, d = x.shape
        q, _ = self.q_proj(x)
        k, _ = self.k_proj(x)
        v, _ = self.v_proj(x)

        q = q.view(b, s, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, s, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, s, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_out = torch.matmul(attn_probs, v)

        attn_out = attn_out.transpose(1, 2).contiguous().view(b, s, d)
        out, meta = self.out_proj(attn_out)
        return out, meta


class BitNetA48FFN(nn.Module):
    """
    Feed-Forward Network with ReLU^2 GLU and hybrid quantization:
      - Up & Gate projections use INT4 activations and ternary weights.
      - ReLU2GLU: (x W_up^T) ⊙ (ReLU(x W_gate^T))^2.
      - Down projection uses 8-bit activations on the naturally sparse (>80%) intermediate state.
    """

    def __init__(self, d_model: int = 64, d_ff: int = 128):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff

        self.up_proj = BitLinearA48(d_model, d_ff, activation_mode="int4")
        self.gate_proj = BitLinearA48(d_model, d_ff, activation_mode="int4")
        self.down_proj = BitLinearA48(d_ff, d_model, activation_mode="relu2_int8")

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        up, _ = self.up_proj(x)
        gate, _ = self.gate_proj(x)

        # ReLU^2 GLU: Eq. (5)
        # ReLU2(z) = (max(0, z))^2
        relu2_gate = torch.square(F.relu(gate))
        intermediate = up * relu2_gate

        # Measure sparsity of intermediate activation
        sparsity = (intermediate == 0.0).float().mean().item()

        out, meta = self.down_proj(intermediate)
        meta["ffn_intermediate_sparsity"] = sparsity
        return out, meta


class BitNetA48TransformerBlock(nn.Module):
    """
    Full BitNet a4.8 Transformer Layer combining Attention and FFN with SubLN.
    """

    def __init__(self, d_model: int = 64, num_heads: int = 4, d_ff: int = 128, topk_ratio: float = 0.5):
        super().__init__()
        self.ln1 = nn.RMSNorm(d_model)
        self.attn = BitNetA48MultiHeadAttention(d_model=d_model, num_heads=num_heads, topk_ratio=topk_ratio)
        self.ln2 = nn.RMSNorm(d_model)
        self.ffn = BitNetA48FFN(d_model=d_model, d_ff=d_ff)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        norm_x1 = self.ln1(x)
        attn_out, attn_meta = self.attn(norm_x1)
        x = x + attn_out

        norm_x2 = self.ln2(x)
        ffn_out, ffn_meta = self.ffn(norm_x2)
        x = x + ffn_out

        combined_meta = {**attn_meta, **ffn_meta}
        return x, combined_meta
