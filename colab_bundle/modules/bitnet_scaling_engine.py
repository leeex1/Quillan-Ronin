#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BITNET: SCALING 1-BIT TRANSFORMERS FOR LARGE LANGUAGE MODELS
=============================================================
Reference:
  Hongyu Wang, Shuming Ma, Li Dong, Shaohan Huang, Huaijie Wang,
  Lingxiao Ma, Fan Yang, Ruiping Wang, Yi Wu, Furu Wei
  (Microsoft Research, UCAS, Tsinghua Univ., Oct 2023)
  "BitNet: Scaling 1-bit Transformers for Large Language Models"
  arXiv:2310.11453v1 [cs.CL] (14 pp)

Core Invariants & Foundations:
  1. Strict 1-Bit Binary Weights in {-1, +1}:
     - Weight centralization: alpha = (1 / nm) * sum_{ij} W_{ij}
     - Binarization: W_tilde = Sign(W - alpha) in {-1, +1}
     - Scale factor: beta = (1 / nm) * ||W||_1
     - Straight-Through Estimator (STE) for backpropagation.
  2. Activation Quantization to 8-Bit (b=8, Q_b = 128):
     - Symmetric absmax: x_tilde = Clip(x * (Q_b / gamma), -Q_b + eps, Q_b - eps), gamma = ||x||_inf
     - Asymmetric non-negative (pre-ReLU): x_tilde = Clip((x - eta) * (Q_b / gamma), eps, Q_b - eps)
  3. Variance-Preserving SubLN Normalization:
     - LayerNorm before activation quantization ensures Var(y) approx 1, matching FP16 dynamics.
     - BitLinear equation: y = W_tilde * Quant(LN(x)) * (beta * gamma / Q_b)
  4. Group Quantization & Normalization for Model Parallelism:
     - Tensor partition into G groups: alpha_g, beta_g, gamma_g computed locally.
     - Eliminates all-reduce communication bottlenecks across multi-device tensor parallel ranks.
  5. Empirical Scaling Law Invariant:
     - Follows power-law scaling with competitive perplexity and delta loss <= 0.09 vs FP16.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


# =====================================================================
# 1. Binary Weight & Activation Quantizers
# =====================================================================

def sign_ste(x: torch.Tensor) -> torch.Tensor:
    """Sign function with Straight-Through Estimator (STE): maps to {-1, +1}."""
    sign = torch.where(x > 0, torch.ones_like(x), -torch.ones_like(x))
    return x + (sign - x).detach()


class BinaryWeightQuantizer:
    """
    Eq. (1)-(3): Zero-mean centralization and binary sign quantization:
      alpha = mean(W)
      W_tilde = Sign(W - alpha) in {-1, +1}
      beta = mean(|W|)
    """

    @staticmethod
    def quantize(w: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        alpha = torch.mean(w)
        w_centered = w - alpha
        w_bin = sign_ste(w_centered)
        beta = torch.mean(torch.abs(w))
        return w_bin, alpha, beta


class AbsmaxActivationQuantizerBitNet:
    """
    Eq. (4)-(6): Absmax 8-bit activation quantization (Q_b = 2^{b-1} = 128):
      - Symmetric signed: [-Q_b + eps, Q_b - eps]
      - Asymmetric non-negative: [eps, Q_b - eps] with eta = min(x)
    """

    @staticmethod
    def quantize_symmetric(
        x: torch.Tensor,
        q_b: float = 128.0,
        eps: float = 1e-4
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        gamma = torch.norm(x, p=float("inf"), dim=-1, keepdim=True) + eps
        scale = q_b / gamma
        x_scaled = x * scale
        # Round-clip with STE
        clipped = torch.clamp(torch.round(x_scaled), -q_b + eps, q_b - eps)
        x_q = x_scaled + (clipped - x_scaled).detach()
        return x_q, gamma

    @staticmethod
    def quantize_asymmetric(
        x: torch.Tensor,
        q_b: float = 128.0,
        eps: float = 1e-4
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        eta = torch.min(x, dim=-1, keepdim=True)[0]
        x_shifted = x - eta
        gamma = torch.norm(x_shifted, p=float("inf"), dim=-1, keepdim=True) + eps
        scale = q_b / gamma
        x_scaled = x_shifted * scale
        clipped = torch.clamp(torch.round(x_scaled), eps, q_b - eps)
        x_q = x_scaled + (clipped - x_scaled).detach()
        return x_q, eta, gamma


# =====================================================================
# 2. Original BitLinear 1-Bit Layer (Eq. 11)
# =====================================================================

class BitLinear1Bit(nn.Module):
    """
    Original 1-bit BitLinear Layer:
      y = W_tilde * Quant(LN(x)) * (beta * gamma / Q_b)
      - W_tilde in {-1, +1}
      - Strict zero bias
      - Variance preservation via SubLN
    """

    def __init__(self, in_features: int, out_features: int, q_b: float = 128.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.q_b = q_b

        # SubLN: LayerNorm without learnable bias
        self.subln = nn.LayerNorm(in_features, elementwise_affine=True)
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * (1.0 / math.sqrt(in_features)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Normalize activation to preserve variance
        x_norm = self.subln(x)

        # 2. Quantize activation
        x_quant, gamma = AbsmaxActivationQuantizerBitNet.quantize_symmetric(x_norm, q_b=self.q_b)

        # 3. Quantize weights to {-1, +1}
        w_bin, _, beta = BinaryWeightQuantizer.quantize(self.weight)

        # 4. Binary matrix multiplication
        y_int = F.linear(x_quant, w_bin)

        # 5. Rescale / dequantize: (beta * gamma / Q_b)
        scale = (beta * gamma) / self.q_b
        return y_int * scale


# =====================================================================
# 3. Group Quantization & Normalization for Model Parallelism (Eq. 13)
# =====================================================================

class GroupQuantizedBitLinear(nn.Module):
    """
    Group Quantization BitLinear (Section 2.2, Eq. 13):
      Divides weight W into G groups along output/partition dimension.
      alpha_g, beta_g, and gamma_g are estimated independently per group,
      allowing communication-free local computation on model-parallel shards.
    """

    def __init__(self, in_features: int, out_features: int, num_groups: int = 2, q_b: float = 128.0):
        super().__init__()
        assert out_features % num_groups == 0, "out_features must be divisible by num_groups"
        self.in_features = in_features
        self.out_features = out_features
        self.num_groups = num_groups
        self.group_size = out_features // num_groups
        self.q_b = q_b

        self.subln = nn.LayerNorm(in_features, elementwise_affine=True)
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * (1.0 / math.sqrt(in_features)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = self.subln(x)
        x_quant, gamma = AbsmaxActivationQuantizerBitNet.quantize_symmetric(x_norm, q_b=self.q_b)

        # Split weight into G groups and compute local alpha_g, beta_g
        w_groups = self.weight.view(self.num_groups, self.group_size, self.in_features)
        y_group_list = []

        for g in range(self.num_groups):
            w_g = w_groups[g]
            w_bin_g, _, beta_g = BinaryWeightQuantizer.quantize(w_g)
            y_int_g = F.linear(x_quant, w_bin_g)
            scale_g = (beta_g * gamma) / self.q_b
            y_group_list.append(y_int_g * scale_g)

        return torch.cat(y_group_list, dim=-1)


# =====================================================================
# 4. Multi-Head Attention & Transformer Block
# =====================================================================

class BitNet1BitAttention(nn.Module):
    """Multi-Head Self-Attention using BitLinear1Bit projections."""

    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.q_proj = BitLinear1Bit(d_model, d_model)
        self.k_proj = BitLinear1Bit(d_model, d_model)
        self.v_proj = BitLinear1Bit(d_model, d_model)
        self.out_proj = BitLinear1Bit(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, s, d = x.shape
        q = self.q_proj(x).view(b, s, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, s, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, s, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        probs = F.softmax(scores, dim=-1)
        attn_out = torch.matmul(probs, v).transpose(1, 2).contiguous().view(b, s, d)
        return self.out_proj(attn_out)


class BitNet1BitFFN(nn.Module):
    """Feed-Forward Network using BitLinear1Bit projections and ReLU activation."""

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.up_proj = BitLinear1Bit(d_model, d_ff)
        self.down_proj = BitLinear1Bit(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.relu(self.up_proj(x)))


class BitNetScalingBlock(nn.Module):
    """
    Standard BitNet Transformer Block:
      Residual -> BitLinear Self-Attention -> Residual -> BitLinear FFN.
    """

    def __init__(self, d_model: int = 64, num_heads: int = 4, d_ff: int = 128):
        super().__init__()
        self.attn = BitNet1BitAttention(d_model, num_heads)
        self.ffn = BitNet1BitFFN(d_model, d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(x)
        x = x + self.ffn(x)
        return x


# =====================================================================
# 5. BitNet Scaling Law & Energy Efficiency Model
# =====================================================================

class BitNetScalingLawAnalyzer:
    """
    Energy and Scaling law analysis from Figure 1 & Section 4:
      - Model sizes: 100M, 1B, 10B, 30B, 100B.
      - Energy cost reduction ratios vs FP16:
        * 100M: 8.8x
        * 1B:   11.0x
        * 10B:  18.6x
        * 30B:  29.1x
        * 100B: 38.8x
      - Scaling loss gap delta L <= 0.09 nats across all scales.
    """

    ENERGY_REDUCTION_RATIOS = {
        "100M": 8.8,
        "1B": 11.0,
        "10B": 18.6,
        "30B": 29.1,
        "100B": 38.8
    }

    @classmethod
    def get_energy_reduction(cls, model_size: str) -> float:
        return cls.ENERGY_REDUCTION_RATIOS.get(model_size, 10.0)

    @classmethod
    def max_scaling_loss_gap(cls) -> float:
        """Paper Figure 1 confirms delta L <= 0.09 between BitNet and FP16 baseline."""
        return 0.09
