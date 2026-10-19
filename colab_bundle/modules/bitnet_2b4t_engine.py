#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BITNET b1.58 2B4T: NATIVE 1-BIT LLM AT SCALE (TECHNICAL REPORT)
================================================================
Reference:
  Shuming Ma, Hongyu Wang, Shaohan Huang, Xingxing Zhang, Ying Hu,
  Ting Song, Yan Xia, Furu Wei (Microsoft Research, UCAS, Tsinghua, 2025)
  "BitNet b1.58 2B4T Technical Report"
  arXiv:2504.12285v2 [cs.CL] (14 pp)

Core Architectural & Training Innovations:
  1. Native 1-bit Transformer Architecture at 2B Parameters:
     - BitLinear layers with absmean weight quantization to {-1, 0, 1} with STE.
     - Per-token absmax 8-bit integer activation quantization.
     - SubLN (Sub-LayerNorm) normalization for quantized training stability.
     - Squared ReLU (ReLU^2) FFN replacing SwiGLU to induce activation sparsity.
     - Rotary Position Embeddings (RoPE).
     - Strict zero-bias invariant across all linear and normalization layers.
  2. Two-Stage Pre-training Training Dynamics:
     - Learning Rate: Stage 1 high peak LR with cosine decay; Stage 2 cooldown
       with abrupt drop to lower peak cosine schedule on curated high-quality data.
     - Weight Decay: Stage 1 cosine schedule up to 0.1; Stage 2 disabled (0.0)
       to let parameters settle into fine-grained 1-bit optima.
  3. SFT & DPO Alignment:
     - SFT Loss: Token-sum reduction instead of token-mean reduction.
     - DPO (Direct Preference Optimization): beta = 0.1, lr = 2e-7, 2 epochs.
  4. 7nm Energy Model Invariant:
     - FP16 ADD: 0.16 pJ, FP16 MUL: 0.34 pJ (Total MAC = 0.50 pJ).
     - INT8 ADD: 0.007 pJ (Arithmetic operations energy ratio = 0.50 / 0.007 = 71.43x).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


# =====================================================================
# 1. Quantization Primitives & SubLN
# =====================================================================

def round_clip(x: torch.Tensor, a: float, b: float) -> torch.Tensor:
    """Straight-Through Estimator (STE) round-clip operator."""
    rounded = torch.round(x)
    clipped = torch.clamp(rounded, a, b)
    return x + (clipped - x).detach()


class SubLN(nn.Module):
    """
    Sub-LayerNorm (Wang et al., 2022) with zero bias invariant:
    Applies LayerNorm before projection with learned scale gamma (no bias beta).
    """

    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        return x_norm * self.weight


class AbsmeanWeightQuantizer2B4T:
    """
    Absmean weight quantization to {-1, 0, 1}:
      gamma = mean(|W|)
      W_tilde = RoundClip(W / (gamma + eps), -1, 1)
    """

    @staticmethod
    def quantize(w: torch.Tensor, eps: float = 1e-5) -> Tuple[torch.Tensor, torch.Tensor]:
        gamma = torch.mean(torch.abs(w))
        scale = gamma + eps
        w_norm = w / scale
        w_ternary = round_clip(w_norm, -1.0, 1.0)
        return gamma * w_ternary, gamma


class AbsmaxActivationQuantizer2B4T:
    """
    Per-token absmax 8-bit activation quantization to [-128, 127]:
      scale = 127.0 / (max_j |X_j| + eps)
      X_tilde = RoundClip(X * scale, -128, 127)
    """

    @staticmethod
    def quantize(x: torch.Tensor, q_b: float = 127.0, eps: float = 1e-5) -> Tuple[torch.Tensor, torch.Tensor]:
        max_val, _ = torch.max(torch.abs(x), dim=-1, keepdim=True)
        scale = q_b / (max_val + eps)
        x_scaled = x * scale
        x_q = round_clip(x_scaled, -128.0, 127.0)
        x_dequant = x_q / scale
        return x_dequant, scale


# =====================================================================
# 2. BitLinear 2B4T Layer
# =====================================================================

class BitLinear2B4T(nn.Module):
    """
    Canonical BitLinear layer for BitNet b1.58 2B4T:
      - 1.58-bit ternary weights {-1, 0, 1}.
      - 8-bit per-token absmax activation quantization.
      - Integrated SubLN normalization.
      - Strict zero-bias invariant.
    """

    def __init__(self, in_features: int, out_features: int, use_subln: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_subln = use_subln

        if use_subln:
            self.subln = SubLN(in_features)
        else:
            self.subln = None

        self.weight = nn.Parameter(torch.randn(out_features, in_features) * (1.0 / math.sqrt(in_features)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.subln is not None:
            x = self.subln(x)
        x_quant, _ = AbsmaxActivationQuantizer2B4T.quantize(x)
        w_quant, _ = AbsmeanWeightQuantizer2B4T.quantize(self.weight)
        return F.linear(x_quant, w_quant)


# =====================================================================
# 3. Squared ReLU (ReLU^2) Feed-Forward Network
# =====================================================================

class ReLU2FFN(nn.Module):
    """
    Feed-Forward Network with Squared ReLU activation (ReLU^2):
      FFN(x) = Down_proj((ReLU(Up_proj(x)))^2)
    Induces higher activation sparsity and improved computational efficiency in 1-bit LLMs.
    """

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.up_proj = BitLinear2B4T(d_model, d_ff, use_subln=True)
        self.down_proj = BitLinear2B4T(d_ff, d_model, use_subln=True)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, float]:
        hidden = self.up_proj(x)
        # Squared ReLU: ReLU(h)^2
        relu_out = F.relu(hidden)
        relu2_out = torch.square(relu_out)

        # Measure sparsity
        sparsity = (relu2_out == 0.0).float().mean().item()

        out = self.down_proj(relu2_out)
        return out, sparsity


# =====================================================================
# 4. Multi-Head Attention & Transformer Block
# =====================================================================

class BitNet2B4TAttention(nn.Module):
    """
    Multi-Head Attention with BitLinear2B4T projections and SubLN.
    """

    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.q_proj = BitLinear2B4T(d_model, d_model, use_subln=True)
        self.k_proj = BitLinear2B4T(d_model, d_model, use_subln=True)
        self.v_proj = BitLinear2B4T(d_model, d_model, use_subln=True)
        self.out_proj = BitLinear2B4T(d_model, d_model, use_subln=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, s, d = x.shape
        q = self.q_proj(x).view(b, s, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, s, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, s, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        probs = F.softmax(scores, dim=-1)
        attn_out = torch.matmul(probs, v).transpose(1, 2).contiguous().view(b, s, d)
        return self.out_proj(attn_out)


class BitNet2B4TTransformerBlock(nn.Module):
    """
    Complete BitNet b1.58 2B4T Transformer Block:
      Residual -> SubLN Attention -> Residual -> SubLN ReLU^2 FFN.
    """

    def __init__(self, d_model: int = 64, num_heads: int = 4, d_ff: int = 128):
        super().__init__()
        self.attn = BitNet2B4TAttention(d_model, num_heads)
        self.ffn = ReLU2FFN(d_model, d_ff)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, float]:
        x = x + self.attn(x)
        ffn_out, sparsity = self.ffn(x)
        x = x + ffn_out
        return x, sparsity


# =====================================================================
# 5. Two-Stage Pre-training Schedulers
# =====================================================================

class TwoStageLRScheduler:
    """
    Two-stage learning rate schedule from Section 3.1.1:
      - Stage 1 (0 to T_split): High peak LR with cosine decay.
      - Stage 2 (T_split to T_total): Cooldown phase with abrupt step to lower peak LR
        and continued cosine decay.
    """

    def __init__(
        self,
        peak_lr_stage1: float = 1.5e-3,
        min_lr_stage1: float = 1.5e-4,
        peak_lr_stage2: float = 3.0e-4,
        min_lr_stage2: float = 3.0e-5,
        total_steps: int = 10000,
        split_ratio: float = 0.5
    ):
        self.peak_lr_stage1 = peak_lr_stage1
        self.min_lr_stage1 = min_lr_stage1
        self.peak_lr_stage2 = peak_lr_stage2
        self.min_lr_stage2 = min_lr_stage2
        self.total_steps = total_steps
        self.split_step = int(total_steps * split_ratio)

    def get_lr(self, step: int) -> float:
        step = max(0, min(step, self.total_steps))
        if step < self.split_step:
            # Stage 1 cosine decay
            progress = step / max(1, self.split_step)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return self.min_lr_stage1 + (self.peak_lr_stage1 - self.min_lr_stage1) * cosine
        else:
            # Stage 2 cooldown cosine decay
            progress = (step - self.split_step) / max(1, (self.total_steps - self.split_step))
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return self.min_lr_stage2 + (self.peak_lr_stage2 - self.min_lr_stage2) * cosine


class TwoStageWeightDecayScheduler:
    """
    Two-stage weight decay schedule from Section 3.1.2:
      - Stage 1: Cosine schedule reaching peak of 0.1.
      - Stage 2: Weight decay set strictly to 0.0 to allow parameters to settle.
    """

    def __init__(self, peak_wd: float = 0.1, total_steps: int = 10000, split_ratio: float = 0.5):
        self.peak_wd = peak_wd
        self.total_steps = total_steps
        self.split_step = int(total_steps * split_ratio)

    def get_wd(self, step: int) -> float:
        step = max(0, min(step, self.total_steps))
        if step < self.split_step:
            progress = step / max(1, self.split_step)
            # Ramp/cosine up to peak_wd
            return self.peak_wd * 0.5 * (1.0 - math.cos(math.pi * progress))
        else:
            # Strict 0.0 in Stage 2 cooldown
            return 0.0


# =====================================================================
# 6. SFT Loss Aggregation & DPO Optimization
# =====================================================================

class SFTLossAggregator:
    """
    Section 3.2.3: Token-sum loss aggregation instead of token-mean reduction.
    Empirically improves convergence and stability for native 1-bit models.
    """

    @staticmethod
    def compute_loss(logits: torch.Tensor, targets: torch.Tensor, ignore_index: int = -100) -> torch.Tensor:
        """
        logits: [batch_size, seq_len, vocab_size]
        targets: [batch_size, seq_len]
        """
        vocab_size = logits.size(-1)
        loss = F.cross_entropy(
            logits.view(-1, vocab_size),
            targets.view(-1),
            ignore_index=ignore_index,
            reduction="sum"  # Strict summation
        )
        return loss


class DPOTrainer2B4T:
    """
    Section 3.3: Direct Preference Optimization (DPO) objective:
      L_DPO = -E[log sigma(beta * (log(pi_theta(y_w|x)/pi_ref(y_w|x)) - log(pi_theta(y_l|x)/pi_ref(y_l|x))))]
    Default beta = 0.1, lr = 2e-7.
    """

    def __init__(self, beta: float = 0.1):
        self.beta = beta

    def loss(
        self,
        policy_chosen_logps: torch.Tensor,
        policy_rejected_logps: torch.Tensor,
        ref_chosen_logps: torch.Tensor,
        ref_rejected_logps: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Computes DPO loss and implicit reward margins."""
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        ref_logratios = ref_chosen_logps - ref_rejected_logps

        logits = self.beta * (pi_logratios - ref_logratios)
        losses = -F.logsigmoid(logits)

        chosen_rewards = self.beta * (policy_chosen_logps - ref_chosen_logps).detach()
        rejected_rewards = self.beta * (policy_rejected_logps - ref_rejected_logps).detach()

        return losses.mean(), chosen_rewards, rejected_rewards


# =====================================================================
# 7. Hardware Energy Metrics Model (Table 4 at 7nm)
# =====================================================================

class BitNet2B4TEnergyMetrics:
    """
    Table 4 (Horowitz 2014, Zhang et al. 2022) at 7nm Process Node:
      - FP16 ADD: 0.16 pJ
      - FP16 MUL: 0.34 pJ
      - Total FP16 MAC: 0.50 pJ
      - INT8 ADD: 0.007 pJ
      - INT8 MUL: 0.070 pJ
    """

    ENERGY_FP16_ADD_PJ = 0.16
    ENERGY_FP16_MUL_PJ = 0.34
    ENERGY_INT8_ADD_PJ = 0.007
    ENERGY_INT8_MUL_PJ = 0.070

    @classmethod
    def arithmetic_energy_reduction(cls) -> float:
        """
        BitNet GEMM replaces FP16 multiply-accumulate with INT8 addition only:
          0.50 pJ / 0.007 pJ = 71.428...x
        """
        fp16_mac = cls.ENERGY_FP16_ADD_PJ + cls.ENERGY_FP16_MUL_PJ
        int8_add = cls.ENERGY_INT8_ADD_PJ
        return fp16_mac / int8_add

    @classmethod
    def system_decoding_energy_comparison(cls) -> Dict[str, float]:
        """Empirical system decoding energy from Table 1 (Surface Laptop Studio 2 / 7nm)."""
        return {
            "BitNet_b1.58_2B": 0.028,      # Joules
            "LLaMA_3.2_1B": 0.258,        # Joules (9.2x higher)
            "Gemma_3_1B": 0.186,          # Joules (6.6x higher)
            "Qwen_2.5_1.5B": 0.347,       # Joules (12.4x higher)
            "SmolLM2_1.7B": 0.425,        # Joules (15.2x higher)
            "MiniCPM_2B": 0.649           # Joules (23.2x higher)
        }
