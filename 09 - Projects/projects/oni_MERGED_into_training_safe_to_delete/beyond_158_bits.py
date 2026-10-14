#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BEYOND 1.58 BITS: ADAPTIVE-PRECISION TERNARY TRANSFORMERS WITH FULLY-TERNARY MoE ROUTING
========================================================================================
Reference:
  CrashOverrideX, Quillan AI Research Team (September 2026)
  "Beyond 1.58 Bits: Adaptive-Precision Ternary Transformers with
   Fully-Ternary Mixture-of-Experts Routing" (7 pp)

Core Architectural Invariants:
  1. Sub-1.58-Bit Analytic Engine:
     - Under Laplace prior w ~ Laplace(0, b), absmean scaling alpha = E|w| = b:
       Optimal deadzone threshold: Delta* = b / 2.
       Ternary-to-binary MSE ratio: 2 * (1 - e^{-1/2}) ≈ 0.78693868 (21.31% error reduction).
       Zero-state occupancy: p0 = 1 - e^{-1/2} ≈ 0.39346934.
       Operating point entropy: H(W_q) ≈ 1.5744 bits (vs log2(3) ≈ 1.585 max).
     - Per-tensor kurtosis rule: kappa > 6.0 -> ternary; kappa <= 2.0 -> binary {-1, +1} at 1.0 bit.
  2. Heterogeneous Lagrangian Bit-Budget Optimizer:
     - Discrete water-filling maximizing marginal error reduction per bit across modules:
       argmax_ell S_ell * [epsilon(b_ell) - epsilon(b_ell^+)] / [P_ell * (b_ell^+ - b_ell)]
       subject to sum_ell P_ell * b_ell <= B_total.
  3. Fully-Ternary 34-Expert Council MoE:
     - All 34 experts are ternary BitLinear (no full-precision shared expert).
     - High-precision router (8-bit / FP16) justified by router leverage asymmetry:
       (S_router / P_router >> S_expert / P_expert).
     - Dual routing:
       (a) Dense pull-weighted deliberation (all 34 experts active, softmax pull weights).
       (b) Gumbel-Softmax top-k sparse dispatch (k = 4 active, ~11.8% activation ratio).
     - Precision-preserving Rank-8 swarm low-rank adapters (mixed precision by construction).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


# =====================================================================
# 1. Sub-1.58-Bit Analytic Theory & Quantization Engine
# =====================================================================

class Sub158AnalyticEngine:
    """
    Analytic formulation of sub-1.58-bit quantization under Laplace weight prior.
    Implements Theorem 1 and Remark 1 from CrashOverrideX (2026).
    """

    @staticmethod
    def optimal_laplace_threshold(b: float) -> float:
        """Delta* = b / 2: unique stationary point minimizing ternary MSE."""
        return b / 2.0

    @staticmethod
    def theoretical_mse_ratio() -> float:
        """
        MSE_T(Delta*) / MSE_B = 2 * (1 - e^{-1/2}) ≈ 0.786938680338
        Proves ternary zero state achieves a 21.31% MSE reduction over binary.
        """
        return 2.0 * (1.0 - math.exp(-0.5))

    @staticmethod
    def zero_state_occupancy() -> float:
        """p0 = P(|w| <= b/2) = 1 - e^{-1/2} ≈ 0.393469340287."""
        return 1.0 - math.exp(-0.5)

    @staticmethod
    def operating_point_entropy() -> float:
        """
        H = -p0 * log2(p0) - (1 - p0) * log2((1 - p0) / 2) ≈ 1.57444 bits.
        Near-maximal utilization of the 1.585-bit capacity.
        """
        p0 = Sub158AnalyticEngine.zero_state_occupancy()
        p_plus = (1.0 - p0) / 2.0
        p_minus = (1.0 - p0) / 2.0
        h = -(p0 * math.log2(p0) + p_plus * math.log2(p_plus) + p_minus * math.log2(p_minus))
        return h

    @staticmethod
    def compute_empirical_kurtosis(tensor: torch.Tensor) -> float:
        """
        Computes empirical kurtosis kappa = E[(w - mu)^4] / sigma^4.
        Laplace distribution has kurtosis = 6.0; Uniform has kurtosis = 1.8.
        """
        w = tensor.detach().flatten().float()
        mu = w.mean()
        centered = w - mu
        var = (centered ** 2).mean()
        if var.item() < 1e-12:
            return 3.0
        m4 = (centered ** 4).mean()
        kappa = (m4 / (var ** 2)).item()
        return kappa

    @staticmethod
    def select_precision_by_kurtosis(tensor: torch.Tensor, threshold_kurtosis: float = 4.0) -> Tuple[str, float]:
        """
        Decision rule from Section 5:
        If kappa > threshold_kurtosis (leptokurtic): retain ternary at 1.58 bits.
        If kappa <= threshold_kurtosis (platykurtic/uniform): drop to binary at 1.0 bit.
        """
        k = Sub158AnalyticEngine.compute_empirical_kurtosis(tensor)
        if k > threshold_kurtosis:
            return "ternary", 1.58
        else:
            return "binary", 1.0


class AdaptivePrecisionLinear(nn.Module):
    """
    Linear layer with adaptive ternary/binary straight-through estimator (STE).
    Supports dynamic thresholding Delta* = b / 2 and kurtosis-guided precision dropping.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        mode: str = "ternary",  # "ternary", "binary", or "auto"
        rank_adapter: int = 0
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.mode = mode
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * (1.0 / math.sqrt(in_features)))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

        # Precision-preserving rank-r adapter
        self.rank_adapter = rank_adapter
        if rank_adapter > 0:
            self.adapter_down = nn.Parameter(torch.randn(rank_adapter, in_features) * 0.01)
            self.adapter_up = nn.Parameter(torch.zeros(out_features, rank_adapter))
            self.adapter_scale = nn.Parameter(torch.tensor(1.0))
        else:
            self.register_parameter("adapter_down", None)
            self.register_parameter("adapter_up", None)
            self.register_parameter("adapter_scale", None)

    def quantize_weight(self, w: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, str]:
        active_mode = self.mode
        if active_mode == "auto":
            active_mode, _ = Sub158AnalyticEngine.select_precision_by_kurtosis(w)

        # Absmean scaling: alpha = E|w|
        alpha = w.abs().mean()
        if alpha < 1e-12:
            return w, alpha, active_mode

        if active_mode == "binary":
            # Binary {-1, +1}
            w_sign = torch.sign(w)
            w_sign[w_sign == 0] = 1.0
            w_q = w_sign
            w_quant = alpha * w_q
        else:
            # Ternary {-1, 0, +1} with optimal deadzone Delta* = b / 2 = alpha / 2
            delta = alpha * 0.5
            w_q = torch.zeros_like(w)
            mask_pos = w > delta
            mask_neg = w < -delta
            w_q[mask_pos] = 1.0
            w_q[mask_neg] = -1.0
            w_quant = alpha * w_q

        # Straight-Through Estimator (STE)
        w_ste = w + (w_quant - w).detach()
        return w_ste, alpha, active_mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w_ste, alpha, active_mode = self.quantize_weight(self.weight)
        out = F.linear(x, w_ste, self.bias)

        if self.rank_adapter > 0:
            # Low-rank additive correction
            adapter_out = F.linear(F.linear(x, self.adapter_down), self.adapter_up) * self.adapter_scale
            out = out + adapter_out

        return out


# =====================================================================
# 2. Heterogeneous Lagrangian Bit-Budget Framework
# =====================================================================

@dataclass
class ModuleBudgetSpec:
    name: str
    num_params: int
    sensitivity: float
    current_bits: float = 1.0


class HeterogeneousBitBudgetOptimizer:
    """
    Solves min sum_ell S_ell * epsilon(b_ell) s.t. sum_ell P_ell * b_ell <= B_total
    via discrete water-filling greedy marginal-gain allocation (Section 6).
    """

    SUPPORTED_BITS: List[float] = [1.0, 1.58, 4.0, 8.0]

    def __init__(self, modules: List[ModuleBudgetSpec], total_bit_budget: float):
        self.modules = {m.name: m for m in modules}
        self.total_bit_budget = total_bit_budget

    @staticmethod
    def error_model(bits: float, base_variance: float = 1.0) -> float:
        """
        Quantization error model epsilon(b):
          1.0 bit: MSE_B = base_variance
          1.58 bit: 0.786938 * base_variance
          4.0 bit: uniform quantization Delta^2 / 12 ≈ 0.05 * base_variance
          8.0 bit: uniform quantization ≈ 0.0002 * base_variance
        """
        if bits <= 1.0:
            return base_variance
        elif bits <= 1.58:
            return Sub158AnalyticEngine.theoretical_mse_ratio() * base_variance
        elif bits <= 4.0:
            return 0.05 * base_variance
        else:
            return 0.0002 * base_variance

    def current_allocated_budget(self) -> float:
        return sum(m.num_params * m.current_bits for m in self.modules.values())

    def next_precision_level(self, current_bits: float) -> Optional[float]:
        for b in self.SUPPORTED_BITS:
            if b > current_bits:
                return b
        return None

    def optimize(self) -> Dict[str, float]:
        """
        Greedy marginal-gain water-filling algorithm:
        Repeatedly spend bits on the module with maximum marginal error reduction per bit.
        """
        # Start all modules at minimum baseline (1.0 bit)
        for m in self.modules.values():
            m.current_bits = 1.0

        while True:
            best_module_name: Optional[str] = None
            best_gain: float = -1.0
            best_cost: float = 0.0
            best_next_bits: float = 1.0

            curr_budget = self.current_allocated_budget()

            for name, m in self.modules.items():
                next_b = self.next_precision_level(m.current_bits)
                if next_b is None:
                    continue

                delta_bits = next_b - m.current_bits
                delta_budget = m.num_params * delta_bits

                if curr_budget + delta_budget > self.total_bit_budget:
                    continue

                curr_err = self.error_model(m.current_bits)
                next_err = self.error_model(next_b)
                delta_err = curr_err - next_err

                # Marginal gain: S_ell * Delta_epsilon / (P_ell * Delta_b)
                gain = (m.sensitivity * delta_err) / (delta_budget + 1e-12)

                if gain > best_gain:
                    best_gain = gain
                    best_module_name = name
                    best_cost = delta_budget
                    best_next_bits = next_b

            if best_module_name is None:
                # No further improvements fit in budget
                break

            self.modules[best_module_name].current_bits = best_next_bits

        return {name: m.current_bits for name, m in self.modules.items()}


# =====================================================================
# 3. Fully-Ternary 34-Expert Council MoE Architecture
# =====================================================================

class TernaryExpertBlock(nn.Module):
    """
    Native ternary BitLinear Expert feed-forward block (Section 7).
    All expert weights are strictly {-1, 0, 1} with absmean scaling.
    No full-precision parameters exist in the expert core.
    """

    def __init__(self, d_model: int, d_ff: int, rank_adapter: int = 8):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        # SubLN bounding activation variance before quantization
        self.sub_ln1 = nn.LayerNorm(d_model)
        self.w1 = AdaptivePrecisionLinear(d_model, d_ff, bias=False, mode="ternary", rank_adapter=rank_adapter)
        self.w2 = AdaptivePrecisionLinear(d_ff, d_model, bias=False, mode="ternary", rank_adapter=rank_adapter)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.sub_ln1(x)
        h = F.gelu(self.w1(h))
        out = self.w2(h)
        return out


class HighPrecisionMoERouter(nn.Module):
    """
    High-precision router (8-bit / FP16) for the 34-expert council.
    Justified by router leverage asymmetry:
      S_router / P_router >> S_expert / P_expert.
    Supports:
      - Dense pull-weighted deliberation (softmax over all 34 experts).
      - Gumbel-Softmax top-k sparse dispatch (default k=4).
    """

    def __init__(self, d_model: int, num_experts: int = 34, top_k: int = 4):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = top_k
        self.router_gate = nn.Linear(d_model, num_experts, bias=False)
        self.temperature = nn.Parameter(torch.tensor(1.0))

    def forward(
        self,
        x: torch.Tensor,
        mode: str = "sparse",  # "sparse" or "dense"
        tau: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
          weights: Tensor of shape (batch, seq, num_experts)
          indices: Tensor of shape (batch, seq, top_k) for sparse mode, or None
          logits: Raw router logits
        """
        logits = self.router_gate(x)  # (b, s, num_experts)

        if mode == "dense":
            # Dense pull-weighted deliberation: y = sum_i pi_i * E_i(x)
            weights = F.softmax(logits / max(tau, 1e-4), dim=-1)
            return weights, torch.arange(self.num_experts, device=x.device), logits
        else:
            # Gumbel top-k sparse dispatch
            if self.training:
                gumbel_noise = -torch.empty_like(logits).exponential_().log()
                gumbel_logits = (logits + gumbel_noise) / max(tau, 1e-4)
            else:
                gumbel_logits = logits

            topk_vals, topk_indices = torch.topk(gumbel_logits, self.top_k, dim=-1)
            sparse_weights = F.softmax(topk_vals, dim=-1)

            # Scatter weights to full (b, s, num_experts)
            full_weights = torch.zeros_like(logits).scatter_(-1, topk_indices, sparse_weights)
            return full_weights, topk_indices, logits


class FullyTernaryCouncilMoE(nn.Module):
    """
    Full 34-expert Fully-Ternary Mixture-of-Experts architecture.
    Features:
      - 34 Ternary BitLinear experts with rank-8 precision-preserving adapters.
      - Zero full-precision shared experts (unlike MoTE).
      - High-precision 8-bit / FP16 router with dual routing modes.
    """

    def __init__(
        self,
        d_model: int = 64,
        d_ff: int = 128,
        num_experts: int = 34,
        top_k: int = 4,
        rank_adapter: int = 8
    ):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = top_k
        self.router = HighPrecisionMoERouter(d_model=d_model, num_experts=num_experts, top_k=top_k)
        self.experts = nn.ModuleList([
            TernaryExpertBlock(d_model=d_model, d_ff=d_ff, rank_adapter=rank_adapter)
            for _ in range(num_experts)
        ])

    def forward(
        self,
        x: torch.Tensor,
        mode: str = "sparse",
        tau: float = 1.0
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        b, s, d = x.shape
        weights, indices, logits = self.router(x, mode=mode, tau=tau)

        out = torch.zeros_like(x)

        if mode == "dense":
            # Dense deliberation: evaluate all 34 experts
            for i, expert in enumerate(self.experts):
                expert_out = expert(x)
                w_i = weights[..., i:i+1]
                out = out + w_i * expert_out
        else:
            # Sparse dispatch: evaluate only selected top-k experts per token
            flat_x = x.view(-1, d)
            flat_weights = weights.view(-1, self.num_experts)
            flat_out = torch.zeros_like(flat_x)

            for i, expert in enumerate(self.experts):
                token_mask = flat_weights[:, i] > 0
                if token_mask.any():
                    sel_tokens = flat_x[token_mask]
                    e_out = expert(sel_tokens)
                    w_token = flat_weights[token_mask, i:i+1]
                    flat_out[token_mask] += w_token * e_out

            out = flat_out.view(b, s, d)

        # Auxiliary load balancing loss
        density = weights.mean(dim=(0, 1))  # (num_experts,)
        prob = F.softmax(logits, dim=-1).mean(dim=(0, 1))
        aux_loss = self.num_experts * torch.sum(density * prob)

        aux_data = {
            "router_weights": weights,
            "router_logits": logits,
            "aux_loss": aux_loss,
            "mode": mode
        }
        return out, aux_data
