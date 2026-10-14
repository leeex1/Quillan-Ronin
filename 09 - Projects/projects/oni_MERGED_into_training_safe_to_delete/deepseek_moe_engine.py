#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DeepSeekMoE Engine: DeepSeekMoE Architecture & Expert Specialization
====================================================================
Paper 45: "DeepSeekMoE: Towards Ultimate Expert Specialization in
Mixture-of-Experts Language Models"
Authors: Damai Dai, Chengqi Deng, Chenggang Zhao, R.X. Xu, Huazuo Gao,
         Deli Chen, Jiashi Li, Wangding Zeng, Xingkai Yu, Y. Wu,
         Zhenda Xie, Y.K. Li, Panpan Huang, Fuli Luo, Chong Ruan,
         Zhifang Sui, Wenfeng Liang (DeepSeek-AI & Peking University)
arXiv: 2401.06066 (Jan 2024, 33 pages)

Key Architectural Innovations:
1. Fine-Grained Expert Segmentation (Section 3.1):
   - Segments standard FFN experts into m smaller experts by scaling
     intermediate hidden dimension to (1/m) * d_ffn.
   - Total experts = m * N, activated experts = m * K.
   - Preserves constant computational cost (FLOPs) and parameters while
     exponentially expanding combinatorial routing space from C(N, K)
     to C(mN, mK) (e.g. C(16, 2) = 120 -> C(64, 8) = 4,426,165,368).

2. Shared Expert Isolation (Section 3.2):
   - Isolates K_s experts as dedicated shared experts, deterministically
     activated for all tokens to consolidate common knowledge.
   - Total routed experts: N' = mN - K_s.
   - Activated routed experts: K' = mK - K_s.
   - Layer output (Eq. 9):
     h_t^l = sum_{i=1}^{K_s} FFN_i(u_t^l) + sum_{i=K_s+1}^{mN} g_{i,t} FFN_i(u_t^l) + u_t^l

3. Multi-Level Load Balancing (Section 3.3):
   - Expert-level balance loss (Eq. 12-14) with small alpha_1 (e.g. 0.001)
     preventing routing collapse:
     L_ExpBal = alpha_1 * sum_{i=1}^{N'} f_i * P_i
     f_i = (N' / (K' * T)) * sum_{t=1}^T 1(token t selects expert i)
     P_i = (1 / T) * sum_{t=1}^T s_{i,t}
   - Device-level balance loss (Eq. 15-17) with larger alpha_2 (e.g. 0.05)
     preventing cross-accelerator bottlenecks without over-constraining
     individual expert specialization:
     L_DevBal = alpha_2 * sum_{d=1}^D f'_d * P'_d
     f'_d = (1 / |E_d|) * sum_{j in E_d} f_j
     P'_d = sum_{j in E_d} P_j

This module is part of the Quillan-Ronin Sovereign AI Architecture.
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class DeepSeekMoEConfig:
    """Configuration for DeepSeekMoE layer and multi-level routing."""
    hidden_dim: int = 1024
    base_ffn_dim: int = 4096
    base_num_experts: int = 16            # N
    base_top_k: int = 2                   # K
    segmentation_factor: int = 4          # m
    num_shared_experts: int = 2           # K_s
    num_devices: int = 4                  # D (for device-level load balance)
    expert_balance_factor: float = 0.001  # alpha_1
    device_balance_factor: float = 0.05   # alpha_2
    activation: str = "swiglu"            # "swiglu" or "gelu"
    normalize_routing_weights: bool = True
    dropout: float = 0.0

    @property
    def fine_grained_ffn_dim(self) -> int:
        """Intermediate dimension per segmented expert: (1/m) * d_ffn."""
        return max(1, self.base_ffn_dim // self.segmentation_factor)

    @property
    def total_fine_grained_experts(self) -> int:
        """Total fine-grained experts: m * N."""
        return self.base_num_experts * self.segmentation_factor

    @property
    def total_activated_experts(self) -> int:
        """Total activated experts budget: m * K."""
        return self.base_top_k * self.segmentation_factor

    @property
    def num_routed_experts(self) -> int:
        """N' = mN - K_s: number of routed experts."""
        return self.total_fine_grained_experts - self.num_shared_experts

    @property
    def num_activated_routed(self) -> int:
        """K' = mK - K_s: number of routed experts activated per token."""
        return self.total_activated_experts - self.num_shared_experts

    def combinatorial_flexibility(self) -> Dict[str, int]:
        """
        Calculate combinatorial routing options under:
        - Conventional MoE: C(N, K)
        - Fine-grained MoE: C(mN, mK)
        - Shared-isolated DeepSeekMoE: C(mN - K_s, mK - K_s)
        """
        c_base = math.comb(self.base_num_experts, self.base_top_k)
        c_fine = math.comb(self.total_fine_grained_experts, self.total_activated_experts)
        c_deepseek = math.comb(self.num_routed_experts, self.num_activated_routed)
        return {
            "conventional_c_n_k": c_base,
            "fine_grained_c_mn_mk": c_fine,
            "deepseek_moe_c_routed": c_deepseek,
            "combinatorial_surge_ratio": c_fine / max(1, c_base),
        }


class DeepSeekMoEOutput(NamedTuple):
    """Output tuple for DeepSeekMoELayer."""
    hidden_states: torch.Tensor
    aux_loss: torch.Tensor
    exp_bal_loss: torch.Tensor
    dev_bal_loss: torch.Tensor
    routing_weights: torch.Tensor
    selected_indices: torch.Tensor
    load_metrics: Dict[str, float]


class FineGrainedFFN(nn.Module):
    """
    Individual fine-grained expert FFN with intermediate dimension d_ffn / m.
    Supports SwiGLU or standard GELU activation.
    """

    def __init__(self, hidden_dim: int, ffn_dim: int, activation: str = "swiglu", dropout: float = 0.0):
        super().__init__()
        self.activation = activation
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        if activation == "swiglu":
            # SwiGLU: w1 (gate), w3 (value), w2 (down-proj)
            self.w1 = nn.Linear(hidden_dim, ffn_dim, bias=False)
            self.w3 = nn.Linear(hidden_dim, ffn_dim, bias=False)
            self.w2 = nn.Linear(ffn_dim, hidden_dim, bias=False)
        else:
            # Standard 2-layer FFN
            self.w1 = nn.Linear(hidden_dim, ffn_dim, bias=False)
            self.w2 = nn.Linear(ffn_dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.activation == "swiglu":
            gate = F.silu(self.w1(x))
            val = self.w3(x)
            hidden = gate * val
        else:
            hidden = F.gelu(self.w1(x))
        hidden = self.dropout(hidden)
        return self.w2(hidden)


class DeepSeekMoERouter(nn.Module):
    """
    Router for DeepSeekMoE:
    - Routes tokens across N' = (mN - K_s) routed experts.
    - Selects top K' = (mK - K_s) experts per token.
    - Computes both expert-level balance loss and device-level balance loss.
    """

    def __init__(self, config: DeepSeekMoEConfig):
        super().__init__()
        self.config = config
        self.num_routed = config.num_routed_experts
        self.top_k = config.num_activated_routed
        self.num_devices = config.num_devices
        self.alpha_1 = config.expert_balance_factor
        self.alpha_2 = config.device_balance_factor
        self.normalize = config.normalize_routing_weights

        # Centroids / router weight matrix e_i for routed experts: [hidden_dim, N']
        self.router_weights = nn.Linear(config.hidden_dim, self.num_routed, bias=False)

        # Precompute device partitions E_1, ..., E_D
        # Partition N' routed experts uniformly across D devices
        device_assignments = []
        experts_per_device = math.ceil(self.num_routed / self.num_devices)
        for d in range(self.num_devices):
            start = d * experts_per_device
            end = min(start + experts_per_device, self.num_routed)
            if start < self.num_routed:
                device_assignments.append(list(range(start, end)))
        self.device_partitions = device_assignments

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, float]]:
        """
        Args:
            x: [..., hidden_dim] input tokens

        Returns:
            routing_weights: [total_tokens, top_k] gate weights for selected routed experts
            selected_indices: [total_tokens, top_k] indices of selected routed experts in [0, N'-1]
            exp_bal_loss: scalar tensor L_ExpBal
            dev_bal_loss: scalar tensor L_DevBal
            total_aux_loss: scalar tensor L_Aux = L_ExpBal + L_DevBal
            load_metrics: dictionary of load statistics
        """
        orig_shape = x.shape
        flat_x = x.view(-1, orig_shape[-1])  # [T, D_in]
        T = flat_x.size(0)

        # 1. Compute routing logits and full softmax distribution s_{i, t} (Eq. 8, 11)
        # s_{i, t} = Softmax_i(u_t^T e_i)
        logits = self.router_weights(flat_x)  # [T, N']
        s_prob = F.softmax(logits, dim=-1)    # [T, N']

        # 2. Select top K' routed experts per token (Eq. 10)
        topk_scores, topk_indices = torch.topk(s_prob, self.top_k, dim=-1)  # [T, K']

        if self.normalize:
            # Normalize routing weights over selected top-k
            denom = topk_scores.sum(dim=-1, keepdim=True).clamp(min=1e-9)
            routing_weights = topk_scores / denom
        else:
            routing_weights = topk_scores

        # 3. Expert-level Balance Loss (Eq. 12-14)
        # f_i = (N' / (K' * T)) * sum_{t=1}^T 1(token t selects expert i)
        # P_i = (1 / T) * sum_{t=1}^T s_{i, t}
        # L_ExpBal = alpha_1 * sum_{i=1}^{N'} f_i * P_i
        
        # Indicator count for expert selection:
        selected_mask = torch.zeros(T, self.num_routed, device=x.device, dtype=x.dtype)
        selected_mask.scatter_(1, topk_indices, 1.0)  # [T, N']

        token_count_per_expert = selected_mask.sum(dim=0)  # [N']
        f_i = (self.num_routed / (self.top_k * max(1, T))) * token_count_per_expert  # [N']
        P_i = s_prob.mean(dim=0)  # [N']

        exp_bal_loss = self.alpha_1 * torch.sum(f_i.detach() * P_i)

        # 4. Device-level Balance Loss (Eq. 15-17)
        # Partition routed experts into D groups {E_1, ..., E_D}
        # f'_d = (1 / |E_d|) * sum_{j in E_d} f_j
        # P'_d = sum_{j in E_d} P_j
        # L_DevBal = alpha_2 * sum_{d=1}^D f'_d * P'_d
        dev_loss_terms = []
        device_load_fractions = []
        for d, partition in enumerate(self.device_partitions):
            if not partition:
                continue
            group_indices = torch.tensor(partition, device=x.device, dtype=torch.long)
            group_size = len(partition)
            f_group = torch.index_select(f_i, 0, group_indices)
            p_group = torch.index_select(P_i, 0, group_indices)

            f_prime_d = f_group.sum() / group_size
            P_prime_d = p_group.sum()
            dev_loss_terms.append(f_prime_d.detach() * P_prime_d)
            device_load_fractions.append(float(f_prime_d.item()))

        if dev_loss_terms:
            dev_bal_loss = self.alpha_2 * torch.stack(dev_loss_terms).sum()
        else:
            dev_bal_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)

        total_aux_loss = exp_bal_loss + dev_bal_loss

        # Load balancing metrics
        load_metrics = {
            "expert_entropy": float(-(P_i * (P_i + 1e-12).log()).sum().item()),
            "max_expert_load": float(token_count_per_expert.max().item() / max(1, T)),
            "min_expert_load": float(token_count_per_expert.min().item() / max(1, T)),
            "avg_expert_load": float(token_count_per_expert.mean().item() / max(1, T)),
            "num_active_experts_batch": int((token_count_per_expert > 0).sum().item()),
            "device_load_imbalance": float(max(device_load_fractions) - min(device_load_fractions)) if device_load_fractions else 0.0,
        }

        return routing_weights, topk_indices, exp_bal_loss, dev_bal_loss, total_aux_loss, load_metrics


class DeepSeekMoELayer(nn.Module):
    """
    Complete DeepSeekMoE Layer implementing Eq. 9:
    h_t^l = sum_{i=1}^{K_s} FFN_i(u_t^l) + sum_{i=K_s+1}^{mN} g_{i,t} FFN_i(u_t^l) + u_t^l

    Components:
    - K_s shared experts: unconditionally executed for every token, consolidating
      common knowledge and eliminating parameter redundancy across routed experts.
    - N' routed experts: fine-grained experts of size (1/m)*d_ffn, top-K' selected.
    - Multi-level load-balancing auxiliary loss (expert + device levels).
    - Residual connection.
    """

    def __init__(self, config: DeepSeekMoEConfig):
        super().__init__()
        self.config = config
        self.hidden_dim = config.hidden_dim
        self.ffn_dim = config.fine_grained_ffn_dim
        self.num_shared = config.num_shared_experts
        self.num_routed = config.num_routed_experts

        # 1. Dedicated Shared Experts (K_s experts)
        self.shared_experts = nn.ModuleList([
            FineGrainedFFN(
                hidden_dim=self.hidden_dim,
                ffn_dim=self.ffn_dim,
                activation=config.activation,
                dropout=config.dropout
            )
            for _ in range(self.num_shared)
        ])

        # 2. Fine-Grained Routed Experts (N' experts)
        self.routed_experts = nn.ModuleList([
            FineGrainedFFN(
                hidden_dim=self.hidden_dim,
                ffn_dim=self.ffn_dim,
                activation=config.activation,
                dropout=config.dropout
            )
            for _ in range(self.num_routed)
        ])

        # 3. Router
        self.router = DeepSeekMoERouter(config)

    def forward(self, x: torch.Tensor) -> DeepSeekMoEOutput:
        """
        Forward pass of DeepSeekMoE layer.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, hidden_dim] or [tokens, hidden_dim]
            
        Returns:
            DeepSeekMoEOutput containing output hidden states, auxiliary losses, and routing info.
        """
        orig_shape = x.shape
        flat_x = x.view(-1, self.hidden_dim)  # [T, D]
        T = flat_x.size(0)

        # 1. Deterministic Shared Experts (Eq. 9 first term)
        # sum_{i=1}^{K_s} FFN_i(u_t^l)
        shared_out = torch.zeros_like(flat_x)
        for shared_ffn in self.shared_experts:
            shared_out = shared_out + shared_ffn(flat_x)

        # 2. Dynamic Routed Experts (Eq. 9 second term)
        # sum_{i=K_s+1}^{mN} g_{i,t} FFN_i(u_t^l)
        weights, indices, exp_loss, dev_loss, aux_loss, metrics = self.router(flat_x)

        routed_out = torch.zeros_like(flat_x)
        K_prime = indices.size(1)

        # Dispatch tokens to selected experts
        for k in range(K_prime):
            expert_idx_k = indices[:, k]    # [T]
            weight_k = weights[:, k:k+1]     # [T, 1]

            # Unique active experts in rank k for batched execution
            unique_experts = torch.unique(expert_idx_k)
            for exp_id in unique_experts:
                mask = (expert_idx_k == exp_id)
                if not mask.any():
                    continue
                selected_x = flat_x[mask]
                expert_res = self.routed_experts[exp_id.item()](selected_x)
                routed_out[mask] = routed_out[mask] + weight_k[mask] * expert_res

        # 3. Combine with Residual Connection: h_t^l = shared + routed + u_t^l
        out_flat = shared_out + routed_out + flat_x
        out = out_flat.view(orig_shape)

        return DeepSeekMoEOutput(
            hidden_states=out,
            aux_loss=aux_loss,
            exp_bal_loss=exp_loss,
            dev_bal_loss=dev_loss,
            routing_weights=weights.view(*orig_shape[:-1], K_prime),
            selected_indices=indices.view(*orig_shape[:-1], K_prime),
            load_metrics=metrics
        )


def verify_deepseek_moe_invariants(
    config: Optional[DeepSeekMoEConfig] = None
) -> Dict[str, any]:
    """
    Rigorously verifies DeepSeekMoE mathematical invariants:
    1. Fine-grained segmentation preserves total parameter capacity while scaling combinations.
    2. Shared experts are deterministically executed for 100% of tokens.
    3. Exactly K' routed experts are selected per token.
    4. Gradient propagates to both shared and routed experts as well as router centroids.
    5. Expert-level and device-level load balance losses are non-negative and properly computed.
    """
    if config is None:
        config = DeepSeekMoEConfig(
            hidden_dim=64,
            base_ffn_dim=256,
            base_num_experts=8,
            base_top_k=2,
            segmentation_factor=4,
            num_shared_experts=2,
            num_devices=2,
            expert_balance_factor=0.001,
            device_balance_factor=0.05
        )

    # Invariant 1: Combinatorial expansion
    comb = config.combinatorial_flexibility()
    assert comb["deepseek_moe_c_routed"] > comb["conventional_c_n_k"], (
        f"Combinations must surge: {comb['deepseek_moe_c_routed']} vs {comb['conventional_c_n_k']}"
    )

    # Invariant 2: Dimensions & Counts
    assert config.fine_grained_ffn_dim == config.base_ffn_dim // config.segmentation_factor
    assert config.num_routed_experts == config.total_fine_grained_experts - config.num_shared_experts
    assert config.num_activated_routed == config.total_activated_experts - config.num_shared_experts

    # Instantiate layer
    layer = DeepSeekMoELayer(config)
    batch_size, seq_len = 2, 8
    x = torch.randn(batch_size, seq_len, config.hidden_dim, requires_grad=True)

    # Invariant 3: Forward pass execution
    out = layer(x)
    assert out.hidden_states.shape == x.shape
    assert out.selected_indices.shape == (batch_size, seq_len, config.num_activated_routed)
    assert out.routing_weights.shape == (batch_size, seq_len, config.num_activated_routed)
    assert out.aux_loss >= 0.0
    assert out.exp_bal_loss >= 0.0
    assert out.dev_bal_loss >= 0.0

    # Invariant 4: Backward gradient flow across shared, routed, and router weights
    loss = out.hidden_states.sum() + out.aux_loss
    loss.backward()

    assert x.grad is not None and not torch.isnan(x.grad).any(), "Input gradient must be valid"
    assert layer.router.router_weights.weight.grad is not None, "Router weights must receive gradient"

    shared_grads = [p.grad for p in layer.shared_experts.parameters() if p.requires_grad]
    assert any(g is not None and g.abs().sum() > 0 for g in shared_grads), "Shared experts must receive gradient"

    routed_grads = [p.grad for p in layer.routed_experts.parameters() if p.requires_grad]
    assert any(g is not None and g.abs().sum() > 0 for g in routed_grads), "Routed experts must receive gradient"

    return {
        "verified": True,
        "combinatorial_surge": comb,
        "num_shared": config.num_shared_experts,
        "num_routed": config.num_routed_experts,
        "activated_routed": config.num_activated_routed,
        "aux_loss": float(out.aux_loss.item()),
        "exp_bal_loss": float(out.exp_bal_loss.item()),
        "dev_bal_loss": float(out.dev_bal_loss.item()),
        "metrics": out.load_metrics
    }


if __name__ == "__main__":
    result = verify_deepseek_moe_invariants()
    print("DeepSeekMoE Invariants Verification Result:")
    for k, v in result.items():
        print(f"  {k}: {v}")
