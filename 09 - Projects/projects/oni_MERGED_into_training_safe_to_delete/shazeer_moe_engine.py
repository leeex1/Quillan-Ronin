#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shazeer Sparsely-Gated Mixture-of-Experts Engine
Paper 69: Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer
Authors: Noam Shazeer, Azalia Mirhoseini, Krzysztof Maziarz, Andy Davis, Quoc Le, Geoffrey Hinton, Jeff Dean
Affiliation: Google Brain & Jagiellonian University (ICLR 2017 / arXiv:1701.06538)

Key Architectural & Mathematical Invariants:
1. Sparsely-Gated MoE Formulation (Section 2, Eq. 1):
   y = sum_{i=1}^n G(x)_i E_i(x)
   where n is the total number of expert networks, E_i(x) is the expert computation,
   and G(x) is an n-dimensional sparse gating vector.
2. Noisy Top-K Gating (Section 2, Eqs. 2-5):
   H(x)_i = (x * W_g)_i + StandardNormal() * Softplus((x * W_noise)_i)
   KeepTopK(v, k)_i = v_i if v_i in top-k else -infinity
   G(x) = Softmax(KeepTopK(H(x), k))
   guaranteeing exactly k non-zero gates summing strictly to 1.0.
3. Importance Loss (Section 4, Eqs. 6-7):
   Importance(X) = sum_{x in X} G(x)
   L_importance(X) = w_importance * CV(Importance(X))^2
   where CV is the coefficient of variation (std / mean).
4. Smooth Normal CDF Load Balancing Loss (Appendix A, Eqs. 8-11):
   P(x, i) = Phi( ((x * W_g)_i - kth_excluding(H(x), k, i)) / Softplus((x * W_noise)_i) )
   Load(X)_i = sum_{x in X} P(x, i)
   L_load(X) = w_load * CV(Load(X))^2
5. Hierarchical Mixture-of-Experts (Appendix B):
   Two-level hierarchical routing decomposing n = A * B experts into A primary groups
   and B secondary experts per group, scaling to thousands of experts.
"""

from dataclasses import dataclass
import math
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ShazeerMoEConfig:
    """Configuration for Sparsely-Gated Mixture-of-Experts."""
    d_model: int = 256
    d_hidden: int = 1024
    num_experts: int = 16
    k: int = 2
    w_importance: float = 0.1
    w_load: float = 0.1
    noisy_gating: bool = True
    dropout: float = 0.0
    activation: str = "relu"  # "relu" or "swiglu"


class ShazeerExpert(nn.Module):
    """
    Standard Feed-Forward Expert Sub-Network.
    E_i(x) = W_2 * Act(W_1 * x + b_1) + b_2
    """
    def __init__(self, d_model: int, d_hidden: int, activation: str = "relu", dropout: float = 0.0):
        super().__init__()
        self.d_model = d_model
        self.d_hidden = d_hidden
        self.activation_type = activation.lower()

        if self.activation_type == "swiglu":
            self.w_gate = nn.Linear(d_model, d_hidden, bias=False)
            self.w_up = nn.Linear(d_model, d_hidden, bias=False)
            self.w_down = nn.Linear(d_hidden, d_model, bias=False)
        else:
            self.fc1 = nn.Linear(d_model, d_hidden)
            self.fc2 = nn.Linear(d_hidden, d_model)

        self.drop = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.activation_type == "swiglu":
            h = F.silu(self.w_gate(x)) * self.w_up(x)
            out = self.w_down(self.drop(h))
        else:
            h = F.relu(self.fc1(x))
            out = self.fc2(self.drop(h))
        return out


class NoisyTopKGating(nn.Module):
    """
    Noisy Top-K Gating Network (Section 2 & Appendix A).
    Computes sparse continuous routing probabilities and smooth load-balancing losses.
    """
    def __init__(self, d_model: int, num_experts: int, k: int = 2,
                 w_importance: float = 0.1, w_load: float = 0.1, noisy_gating: bool = True):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.k = min(k, num_experts)
        self.w_importance = w_importance
        self.w_load = w_load
        self.noisy_gating = noisy_gating

        # Trainable parameters W_g and W_noise
        self.w_gate = nn.Parameter(torch.empty(d_model, num_experts))
        self.w_noise = nn.Parameter(torch.empty(d_model, num_experts))

        self.reset_parameters()

    def reset_parameters(self):
        """
        Initializes W_g and W_noise as specified in Appendix A:
        'To accomplish this, we initialize the matrices Wg and Wnoise to all zeros,
        which yields no signal and some noise.'
        """
        nn.init.zeros_(self.w_gate)
        nn.init.zeros_(self.w_noise)

    @staticmethod
    def _cv_squared(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        """
        Computes the square of the coefficient of variation:
        CV(x)^2 = Var(x) / (Mean(x)^2)
        """
        mean = x.mean()
        var = ((x - mean) ** 2).mean()
        return var / ((mean + eps) ** 2)

    def forward(self, x: torch.Tensor, train_mode: Optional[bool] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass for Noisy Top-K Gating.
        Args:
            x: Input tensor of shape [batch_size, seq_len, d_model] or [N, d_model].
            train_mode: Whether to sample Gaussian noise (defaults to self.training and self.noisy_gating).
        Returns:
            gates: [..., num_experts] sparse routing weights (exactly k non-zero per token, sums to 1.0).
            topk_indices: [..., k] indices of active experts.
            topk_gates: [..., k] values of active gates.
            loss_dict: Dictionary containing 'loss_importance', 'loss_load', and 'loss_aux'.
        """
        is_training = self.training if train_mode is None else train_mode
        orig_shape = x.shape[:-1]
        x_flat = x.reshape(-1, self.d_model)  # [Total_Tokens, d_model]

        # 1. Clean logits: x * W_g
        clean_logits = torch.matmul(x_flat, self.w_gate)  # [N, num_experts]

        # 2. Noise generation: StandardNormal() * Softplus(x * W_noise)
        raw_noise_std = torch.matmul(x_flat, self.w_noise)
        noise_std = F.softplus(raw_noise_std)

        if is_training and self.noisy_gating:
            eps = torch.randn_like(clean_logits)
            noisy_logits = clean_logits + eps * noise_std
        else:
            noisy_logits = clean_logits

        # 3. KeepTopK: identify top-k and top-(k+1)
        topk_vals, topk_indices = torch.topk(noisy_logits, self.k, dim=-1)
        kth_val = topk_vals[..., -1:]  # [N, 1]

        # If num_experts > k, get (k+1)-th value for smooth exclusion calculation
        if self.num_experts > self.k:
            topk_plus1_vals = torch.topk(noisy_logits, self.k + 1, dim=-1)[0]
            kplus1_val = topk_plus1_vals[..., -1:]  # [N, 1]
        else:
            kplus1_val = kth_val

        # Create sparse masked logits (Eq. 5)
        mask = noisy_logits >= kth_val
        masked_logits = torch.where(mask, noisy_logits, torch.full_like(noisy_logits, float('-inf')))

        # Softmax over masked logits (Eq. 3)
        gates_flat = F.softmax(masked_logits, dim=-1)  # [N, num_experts]
        topk_gates_flat = torch.gather(gates_flat, -1, topk_indices)

        # 4. Importance Loss (Section 4, Eqs. 6-7)
        importance = gates_flat.sum(dim=0)  # [num_experts]
        cv_importance_sq = self._cv_squared(importance)
        loss_importance = self.w_importance * cv_importance_sq

        # 5. Smooth Normal CDF Load Loss (Appendix A, Eqs. 8-11)
        kth_excluding = torch.where(mask, kplus1_val, kth_val)
        
        norm_arg = (clean_logits - kth_excluding) / (noise_std + 1e-6)
        prob = 0.5 * (1.0 + torch.erf(norm_arg / math.sqrt(2.0)))
        load = prob.sum(dim=0)  # [num_experts]
        cv_load_sq = self._cv_squared(load)
        loss_load = self.w_load * cv_load_sq

        loss_aux = loss_importance + loss_load

        # Reshape back to original dimensions
        gates = gates_flat.reshape(*orig_shape, self.num_experts)
        topk_indices = topk_indices.reshape(*orig_shape, self.k)
        topk_gates = topk_gates_flat.reshape(*orig_shape, self.k)

        losses = {
            "loss_importance": loss_importance,
            "loss_load": loss_load,
            "loss_aux": loss_aux,
            "cv_importance": torch.sqrt(cv_importance_sq + 1e-8),
            "cv_load": torch.sqrt(cv_load_sq + 1e-8)
        }

        return gates, topk_indices, topk_gates, losses


class ShazeerSparselyGatedMoELayer(nn.Module):
    """
    Sparsely-Gated Mixture-of-Experts Layer (Shazeer et al., 2017).
    Dispatches tokens dynamically to top-k experts and modulates outputs.
    """
    def __init__(self, config: ShazeerMoEConfig):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.num_experts = config.num_experts
        self.k = config.k

        # Gating network
        self.gating = NoisyTopKGating(
            d_model=config.d_model,
            num_experts=config.num_experts,
            k=config.k,
            w_importance=config.w_importance,
            w_load=config.w_load,
            noisy_gating=config.noisy_gating
        )

        # Expert sub-networks
        self.experts = nn.ModuleList([
            ShazeerExpert(
                d_model=config.d_model,
                d_hidden=config.d_hidden,
                activation=config.activation,
                dropout=config.dropout
            )
            for _ in range(config.num_experts)
        ])

    def forward(self, x: torch.Tensor, train_mode: Optional[bool] = None) -> Tuple[torch.Tensor, Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]]:
        """
        Forward pass for the Sparsely-Gated MoE layer.
        Args:
            x: Input tensor [batch_size, seq_len, d_model].
        Returns:
            out: Output tensor [batch_size, seq_len, d_model].
            metadata: Dictionary containing gate probabilities, indices, and auxiliary losses.
        """
        orig_shape = x.shape
        x_flat = x.reshape(-1, self.d_model)  # [N, d_model]

        # Compute routing gates and load balancing losses
        gates, topk_indices, topk_gates, losses = self.gating(x, train_mode=train_mode)
        gates_flat = gates.reshape(-1, self.num_experts)  # [N, num_experts]

        # Vectorized expert computation:
        # Accumulate output: y = sum_{i=1}^n G(x)_i * E_i(x)
        out_flat = torch.zeros_like(x_flat)

        for i, expert in enumerate(self.experts):
            expert_weights = gates_flat[:, i]  # [N]
            token_mask = expert_weights > 0.0

            if token_mask.any():
                selected_x = x_flat[token_mask]
                expert_out = expert(selected_x)
                out_flat[token_mask] += expert_weights[token_mask].unsqueeze(-1) * expert_out

        out = out_flat.reshape(orig_shape)

        metadata = {
            "gates": gates,
            "topk_indices": topk_indices,
            "topk_gates": topk_gates,
            "losses": losses
        }

        return out, metadata


class ShazeerHierarchicalMoELayer(nn.Module):
    """
    Two-Level Hierarchical Mixture of Experts (Appendix B).
    Decomposes total_experts = num_groups * experts_per_group.
    Primary gating chooses top-k1 groups, secondary gating chooses top-k2 experts within group.
    """
    def __init__(self, d_model: int, d_hidden: int, num_groups: int = 4, experts_per_group: int = 4,
                 k1: int = 1, k2: int = 2, w_importance: float = 0.1, w_load: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.num_groups = num_groups
        self.experts_per_group = experts_per_group
        self.total_experts = num_groups * experts_per_group
        self.k1 = min(k1, num_groups)
        self.k2 = min(k2, experts_per_group)

        # Primary gating across groups
        self.primary_gating = NoisyTopKGating(
            d_model=d_model,
            num_experts=num_groups,
            k=self.k1,
            w_importance=w_importance,
            w_load=w_load
        )

        # Secondary sub-MoE for each group
        self.sub_moes = nn.ModuleList([
            ShazeerSparselyGatedMoELayer(ShazeerMoEConfig(
                d_model=d_model,
                d_hidden=d_hidden,
                num_experts=experts_per_group,
                k=self.k2,
                w_importance=w_importance,
                w_load=w_load
            ))
            for _ in range(num_groups)
        ])

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        orig_shape = x.shape
        x_flat = x.reshape(-1, self.d_model)
        
        # Primary gating
        p_gates, p_indices, p_top_gates, p_losses = self.primary_gating(x)
        p_gates_flat = p_gates.reshape(-1, self.num_groups)

        out_flat = torch.zeros_like(x_flat)
        total_aux_loss = p_losses["loss_aux"]

        for g, sub_moe in enumerate(self.sub_moes):
            g_weights = p_gates_flat[:, g]
            token_mask = g_weights > 0.0

            if token_mask.any():
                sub_x = x_flat[token_mask].unsqueeze(1)  # [M, 1, d]
                sub_out, sub_meta = sub_moe(sub_x)
                sub_out_flat = sub_out.squeeze(1)
                out_flat[token_mask] += g_weights[token_mask].unsqueeze(-1) * sub_out_flat
                total_aux_loss = total_aux_loss + sub_meta["losses"]["loss_aux"]

        out = out_flat.reshape(orig_shape)
        return out, {"primary_gates": p_gates, "total_aux_loss": total_aux_loss}


def verify_shazeer_moe_invariants() -> bool:
    """
    Self-contained verification suite checking all 6 mathematical invariants
    of the Shazeer Sparsely-Gated Mixture-of-Experts architecture.
    """
    print("[Shazeer-MoE] Verifying Mathematical & Architectural Invariants...")
    torch.manual_seed(42)

    config = ShazeerMoEConfig(
        d_model=64,
        d_hidden=256,
        num_experts=8,
        k=2,
        w_importance=0.1,
        w_load=0.1,
        noisy_gating=True
    )

    layer = ShazeerSparselyGatedMoELayer(config)
    B, S, D = 4, 16, config.d_model
    x = torch.randn(B, S, D, requires_grad=True)

    # Invariant 1: Zero parameter initialization
    layer.gating.reset_parameters()
    assert (layer.gating.w_gate == 0).all(), "W_g must initialize to zeros (Appendix A)"
    assert (layer.gating.w_noise == 0).all(), "W_noise must initialize to zeros (Appendix A)"
    print("  [1] Zero-initialization invariant verified -> PASS")

    # Re-initialize with small weights for gradient flow testing
    nn.init.normal_(layer.gating.w_gate, std=0.05)
    nn.init.normal_(layer.gating.w_noise, std=0.05)

    # Invariant 2: Noisy Top-K Sparsity & Gate Sum Invariant
    out, meta = layer(x, train_mode=True)
    gates = meta["gates"]  # [B, S, num_experts]
    
    active_per_token = (gates > 0).sum(dim=-1)
    assert (active_per_token == config.k).all(), f"Each token must activate exactly k={config.k} experts"

    gate_sums = gates.sum(dim=-1)
    assert torch.allclose(gate_sums, torch.ones_like(gate_sums), atol=1e-5), "Gate weights must sum to 1.0"
    print(f"  [2] Sparsity (exactly k={config.k}) & Partition of Unity (sum=1.0) verified -> PASS")

    # Invariant 3: Smooth Normal CDF Load Estimator Probability Bounds
    losses = meta["losses"]
    assert losses["loss_importance"] >= 0.0, "Importance loss must be non-negative"
    assert losses["loss_load"] >= 0.0, "Load loss must be non-negative"
    assert losses["loss_aux"] > 0.0, "Total aux loss must be strictly positive"
    print(f"  [3] Smooth Normal CDF Load Loss & Importance Loss verified (L_aux={losses['loss_aux'].item():.4f}) -> PASS")

    # Invariant 4: End-to-End Backward Gradient Flow
    total_loss = out.sum() + losses["loss_aux"]
    total_loss.backward()
    assert x.grad is not None and not torch.isnan(x.grad).any(), "Input gradient must exist and be finite"
    assert layer.gating.w_gate.grad is not None, "Gating logits W_g must receive gradients"
    assert layer.gating.w_noise.grad is not None, "Noise parameter W_noise must receive gradients"
    print("  [4] Full backpropagation & gradient flow through gating network verified -> PASS")

    # Invariant 5: Deterministic Evaluation Mode
    layer.eval()
    with torch.no_grad():
        eval_x = torch.randn(2, 8, D)
        out1, _ = layer(eval_x, train_mode=False)
        out2, _ = layer(eval_x, train_mode=False)
        assert torch.allclose(out1, out2, atol=1e-6), "Evaluation mode must be completely deterministic (zero noise)"
    print("  [5] Deterministic evaluation (zero noise) invariant verified -> PASS")

    # Invariant 6: Hierarchical MoE (Appendix B) Multi-Tier Composition
    hierarchical_moe = ShazeerHierarchicalMoELayer(
        d_model=64,
        d_hidden=128,
        num_groups=4,
        experts_per_group=4,
        k1=1,
        k2=2
    )
    h_out, h_meta = hierarchical_moe(eval_x)
    assert h_out.shape == eval_x.shape, "Hierarchical MoE output shape must match input shape"
    assert "total_aux_loss" in h_meta and h_meta["total_aux_loss"] > 0.0, "Hierarchical MoE must aggregate auxiliary losses"
    print("  [6] Hierarchical MoE (Appendix B) 4x4 grouping invariant verified -> PASS")

    print("[Shazeer-MoE] All 6 Mathematical & Architectural Invariants 100% Verified Successfully!\n")
    return True


if __name__ == "__main__":
    verify_shazeer_moe_invariants()
