"""
Switch Transformer: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity
==========================================================================================
Mathematical & Computational Invariant Implementation based on:
Fedus, Zoph, and Shazeer (Google Brain, 2021/2022; JMLR 23 (2022) 1-40 / arXiv:2101.03961)

Key Invariants & Formulations Implemented:
1. Top-1 Sparse Routing (Section 2.1, Eqs. 1-2):
   - Computes router logits h(x) = x @ W_r and normalized softmax probabilities p_i(x).
   - Dispatches each token to exactly one expert: i* = argmax_i p_i(x), gate = p_{i*}(x).
   - Output computation: y = gate * E_{i*}(x) (halving routing computation & expert capacity over Top-2).
2. Expert Capacity Factor & Dropping Bypass (Section 2.2, Eq. 3):
   - Expert Capacity C = floor((T / N) * capacity_factor).
   - Capacity overflow tokens are skipped from expert computation and passed directly through residual bypass.
3. Differentiable Load Balancing Auxiliary Loss (Section 2.2, Eqs. 4-6, Appendix F Fig. 14):
   - f_i = (1 / T) * sum_{x in B} 1_{argmax p(x) == i} (fraction of tokens dispatched to expert i).
   - P_i = (1 / T) * sum_{x in B} p_i(x) (fraction of probability mass assigned to expert i).
   - L_balance = alpha * N * sum_{i=1}^N f_i * P_i (minimized under uniform routing, alpha default = 0.01).
4. Selective Precision Training (Section 2.4, Table 2):
   - Casts router operations (logits, noise, softmax, top-1) locally to float32 for numerical stability.
   - Recasts dispatched tokens and combine weights back to input precision (bfloat16/float32).
5. Reduced Initialization Scale (Section 2.4, Table 3):
   - Weight initialization using truncated normal with scale parameter s = 0.1:
     sigma = sqrt(s / n_in), resampling values outside [-2*sigma, 2*sigma].
6. Expert Dropout Regularization (Section 2.4, Table 4):
   - Independent dropout rates for standard layers (d=0.1) versus interim expert FFN layers (ed=0.4).
"""

import math
from typing import Dict, List, Optional, Tuple, NamedTuple
import torch
import torch.nn as nn
import torch.nn.functional as F


def truncated_normal_init_(
    tensor: torch.Tensor,
    mean: float = 0.0,
    scale: float = 0.1,
    fan_in: Optional[int] = None
) -> torch.Tensor:
    """
    Initializes tensor from a truncated normal distribution with reduced scale s=0.1.
    Values more than 2 standard deviations from the mean are resampled (Section 2.4).
    """
    if fan_in is None:
        fan_in = tensor.shape[0] if tensor.ndim >= 1 else 1
    std = math.sqrt(scale / max(1, fan_in))
    with torch.no_grad():
        nn.init.trunc_normal_(tensor, mean=mean, std=std, a=-2.0 * std, b=2.0 * std)
    return tensor


class SwitchRoutingOutput(NamedTuple):
    """Container for outputs produced by the Switch Transformer router."""
    dispatched_indices: torch.Tensor       # Shape: [T], int64 expert assigned to each token
    gate_values: torch.Tensor              # Shape: [T], float32/input_dtype routing probability
    expert_mask: torch.Tensor              # Shape: [T, num_experts], one-hot assignment mask
    dropped_mask: torch.Tensor             # Shape: [T], bool mask (True if token exceeded expert capacity)
    aux_loss: torch.Tensor                 # Scalar load balancing auxiliary loss
    router_probs: torch.Tensor             # Shape: [T, num_experts], full softmax routing distribution
    tokens_per_expert: torch.Tensor        # Shape: [num_experts], count of tokens assigned to each expert


class SwitchRouter(nn.Module):
    """
    Switch Transformer Router Module (Section 2.1 & 2.4, Appendix F Fig. 15).
    Features:
    - Selective precision (float32 router core).
    - Optional uniform exploration noise during training.
    - Exact Top-1 routing.
    - Capacity-capped token dispatch with overflow tracking.
    - Differentiable load balancing auxiliary loss.
    """

    def __init__(
        self,
        d_model: int,
        num_experts: int,
        capacity_factor: float = 1.0,
        aux_loss_coeff: float = 0.01,
        noise_eps: float = 0.0,
        selective_precision: bool = True,
        init_scale: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.capacity_factor = capacity_factor
        self.aux_loss_coeff = aux_loss_coeff
        self.noise_eps = noise_eps
        self.selective_precision = selective_precision

        # Router projection matrix: [d_model, num_experts]
        self.weight = nn.Parameter(torch.empty(d_model, num_experts))
        truncated_normal_init_(self.weight, mean=0.0, scale=init_scale, fan_in=d_model)

    def forward(
        self,
        inputs: torch.Tensor,
        custom_capacity_factor: Optional[float] = None
    ) -> SwitchRoutingOutput:
        """
        Forward routing computation for flat or batched tokens.
        Args:
            inputs: Tensor of shape [..., d_model], typically [T, d_model] or [B, S, d_model].
            custom_capacity_factor: Optional runtime override for expert capacity factor.
        """
        orig_shape = inputs.shape
        flat_inputs = inputs.reshape(-1, self.d_model)
        num_tokens = flat_inputs.size(0)
        orig_dtype = inputs.dtype

        # Selective precision: cast router core computation to float32 (Section 2.4)
        if self.selective_precision:
            router_inputs = flat_inputs.to(torch.float32)
            router_weight = self.weight.to(torch.float32)
        else:
            router_inputs = flat_inputs
            router_weight = self.weight

        # Linear projection: [num_tokens, num_experts]
        logits = torch.matmul(router_inputs, router_weight)

        # Add uniform exploration noise during training if enabled (Appendix F Fig. 15)
        if self.training and self.noise_eps > 0.0:
            noise = torch.empty_like(logits).uniform_(1.0 - self.noise_eps, 1.0 + self.noise_eps)
            logits = logits + noise

        # Softmax probabilities over experts: [num_tokens, num_experts]
        router_probs = F.softmax(logits, dim=-1)

        # Top-1 selection: [num_tokens]
        top1_gates, top1_indices = torch.max(router_probs, dim=-1)

        # One-hot dispatch mask: [num_tokens, num_experts]
        expert_mask = F.one_hot(top1_indices, num_classes=self.num_experts).float()

        # Load balancing auxiliary loss (Section 2.2, Eqs. 4-6)
        # f_i: fraction of tokens dispatched to expert i
        f_i = expert_mask.mean(dim=0)  # [num_experts]
        # P_i: fraction of probability mass allocated to expert i
        p_i = router_probs.mean(dim=0)  # [num_experts]
        # L_balance = alpha * N * sum_{i=1}^N f_i * P_i
        balance_loss = self.aux_loss_coeff * float(self.num_experts) * torch.sum(f_i * p_i)

        # Calculate expert capacity (Section 2.2, Eq. 3)
        cf = custom_capacity_factor if custom_capacity_factor is not None else self.capacity_factor
        expert_capacity = max(1, math.floor((num_tokens / self.num_experts) * cf))

        # Track token positions per expert to identify overflow
        # Cumulative sum along tokens for each expert
        cumsum = torch.cumsum(expert_mask, dim=0)  # [num_tokens, num_experts]
        token_positions = cumsum * expert_mask      # 1-indexed position in expert queue

        # An overflow occurs if token position > expert_capacity
        within_capacity = token_positions <= expert_capacity  # [num_tokens, num_experts]
        # Valid assignment requires being assigned and within capacity
        valid_mask = (expert_mask * within_capacity.float()).sum(dim=-1) > 0.5  # [num_tokens]
        dropped_mask = ~valid_mask  # [num_tokens]

        # Mask gate values: dropped tokens have 0 gate multiplier
        effective_gates = top1_gates * valid_mask.to(top1_gates.dtype)

        # Cast outputs back to orig_dtype
        effective_gates = effective_gates.to(orig_dtype)
        tokens_per_expert = expert_mask.sum(dim=0).to(torch.int64)

        return SwitchRoutingOutput(
            dispatched_indices=top1_indices,
            gate_values=effective_gates,
            expert_mask=expert_mask,
            dropped_mask=dropped_mask,
            aux_loss=balance_loss,
            router_probs=router_probs.to(orig_dtype),
            tokens_per_expert=tokens_per_expert,
        )


class SwitchExpert(nn.Module):
    """
    Individual Feed-Forward Expert Network with configurable expert dropout (Section 2.4).
    Architecture:
    Linear(d_model, d_ff) -> Activation (GELU/ReLU) -> Dropout(expert_dropout) -> Linear(d_ff, d_model)
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        expert_dropout: float = 0.4,
        activation_fn: str = "gelu",
        init_scale: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(expert_dropout)

        if activation_fn == "gelu":
            self.act = F.gelu
        elif activation_fn == "relu":
            self.act = F.relu
        else:
            raise ValueError(f"Unsupported activation: {activation_fn}")

        # Reduced scale initialization
        truncated_normal_init_(self.w1.weight, scale=init_scale, fan_in=d_model)
        truncated_normal_init_(self.w2.weight, scale=init_scale, fan_in=d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.w1(x))
        h = self.dropout(h)
        return self.w2(h)


class SwitchFeedForward(nn.Module):
    """
    Complete Switch Transformer Sparse Feed-Forward Layer.
    Integrates:
    - SwitchRouter (Top-1 gating, selective precision, aux load loss, capacity factor).
    - N SwitchExperts.
    - Residual bypass for dropped tokens (tokens exceeding capacity skip expert computation).
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        num_experts: int,
        capacity_factor: float = 1.0,
        aux_loss_coeff: float = 0.01,
        expert_dropout: float = 0.4,
        noise_eps: float = 0.0,
        selective_precision: bool = True,
        init_scale: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.num_experts = num_experts
        self.capacity_factor = capacity_factor

        self.router = SwitchRouter(
            d_model=d_model,
            num_experts=num_experts,
            capacity_factor=capacity_factor,
            aux_loss_coeff=aux_loss_coeff,
            noise_eps=noise_eps,
            selective_precision=selective_precision,
            init_scale=init_scale,
        )

        self.experts = nn.ModuleList([
            SwitchExpert(
                d_model=d_model,
                d_ff=d_ff,
                expert_dropout=expert_dropout,
                init_scale=init_scale,
            )
            for _ in range(num_experts)
        ])

    def forward(
        self,
        inputs: torch.Tensor,
        custom_capacity_factor: Optional[float] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward computation of Switch FFN layer.
        Args:
            inputs: [batch_size, seq_len, d_model] or [T, d_model].
        Returns:
            outputs: Processed tensor with same shape as inputs.
            aux_loss: Load balancing auxiliary loss.
            stats: Dictionary of routing metrics (dropped_count, per_expert_tokens, etc.).
        """
        orig_shape = inputs.shape
        flat_x = inputs.reshape(-1, self.d_model)
        num_tokens = flat_x.size(0)

        # Compute routing decisions
        routing = self.router(flat_x, custom_capacity_factor=custom_capacity_factor)

        # Output buffer initialized to zero
        # Dropped tokens receive 0 contribution here, which leaves residual stream pure
        output_flat = torch.zeros_like(flat_x)

        # Execute expert computation for assigned & non-dropped tokens
        # Group tokens by expert index
        for expert_id, expert in enumerate(self.experts):
            # Select tokens assigned to this expert that are NOT dropped
            expert_assigned = (routing.dispatched_indices == expert_id) & (~routing.dropped_mask)
            if expert_assigned.any():
                token_indices = expert_assigned.nonzero(as_tuple=True)[0]
                expert_in = flat_x[token_indices]
                expert_out = expert(expert_in)
                gates = routing.gate_values[token_indices].unsqueeze(-1)
                output_flat[token_indices] = gates * expert_out

        outputs = output_flat.reshape(orig_shape)

        stats = {
            "dropped_count": routing.dropped_mask.sum(),
            "dropped_ratio": routing.dropped_mask.float().mean(),
            "tokens_per_expert": routing.tokens_per_expert,
            "router_probs": routing.router_probs,
        }

        return outputs, routing.aux_loss, stats


class SwitchTransformerLayer(nn.Module):
    """
    A Complete Switch Transformer Block with Self-Attention and Switch Sparse FFN.
    Residual structure:
        x1 = x + Attention(LayerNorm(x))
        x2 = x1 + SwitchFFN(LayerNorm(x1))
    Note: For dropped tokens, SwitchFFN contribution is 0, so x2 = x1 (pure residual bypass).
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        num_experts: int,
        capacity_factor: float = 1.0,
        standard_dropout: float = 0.1,
        expert_dropout: float = 0.4,
        aux_loss_coeff: float = 0.01,
        init_scale: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads

        # Self-Attention sub-layer
        self.ln1 = nn.LayerNorm(d_model)
        self.mha = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=standard_dropout,
            batch_first=True
        )
        self.dropout1 = nn.Dropout(standard_dropout)

        # Switch Sparse FFN sub-layer
        self.ln2 = nn.LayerNorm(d_model)
        self.switch_ffn = SwitchFeedForward(
            d_model=d_model,
            d_ff=d_ff,
            num_experts=num_experts,
            capacity_factor=capacity_factor,
            aux_loss_coeff=aux_loss_coeff,
            expert_dropout=expert_dropout,
            init_scale=init_scale,
        )

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward execution of full Switch Transformer Layer.
        """
        # Multi-head attention with residual
        norm_x = self.ln1(x)
        attn_out, _ = self.mha(norm_x, norm_x, norm_x, attn_mask=attn_mask)
        x = x + self.dropout1(attn_out)

        # Switch sparse FFN with residual bypass
        norm_x2 = self.ln2(x)
        ffn_out, aux_loss, stats = self.switch_ffn(norm_x2)
        x = x + ffn_out

        return x, aux_loss, stats
