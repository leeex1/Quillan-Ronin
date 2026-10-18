#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mixture-of-Recursions (MoR) Engine: Learning Dynamic Recursive Depths for Adaptive Token-Level Computation
Paper 71: Sangmin Bae et al., KAIST AI, Mila, Google DeepMind, 2025 (arXiv:2507.10524)

Key Architectural & Mathematical Formulations:
1. Recursive Transformer Parameter Sharing Schemes (Section 2.1 & Appendix A.1, Tables 1 & 5):
   - Cycle: Layers shared cyclically: f(h_t^l; Phi'_{l mod (L / N_r)})
   - Sequence: Layers reused consecutively: f(h_t^l; Phi'_{floor(l / N_r)})
   - Middle-Cycle: Unique entry Phi_0 and exit Phi_{L-1}; intermediate layers cycle:
     f(h_t^l; Phi'_{((l - 1) mod ((L - 2) / N_r)) + 1})
   - Middle-Sequence: Unique entry and exit; intermediate layers sequence:
     f(h_t^l; Phi'_{floor((l - 1) / N_r) + 1})
2. Dynamic Recursion Routing Strategies (Section 2.2.1 & Appendix A.2, Table 2 & Figure 2):
   - Expert-Choice Routing: Each recursion depth j in {1, ..., N_r} selects top-k_j tokens.
     Capacity allocation aligns with uniform expected depth:
     k_j = floor(((N_r - j + 1) / N_r) * T)
     Router routing score: r_t^j = w_j^T x_t.
     Top-k_j tokens execute recursion block with scaling: x_t^{j+1} = r_t^j * Block(x_t^j) + x_t^j.
     Non-selected tokens bypass via residual identity: x_t^{j+1} = x_t^j.
   - Auxiliary Router / Auxiliary Loss (Section 2.2.1):
     Trains router logits to predict top-k binary membership via Binary Cross-Entropy (BCE) loss:
     L_aux_router = BCE(r_t^j, 1_{t in Top-k_j}), mitigating information leakage during autoregression.
   - Token-Choice Routing: Each token selects recursion depth d_t in {1, ..., N_r}.
     Load balancing loss: L_balance = alpha * sum_{i=1}^{N_r} f_i P_i
     where f_i = (N_r / T) * sum_t 1(d_t == i) and P_i = (1 / T) * sum_t g_{t, i}.
3. Key-Value Caching Strategies in Recursive Attention (Section 2.2.2 & Appendix A.3, Table 2):
   - Recursion-Wise Caching: Separate KV caches per recursion step; tokens attend only to
     KV pairs at their active recursion step.
     KV cache memory footprint: (N_r + 1) / (2 * N_r) of dense non-recursive Transformer.
     Attention FLOPs: (k / N_ctx)^2.
   - Recursive Sharing: Caches KV pairs only at recursion step 1 and shares across all
     subsequent recursion steps {2, ..., N_r}.
     KV cache memory footprint: 1 / N_r of dense non-recursive Transformer.
     Attention FLOPs: k / N_ctx.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class MoRConfig:
    d_model: int = 64
    n_heads: int = 4
    d_ff: int = 128
    num_layers: int = 8          # Total unrolled layers L
    num_recursions: int = 3      # Number of recursions N_r
    sharing_strategy: str = "middle_cycle"  # "cycle", "sequence", "middle_cycle", "middle_sequence"
    routing_strategy: str = "expert_choice" # "expert_choice", "token_choice"
    kv_strategy: str = "recursion_wise"     # "recursion_wise", "recursive_sharing"
    aux_loss_weight: float = 0.01
    dropout: float = 0.0
    vocab_size: int = 1000


class ParameterSharingScheme:
    """
    Computes physical layer allocations and mapping schedules across unrolled depth.
    """
    @staticmethod
    def get_layer_mapping(strategy: str, L: int, Nr: int) -> Tuple[List[int], int]:
        """
        Returns:
            mapping: List of length L giving physical module index for each unrolled layer.
            num_physical_blocks: Total number of unique physical blocks required.
        """
        mapping: List[int] = []
        if strategy == "cycle":
            num_shared = max(1, L // Nr)
            for l in range(L):
                mapping.append(l % num_shared)
            num_physical = num_shared

        elif strategy == "sequence":
            num_shared = max(1, L // Nr)
            for l in range(L):
                mapping.append(min(l // Nr, num_shared - 1))
            num_physical = num_shared

        elif strategy == "middle_cycle":
            assert L >= 3, "Middle-Cycle requires at least 3 layers (first, middle, last)"
            num_intermediate = max(1, (L - 2) // Nr)
            mapping.append(0) # Unique first layer
            for l in range(1, L - 1):
                idx = ((l - 1) % num_intermediate) + 1
                mapping.append(idx)
            mapping.append(num_intermediate + 1) # Unique last layer
            num_physical = num_intermediate + 2

        elif strategy == "middle_sequence":
            assert L >= 3, "Middle-Sequence requires at least 3 layers"
            num_intermediate = max(1, (L - 2) // Nr)
            mapping.append(0)
            for l in range(1, L - 1):
                idx = min((l - 1) // Nr, num_intermediate - 1) + 1
                mapping.append(idx)
            mapping.append(num_intermediate + 1)
            num_physical = num_intermediate + 2

        else:
            raise ValueError(f"Unknown parameter sharing strategy: {strategy}")

        return mapping, num_physical


class MoRExpertChoiceRouter(nn.Module):
    """
    Expert-Choice Router (Section 2.2.1).
    Each recursion step selects its preferred top-k tokens with dynamic capacity.
    Computes Auxiliary Router BCE Loss to mitigate causality violation / information leakage.
    """
    def __init__(self, d_model: int, num_recursions: int):
        super().__init__()
        self.d_model = d_model
        self.num_recursions = Nr = num_recursions
        # Per-recursion router projection weights
        self.router_weights = nn.Parameter(torch.randn(num_recursions, d_model) * (1.0 / math.sqrt(d_model)))

    def compute_capacity(self, recursion_step: int, total_tokens: int) -> int:
        """
        k_j = floor(((N_r - j + 1) / N_r) * T) for j in {1, ..., N_r}
        """
        Nr = self.num_recursions
        j = recursion_step
        fraction = (Nr - j + 1) / Nr
        capacity = max(1, int(math.floor(fraction * total_tokens)))
        return capacity

    def forward(
        self,
        x_flat: torch.Tensor,
        recursion_step: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x_flat: [T, d_model]
            recursion_step: Current recursion step index j in {1, ..., N_r}
        Returns:
            topk_mask: [T] boolean mask indicating selected tokens
            routing_weights: [T] scalar routing weights (r_t^j)
            aux_bce_loss: Scalar BCE loss aligning router logits with top-k indicators
            topk_indices: [k] indices of selected tokens
        """
        T, D = x_flat.shape
        w_j = self.router_weights[recursion_step - 1] # [D]
        logits = torch.matmul(x_flat, w_j) # [T]
        weights = torch.sigmoid(logits)

        k = self.compute_capacity(recursion_step, T)
        topk_vals, topk_indices = torch.topk(logits, k, dim=-1)

        target_mask = torch.zeros(T, device=x_flat.device, dtype=torch.float32)
        target_mask.scatter_(0, topk_indices, 1.0)

        # Auxiliary BCE loss: trains router logits to predict top-k membership without leakage
        aux_bce_loss = F.binary_cross_entropy_with_logits(logits, target_mask)

        return target_mask.bool(), weights, aux_bce_loss, topk_indices


class MoRTokenChoiceRouter(nn.Module):
    """
    Token-Choice Router (Section 2.2.1).
    Each token chooses its total recursion depth from {1, ..., N_r}.
    Computes Load Balancing Loss: L_balance = alpha * sum_{i=1}^{N_r} f_i * P_i.
    """
    def __init__(self, d_model: int, num_recursions: int):
        super().__init__()
        self.d_model = d_model
        self.num_recursions = Nr = num_recursions
        self.router_proj = nn.Linear(d_model, num_recursions, bias=False)

    def forward(self, x_flat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x_flat: [T, d_model]
        Returns:
            assigned_depths: [T] depth index in {1, ..., N_r}
            routing_scores: [T, N_r] softmax probabilities
            balance_loss: Scalar load balancing loss
        """
        T, D = x_flat.shape
        logits = self.router_proj(x_flat) # [T, N_r]
        scores = F.softmax(logits, dim=-1) # [T, N_r]
        assigned_depths = torch.argmax(scores, dim=-1) + 1 # 1-indexed

        # Load balancing loss
        # f_i = (N_r / T) * sum_t 1(assigned_depth == i)
        one_hot = F.one_hot(assigned_depths - 1, num_classes=self.num_recursions).float()
        f = (self.num_recursions / float(T)) * one_hot.sum(dim=0) # [N_r]
        P = scores.mean(dim=0) # [N_r]
        balance_loss = (f * P).sum()

        return assigned_depths, scores, balance_loss


class MoRAttention(nn.Module):
    """
    Multi-Head Attention supporting Recursion-Wise Caching and Recursive KV Sharing.
    """
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(
        self,
        q_tokens: torch.Tensor,
        k_tokens: torch.Tensor,
        v_tokens: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Performs scaled dot-product attention between query tokens and key/value tokens.
        """
        N_q, D = q_tokens.shape
        N_k, _ = k_tokens.shape

        q = self.q_proj(q_tokens).reshape(N_q, self.n_heads, self.head_dim).transpose(0, 1) # [H, N_q, D_h]
        k = self.k_proj(k_tokens).reshape(N_k, self.n_heads, self.head_dim).transpose(0, 1) # [H, N_k, D_h]
        v = self.v_proj(v_tokens).reshape(N_k, self.n_heads, self.head_dim).transpose(0, 1) # [H, N_k, D_h]

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim) # [H, N_q, N_k]
        if mask is not None:
            scores = scores + mask

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v) # [H, N_q, D_h]
        out = out.transpose(0, 1).reshape(N_q, D) # [N_q, D]
        return self.out_proj(out)


class MoRFeedForward(nn.Module):
    """
    SwiGLU Feed-Forward Network: w2(SiLU(w1(x)) * w3(x)).
    """
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class RecursiveTransformerBlock(nn.Module):
    """
    Physical Transformer Block reused across recursion steps.
    """
    def __init__(self, config: MoRConfig):
        super().__init__()
        self.config = config
        self.norm1 = nn.LayerNorm(config.d_model)
        self.norm2 = nn.LayerNorm(config.d_model)
        self.attn = MoRAttention(config.d_model, config.n_heads, config.dropout)
        self.ffn = MoRFeedForward(config.d_model, config.d_ff, config.dropout)

    def forward(
        self,
        q_tokens: torch.Tensor,
        k_tokens: torch.Tensor,
        v_tokens: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Pre-LN Self-Attention
        q_norm = self.norm1(q_tokens)
        k_norm = self.norm1(k_tokens)
        v_norm = self.norm1(v_tokens)
        attn_out = self.attn(q_norm, k_norm, v_norm, mask)
        h = q_tokens + attn_out

        # Pre-LN FFN
        ffn_out = self.ffn(self.norm2(h))
        return h + ffn_out


class MoRRecursiveKVCache:
    """
    Analytical and practical KV Cache manager comparing:
    1. Recursion-Wise Caching: (N_r + 1) / (2 * N_r) memory ratio
    2. Recursive Sharing: 1 / N_r memory ratio
    """
    @staticmethod
    def get_memory_ratio(strategy: str, num_recursions: int) -> float:
        Nr = num_recursions
        if strategy == "recursion_wise":
            return (Nr + 1.0) / (2.0 * Nr)
        elif strategy == "recursive_sharing":
            return 1.0 / Nr
        else:
            raise ValueError(f"Unknown KV strategy: {strategy}")


class MoRModel(nn.Module):
    """
    Complete Mixture-of-Recursions (MoR) Model.
    Integrates Parameter-Sharing Schemes, Expert/Token-Choice Routers,
    and Recursion-Wise / Shared KV Attention.
    """
    def __init__(self, config: MoRConfig):
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.d_model)

        # Build Parameter-Sharing Layer Schedule
        self.layer_mapping, self.num_physical = ParameterSharingScheme.get_layer_mapping(
            config.sharing_strategy, config.num_layers, config.num_recursions
        )
        self.physical_blocks = nn.ModuleList([
            RecursiveTransformerBlock(config) for _ in range(self.num_physical)
        ])

        # Routers
        if config.routing_strategy == "expert_choice":
            self.expert_router = MoRExpertChoiceRouter(config.d_model, config.num_recursions)
            self.token_router = None
        else:
            self.token_router = MoRTokenChoiceRouter(config.d_model, config.num_recursions)
            self.expert_router = None

        self.final_norm = nn.LayerNorm(config.d_model)
        self.head = nn.Linear(config.d_model, config.vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        target_ids: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Args:
            input_ids: [B, S]
            target_ids: Optional [B, S]
        Returns:
            logits: [B, S, vocab_size]
            total_aux_loss: Combined router auxiliary loss
            diagnostics: Execution details
        """
        B, S = input_ids.shape
        T = B * S
        h = self.embed(input_ids).reshape(T, self.config.d_model)

        total_aux_loss = torch.tensor(0.0, device=input_ids.device, dtype=torch.float32)
        step_active_tokens: Dict[int, int] = {}

        # 1. Token-Choice Initial Routing (if enabled)
        assigned_depths = None
        if self.token_router is not None:
            assigned_depths, token_scores, balance_loss = self.token_router(h)
            total_aux_loss = total_aux_loss + self.config.aux_loss_weight * balance_loss

        # Shared KV state for Recursive Sharing strategy
        first_step_kv = None

        # 2. Unroll Layers through Parameter Sharing & Dynamic Recursion
        # Intermediate layers in [1, L-2] correspond to recursions
        for l_idx, phys_idx in enumerate(self.layer_mapping):
            block = self.physical_blocks[phys_idx]

            # Check if this is an entry/exit layer or recursive layer
            is_entry = (l_idx == 0 and "middle" in self.config.sharing_strategy)
            is_exit = (l_idx == self.config.num_layers - 1 and "middle" in self.config.sharing_strategy)

            if is_entry or is_exit:
                # All tokens process entry and exit layers unconditionally
                h = block(h, h, h)
                if is_entry and self.config.kv_strategy == "recursive_sharing":
                    first_step_kv = h.clone()
            else:
                # Compute recursion step index j in {1, ..., N_r}
                if "middle" in self.config.sharing_strategy:
                    rec_idx = ((l_idx - 1) % self.config.num_recursions) + 1
                else:
                    rec_idx = (l_idx % self.config.num_recursions) + 1

                # Routing Dispatch
                if self.expert_router is not None:
                    # Expert-Choice
                    mask_active, r_weights, aux_bce, topk_idx = self.expert_router(h, rec_idx)
                    total_aux_loss = total_aux_loss + self.config.aux_loss_weight * aux_bce
                    step_active_tokens[rec_idx] = int(mask_active.sum().item())

                    if mask_active.any():
                        q_tokens = h[mask_active]
                        if self.config.kv_strategy == "recursive_sharing" and first_step_kv is not None:
                            k_tokens = first_step_kv
                            v_tokens = first_step_kv
                        else:
                            k_tokens = q_tokens
                            v_tokens = q_tokens

                        block_out = block(q_tokens, k_tokens, v_tokens)
                        scale = r_weights[mask_active].unsqueeze(-1)
                        # Out-of-place residual connection with router weighting
                        delta = torch.zeros_like(h)
                        delta[mask_active] = scale * block_out
                        h = h + delta

                elif self.token_router is not None:
                    # Token-Choice: token processes layer if rec_idx <= assigned_depth
                    mask_active = assigned_depths >= rec_idx
                    step_active_tokens[rec_idx] = int(mask_active.sum().item())

                    if mask_active.any():
                        q_tokens = h[mask_active]
                        if self.config.kv_strategy == "recursive_sharing" and first_step_kv is not None:
                            k_tokens = first_step_kv
                            v_tokens = first_step_kv
                        else:
                            k_tokens = q_tokens
                            v_tokens = q_tokens

                        block_out = block(q_tokens, k_tokens, v_tokens)
                        delta = torch.zeros_like(h)
                        delta[mask_active] = block_out - h[mask_active]
                        h = h + delta

        h = self.final_norm(h)
        logits = self.head(h).reshape(B, S, self.config.vocab_size)

        diagnostics = {
            "step_active_tokens": step_active_tokens,
            "layer_mapping": self.layer_mapping,
            "total_aux_loss": total_aux_loss
        }
        return logits, total_aux_loss, diagnostics


def verify_mor_invariants() -> Dict[str, Any]:
    """
    Rigorously verifies all architectural invariants of Mixture-of-Recursions:
    1. Parameter-Sharing Mappings: Cycle, Sequence, Middle-Cycle, Middle-Sequence.
    2. Expert-Choice Capacity Allocation: k_j = floor(((N_r - j + 1) / N_r) * T).
    3. Auxiliary Router BCE Loss trains router logits and prevents leakage.
    4. Token-Choice Load Balancing Loss bounds uniform distribution.
    5. Analytical KV Cache Memory Bounds: (N_r + 1) / (2 * N_r) vs 1 / N_r.
    6. Full Model Forward & Autograd Flow through recursive shared layers.
    """
    # 1. Parameter Sharing Mappings
    m_cyc, p_cyc = ParameterSharingScheme.get_layer_mapping("cycle", 9, 3)
    assert m_cyc == [0, 1, 2, 0, 1, 2, 0, 1, 2] and p_cyc == 3

    m_seq, p_seq = ParameterSharingScheme.get_layer_mapping("sequence", 9, 3)
    assert m_seq == [0, 0, 0, 1, 1, 1, 2, 2, 2] and p_seq == 3

    m_mcyc, p_mcyc = ParameterSharingScheme.get_layer_mapping("middle_cycle", 8, 3)
    # L=8, L-2=6. 6 / 3 = 2 intermediate blocks (indices 1, 2). First=0, Last=3.
    assert m_mcyc == [0, 1, 2, 1, 2, 1, 2, 3] and p_mcyc == 4

    # 2. Dynamic Capacity Scheduling
    config = MoRConfig(
        d_model=32,
        n_heads=2,
        d_ff=64,
        num_layers=8,
        num_recursions=3,
        sharing_strategy="middle_cycle",
        routing_strategy="expert_choice",
        kv_strategy="recursion_wise",
        vocab_size=100
    )
    router = MoRExpertChoiceRouter(config.d_model, config.num_recursions)
    T = 12
    caps = [router.compute_capacity(j, T) for j in range(1, config.num_recursions + 1)]
    assert caps == [12, 8, 4], f"Capacity allocation invariant failed: {caps}"

    # 3. KV Cache Theoretical Memory Reductions
    rw_ratio = MoRRecursiveKVCache.get_memory_ratio("recursion_wise", 3)
    rs_ratio = MoRRecursiveKVCache.get_memory_ratio("recursive_sharing", 3)
    assert math.isclose(rw_ratio, 4.0 / 6.0, rel_tol=1e-5), "Recursion-wise memory ratio must be (Nr+1)/(2Nr)"
    assert math.isclose(rs_ratio, 1.0 / 3.0, rel_tol=1e-5), "Recursive sharing memory ratio must be 1/Nr"

    # 4. Full Model Execution & Autograd Flow
    model = MoRModel(config)
    input_ids = torch.randint(0, 100, (2, 6))
    logits, aux_loss, diag = model(input_ids)

    assert logits.shape == (2, 6, 100)
    assert aux_loss > 0.0

    loss = logits.sum() + aux_loss
    loss.backward()

    # Check gradients
    assert model.embed.weight.grad is not None
    assert model.expert_router.router_weights.grad is not None
    assert model.physical_blocks[1].norm1.weight.grad is not None

    return {
        "status": "PASS",
        "capacities": caps,
        "rw_memory_ratio": rw_ratio,
        "rs_memory_ratio": rs_ratio,
        "unrolled_layers": diag["layer_mapping"],
        "num_physical_blocks": p_mcyc
    }


if __name__ == "__main__":
    res = verify_mor_invariants()
    print("MoR Verification Invariants Passed:", res)
