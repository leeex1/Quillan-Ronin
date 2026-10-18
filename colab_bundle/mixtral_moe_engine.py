"""
Mixtral 8x7B Sparse Mixture of Experts (SMoE) Engine
=====================================================
Formal Paper Reference:
"Mixtral of Experts" (Albert Q. Jiang et al., Mistral AI, Dec 2023 / arXiv:2401.04088)

Core Architectural & Mathematical Invariants:
1. Sparse Mixture of Experts (SMoE, Section 2.1):
   Replaces dense FFNs with 8 SwiGLU expert networks at every transformer layer.
   For each token x:
       G(x) := Softmax(TopK(x * W_g))
       y = sum_{i=0}^{n-1} G(x)_i * E_i(x)
   where K = 2 (Top-2 routing) and n = 8 experts. Only 2 experts computed per token.
2. Normalized Top-K Softmax:
   Softmax is computed over the selected top-K logits, guaranteeing that expert weights
   strictly sum to 1.0 for each token.
3. SwiGLU Expert Architecture:
   E_i(x) = (SiLU(x * W_gate^{(i)}) * (x * W_up^{(i)})) * W_down^{(i)}
4. Grouped-Query Attention (GQA):
   32 query heads mapped to 8 key/value heads (4:1 query-to-KV ratio).
5. Auxiliary Load Balancing Loss:
   L_aux = alpha * n * sum_{i=1}^n f_i * P_i
   where f_i is the fraction of tokens routed to expert i and P_i is the mean probability.
"""

import math
import logging
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, List

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger("MixtralMoE")


@dataclass
class MixtralConfig:
    """Configuration hyperparameters for Mixtral 8x7B architecture."""
    dim: int = 128                      # Hidden dimension d_model (default 4096, scaled for test)
    n_layers: int = 2                   # Number of transformer layers (default 32)
    head_dim: int = 32                  # Dimension per attention head (default 128)
    hidden_dim: int = 256               # SwiGLU intermediate dimension (default 14336)
    n_heads: int = 4                    # Query attention heads (default 32)
    n_kv_heads: int = 1                 # Key/Value attention heads (default 8, 4:1 GQA)
    num_experts: int = 8                # Number of experts per layer (default 8)
    top_k_experts: int = 2              # Active experts per token (K=2)
    context_len: int = 32768            # Max dense sequence length (32k)
    vocab_size: int = 32000             # Vocabulary size (32,000)
    rope_theta: float = 10000.0         # RoPE base frequency
    rms_norm_eps: float = 1e-5          # RMSNorm epsilon
    aux_loss_coef: float = 0.01         # Load balancing loss multiplier alpha

    @property
    def num_queries_per_kv(self) -> int:
        assert self.n_heads % self.n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"
        return self.n_heads // self.n_kv_heads


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight


def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor
) -> torch.Tensor:
    """Apply Rotary Position Embedding (RoPE) to tensor of shape (batch, heads, seq, head_dim)."""
    d = x.shape[-1]
    x1 = x[..., : d // 2]
    x2 = x[..., d // 2 :]
    rotated = torch.cat((-x2, x1), dim=-1)
    return (x * cos) + (rotated * sin)


class RoPEEmbedding(nn.Module):
    """Precomputes Rotary Positional Embedding sinusoids."""

    def __init__(self, dim: int, max_seq_len: int = 32768, theta: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.theta = theta
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def get_cos_sin(self, seq_len: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq.to(device))
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos().unsqueeze(0).unsqueeze(0)
        sin = emb.sin().unsqueeze(0).unsqueeze(0)
        return cos, sin


class MixtralAttention(nn.Module):
    """
    Grouped-Query Attention (GQA) for Mixtral 8x7B.
    Query heads: n_heads, Key/Value heads: n_kv_heads.
    """

    def __init__(self, config: MixtralConfig):
        super().__init__()
        self.config = config
        self.dim = config.dim
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.head_dim
        self.num_queries_per_kv = config.num_queries_per_kv

        self.q_proj = nn.Linear(self.dim, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, self.dim, bias=False)
        self.rope = RoPEEmbedding(self.head_dim, max_seq_len=config.context_len, theta=config.rope_theta)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rope.get_cos_sin(seq_len, x.device)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)

        if self.num_queries_per_kv > 1:
            k = k.repeat_interleave(self.num_queries_per_kv, dim=1)
            v = v.repeat_interleave(self.num_queries_per_kv, dim=1)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask
        else:
            # Standard causal lower-triangular mask
            causal_mask = torch.triu(torch.full((seq_len, seq_len), float("-inf"), device=x.device), diagonal=1)
            scores = scores + causal_mask.unsqueeze(0).unsqueeze(0)

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)

        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.n_heads * self.head_dim)
        return self.o_proj(out)


class SwiGLUExpert(nn.Module):
    """
    SwiGLU Feed-Forward Network:
    E(x) = (SiLU(x W_gate) * (x W_up)) W_down
    """

    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w_gate = nn.Linear(dim, hidden_dim, bias=False)
        self.w_up = nn.Linear(dim, hidden_dim, bias=False)
        self.w_down = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class MixtralRouter(nn.Module):
    """
    Top-K Gating Router with Normalized Softmax and Auxiliary Load Balancing Loss.
    Formula:
        logits = x * W_g
        top_k_logits, top_k_indices = topk(logits, k=2)
        weights = softmax(top_k_logits, dim=-1)
    """

    def __init__(self, dim: int, num_experts: int = 8, top_k: int = 2, aux_loss_coef: float = 0.01):
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.aux_loss_coef = aux_loss_coef
        self.gate = nn.Linear(dim, num_experts, bias=False)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, dim) or (num_tokens, dim)
        Returns:
            routing_weights: (num_tokens, top_k) normalized softmax weights summing to 1.0 per token
            selected_experts: (num_tokens, top_k) indices of chosen experts in {0, ..., num_experts - 1}
            aux_loss: Scalar auxiliary load balancing loss
        """
        x_flat = x.view(-1, self.dim)
        num_tokens = x_flat.shape[0]

        # 1. Compute raw routing logits (num_tokens, num_experts)
        logits = self.gate(x_flat)

        # 2. Select top-K experts per token
        top_k_logits, selected_experts = torch.topk(logits, self.top_k, dim=-1)

        # 3. Softmax over Top-K logits (Section 2.1)
        routing_weights = F.softmax(top_k_logits, dim=-1)

        # 4. Auxiliary Load Balancing Loss:
        # L_aux = alpha * num_experts * sum_{i=0}^{num_experts-1} f_i * P_i
        router_probs = F.softmax(logits, dim=-1)  # (num_tokens, num_experts)
        P_i = router_probs.mean(dim=0)            # (num_experts,)

        mask_selected = torch.zeros(num_tokens, self.num_experts, device=x.device)
        mask_selected.scatter_(1, selected_experts, 1.0)
        f_i = mask_selected.mean(dim=0)           # (num_experts,)

        aux_loss = self.aux_loss_coef * float(self.num_experts) * torch.sum(f_i * P_i)

        return routing_weights, selected_experts, aux_loss


class MixtralSparseMoEBlock(nn.Module):
    """
    Mixtral Sparse Mixture-of-Experts Block:
    Combines Top-2 Gating with 8 SwiGLU expert networks.
    """

    def __init__(self, config: MixtralConfig):
        super().__init__()
        self.config = config
        self.dim = config.dim
        self.num_experts = config.num_experts
        self.top_k = config.top_k_experts

        self.router = MixtralRouter(
            dim=config.dim,
            num_experts=config.num_experts,
            top_k=config.top_k_experts,
            aux_loss_coef=config.aux_loss_coef
        )
        self.experts = nn.ModuleList([
            SwiGLUExpert(config.dim, config.hidden_dim)
            for _ in range(config.num_experts)
        ])

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with vectorized token dispatch across top-k experts.
        Args:
            x: (batch_size, seq_len, dim)
        Returns:
            out: (batch_size, seq_len, dim)
            aux_loss: scalar load balancing loss
        """
        batch_size, seq_len, dim = x.shape
        x_flat = x.view(-1, dim)

        # Route tokens
        routing_weights, selected_experts, aux_loss = self.router(x_flat)

        out_flat = torch.zeros_like(x_flat)

        # Dispatch tokens to selected experts
        for expert_idx in range(self.num_experts):
            token_indices, k_rank = torch.where(selected_experts == expert_idx)
            if token_indices.numel() == 0:
                continue

            expert_tokens = x_flat[token_indices]
            expert_out = self.experts[expert_idx](expert_tokens)
            weights = routing_weights[token_indices, k_rank].unsqueeze(-1)
            out_flat.index_add_(0, token_indices, weights * expert_out)

        out = out_flat.view(batch_size, seq_len, dim)
        return out, aux_loss


class MixtralDecoderLayer(nn.Module):
    """Single Mixtral 8x7B Transformer Decoder Layer."""

    def __init__(self, config: MixtralConfig):
        super().__init__()
        self.attn_norm = RMSNorm(config.dim, eps=config.rms_norm_eps)
        self.attn = MixtralAttention(config)
        self.ffn_norm = RMSNorm(config.dim, eps=config.rms_norm_eps)
        self.moe = MixtralSparseMoEBlock(config)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # Pre-norm Attention with residual
        h = x + self.attn(self.attn_norm(x), mask=mask)
        # Pre-norm Sparse MoE with residual
        moe_out, aux_loss = self.moe(self.ffn_norm(h))
        out = h + moe_out
        return out, aux_loss


class MixtralModel(nn.Module):
    """
    Mixtral 8x7B Language Model Backbone.
    Composes embedding, stacked MixtralDecoderLayers, final RMSNorm, and LM head.
    """

    def __init__(self, config: MixtralConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.dim)
        self.layers = nn.ModuleList([
            MixtralDecoderLayer(config) for _ in range(config.n_layers)
        ])
        self.norm = RMSNorm(config.dim, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if inputs_embeds is None:
            assert input_ids is not None, "Must provide input_ids or inputs_embeds"
            h = self.embed_tokens(input_ids)
        else:
            h = inputs_embeds

        total_aux_loss = torch.tensor(0.0, device=h.device)
        for layer in self.layers:
            h, layer_aux = layer(h, mask=mask)
            total_aux_loss = total_aux_loss + layer_aux

        h = self.norm(h)
        logits = self.lm_head(h)
        return logits, total_aux_loss


def verify_mixtral_invariants() -> Dict[str, Any]:
    """
    Verifies all mathematical and architectural invariants of Mixtral 8x7B:
    1. Top-2 Gating normalization: weights strictly sum to 1.0 for each token.
    2. Sparsity invariant: exactly K=2 of 8 experts active per token.
    3. Vectorized MoE forward output shape & numerical stability.
    4. Auxiliary load balance loss calculation and non-zero gradient flow.
    5. Full model forward, causal generation, and backward backpropagation.
    """
    results: Dict[str, Any] = {}
    cfg = MixtralConfig(
        dim=64,
        n_layers=2,
        head_dim=16,
        hidden_dim=128,
        n_heads=4,
        n_kv_heads=1,
        num_experts=8,
        top_k_experts=2,
        context_len=64,
        vocab_size=120,
        aux_loss_coef=0.01
    )

    batch_size = 2
    seq_len = 8

    # 1. Verify Top-2 Gating Normalization
    router = MixtralRouter(cfg.dim, cfg.num_experts, cfg.top_k_experts, cfg.aux_loss_coef)
    x = torch.randn(batch_size * seq_len, cfg.dim)
    weights, indices, aux_loss = router(x)

    weight_sums = weights.sum(dim=-1)
    assert torch.allclose(weight_sums, torch.ones_like(weight_sums), atol=1e-5), \
        "Top-K router weights do not sum to 1.0"
    assert indices.shape == (batch_size * seq_len, 2)
    assert (indices >= 0).all() and (indices < cfg.num_experts).all()
    assert (indices[:, 0] != indices[:, 1]).all(), "Router selected duplicate experts for token"
    results["top2_normalization_and_uniqueness_verified"] = True

    # 2. Auxiliary Load Balance Loss
    assert aux_loss.item() > 0.0, "Auxiliary load balance loss must be positive"
    aux_loss.backward()
    assert router.gate.weight.grad is not None, "Router gate did not receive gradient from aux_loss"
    results["aux_loss_gradient_verified"] = True

    # 3. Sparse MoE Block Forward & Backward
    moe_block = MixtralSparseMoEBlock(cfg)
    x_in = torch.randn(batch_size, seq_len, cfg.dim, requires_grad=True)
    moe_out, moe_aux = moe_block(x_in)
    assert moe_out.shape == (batch_size, seq_len, cfg.dim)
    loss = moe_out.sum() + moe_aux
    loss.backward()
    assert x_in.grad is not None and not torch.isnan(x_in.grad).any()
    has_expert_grad = any(e.w_gate.weight.grad is not None for e in moe_block.experts)
    assert has_expert_grad, "No expert received gradients"
    results["moe_block_dispatch_and_grad_verified"] = True

    # 4. Full Mixtral Model Forward and Backward
    model = MixtralModel(cfg)
    input_ids = torch.randint(0, cfg.vocab_size, (batch_size, seq_len))
    logits, total_aux = model(input_ids)
    assert logits.shape == (batch_size, seq_len, cfg.vocab_size)
    assert total_aux.item() > 0.0

    target = torch.randint(0, cfg.vocab_size, (batch_size, seq_len))
    lm_loss = F.cross_entropy(logits.view(-1, cfg.vocab_size), target.view(-1)) + total_aux
    lm_loss.backward()

    assert model.embed_tokens.weight.grad is not None
    assert model.layers[0].attn.q_proj.weight.grad is not None
    assert model.layers[0].moe.router.gate.weight.grad is not None
    assert model.lm_head.weight.grad is not None
    results["full_model_training_verified"] = True

    results["all_invariants_passed"] = True
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Testing Mixtral 8x7B MoE Invariants...")
    res = verify_mixtral_invariants()
    print("Verification Results:", res)
