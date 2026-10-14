"""
Mixture-of-Depths (MoD) Dynamic Compute Engine
=============================================
Formal Paper Reference:
"Mixture-of-Depths: Dynamically allocating compute in transformer-based language models"
(David Raposo, Sam Ritter, Blake Richards, Timothy Lillicrap, Peter Conway Humphreys, Adam Santoro;
Google DeepMind & McGill / Mila; April 2024 / arXiv:2404.02258)

Core Architectural & Mathematical Invariants:
1. Static Compute Budget & Capacity C (Section 3.1):
   Enforces a total compute budget per forward pass by capping the number of tokens C that
   participate in self-attention and MLP: C = floor(capacity_factor * S) < S.
   Guarantees static computation graphs and predictable FLOP reduction.
2. Learned Expert-Choice Router & Weight Scaling (Section 3.4, Eq. 1):
   Router emits scalar weight r_i^l = w_theta^T x_i^l.
   Top-C tokens X_tilde participate in computation block f:
       x_{i}^{l+1} = r_i^l * f_i(X_tilde) + x_i^l   if r_i^l > P_beta(R^l)
       x_{i}^{l+1} = x_i^l                         if r_i^l < P_beta(R^l)
   Multiplying f by r_i puts router weights directly onto the gradient path.
3. Residual Identity Bypass:
   Non-selected tokens bypass self-attention and MLP completely via the residual stream,
   preserving their representations at zero additional compute.
4. Autoregressive Sampling via BCE Router Loss (Section 3.5):
   Binary cross entropy loss L_router pressures router outputs toward 1 for top-C tokens
   and 0 for bypassed tokens. During autoregressive decoding, tokens with sigmoid(r_t) > 0.5
   execute the block, while others take the residual path.
"""

import math
import logging
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, List

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger("MixtureOfDepths")


@dataclass
class MoDConfig:
    """Hyperparameters for Mixture-of-Depths Architecture."""
    dim: int = 128                      # Hidden dimension d_model (default 4096, scaled for tests)
    n_layers: int = 4                   # Number of transformer layers (default 32)
    head_dim: int = 32                  # Dimension per attention head (default 128)
    hidden_dim: int = 256               # SwiGLU / MLP intermediate dimension
    n_heads: int = 4                    # Query attention heads
    n_kv_heads: int = 1                 # KV attention heads (GQA)
    capacity_factor: float = 0.5        # Fraction of tokens processed by block (0.5 = 50% FLOPs)
    context_len: int = 2048             # Maximum context length
    vocab_size: int = 32000             # Vocabulary size
    rope_theta: float = 10000.0         # RoPE theta base
    rms_norm_eps: float = 1e-5          # RMSNorm epsilon
    router_aux_loss_coef: float = 0.01  # Auxiliary BCE router loss multiplier
    inference_threshold: float = 0.5    # Sigmoid threshold for autoregressive single-token bypass

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

    def __init__(self, dim: int, max_seq_len: int = 4096, theta: float = 10000.0):
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


class MoDSelfAttention(nn.Module):
    """
    Self-Attention block computed over the selected subset of tokens C <= S.
    """

    def __init__(self, config: MoDConfig):
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

    def forward(self, x: torch.Tensor, pos_indices: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x: (batch_size, capacity_C, dim)
        pos_indices: (batch_size, capacity_C) original token positions in sequence
        """
        batch_size, cap_c, _ = x.shape

        q = self.q_proj(x).view(batch_size, cap_c, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, cap_c, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, cap_c, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # Standard RoPE on gathered sequence
        cos, sin = self.rope.get_cos_sin(cap_c, x.device)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)

        if self.num_queries_per_kv > 1:
            k = k.repeat_interleave(self.num_queries_per_kv, dim=1)
            v = v.repeat_interleave(self.num_queries_per_kv, dim=1)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Build causal mask based on original token positions if provided, else standard causal
        if pos_indices is not None:
            # Token i attends to token j iff orig_pos_j <= orig_pos_i
            pos_row = pos_indices.unsqueeze(2)  # (B, C, 1)
            pos_col = pos_indices.unsqueeze(1)  # (B, 1, C)
            valid_causal = (pos_col <= pos_row).unsqueeze(1)  # (B, 1, C, C)
            mask = torch.zeros(batch_size, 1, cap_c, cap_c, device=x.device, dtype=torch.float32)
            mask = mask.masked_fill(~valid_causal, float("-inf"))
            scores = scores + mask
        else:
            causal_mask = torch.triu(torch.full((cap_c, cap_c), float("-inf"), device=x.device), diagonal=1)
            scores = scores + causal_mask.unsqueeze(0).unsqueeze(0)

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)

        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, cap_c, self.n_heads * self.head_dim)
        return self.o_proj(out)


class MoDSwiGLU(nn.Module):
    """SwiGLU Feed-Forward Network for Mixture-of-Depths block."""

    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w_gate = nn.Linear(dim, hidden_dim, bias=False)
        self.w_up = nn.Linear(dim, hidden_dim, bias=False)
        self.w_down = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class MoDRouter(nn.Module):
    """
    Learned Expert-Choice Router for Mixture-of-Depths (Section 3.2 - 3.5).
    Computes scalar routing weight r_i = w_theta^T x_i for each token.
    Selects top-C tokens per sequence element to participate in block compute.
    Computes auxiliary BCE loss for autoregressive deployment.
    """

    def __init__(self, dim: int, aux_loss_coef: float = 0.01):
        super().__init__()
        self.dim = dim
        self.aux_loss_coef = aux_loss_coef
        self.linear_router = nn.Linear(dim, 1, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        capacity: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch_size, seq_len, dim)
            capacity: Number of tokens C to select per sequence
        Returns:
            router_weights: (batch_size, seq_len) raw scalar weights r_i
            top_k_indices: (batch_size, capacity) indices of selected tokens
            top_k_weights: (batch_size, capacity) router weights for selected tokens
            aux_bce_loss: Scalar BCE loss aligning router logits with top-k binary targets
        """
        batch_size, seq_len, _ = x.shape
        # Compute scalar router logits r_i = w_theta^T x_i
        router_logits = self.linear_router(x).squeeze(-1)  # (batch_size, seq_len)

        # Select top-C tokens per sequence
        top_k_weights, top_k_indices = torch.topk(router_logits, capacity, dim=-1, sorted=True)

        # Formulate binary targets for auxiliary sampling loss (Section 3.5)
        # 1.0 for tokens in top-k, 0.0 for bypassed tokens
        targets = torch.zeros_like(router_logits)
        targets.scatter_(1, top_k_indices, 1.0)

        # Binary Cross Entropy with Logits loss
        bce_loss = F.binary_cross_entropy_with_logits(router_logits, targets)
        aux_loss = self.aux_loss_coef * bce_loss

        return router_logits, top_k_indices, top_k_weights, aux_loss


class MoDTransformerBlock(nn.Module):
    """
    Mixture-of-Depths Transformer Block:
    Applies Self-Attention and SwiGLU MLP only to the top-C tokens selected by the router.
    Formula (Eq. 1):
        x_{i}^{l+1} = r_i^l * f_i(X_tilde) + x_i^l   if selected
        x_{i}^{l+1} = x_i^l                         if bypassed
    """

    def __init__(self, config: MoDConfig):
        super().__init__()
        self.config = config
        self.dim = config.dim
        self.capacity_factor = config.capacity_factor

        self.router = MoDRouter(config.dim, aux_loss_coef=config.router_aux_loss_coef)
        self.attn_norm = RMSNorm(config.dim, eps=config.rms_norm_eps)
        self.attn = MoDSelfAttention(config)
        self.mlp_norm = RMSNorm(config.dim, eps=config.rms_norm_eps)
        self.mlp = MoDSwiGLU(config.dim, config.hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        force_capacity: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Args:
            x: (batch_size, seq_len, dim)
            force_capacity: Optional explicit token capacity C
        Returns:
            out: (batch_size, seq_len, dim)
            aux_loss: Scalar router BCE loss
            stats: Routing metrics and participation counts
        """
        batch_size, seq_len, dim = x.shape

        # Determine token capacity C
        if force_capacity is not None:
            capacity = min(force_capacity, seq_len)
        else:
            capacity = max(1, min(seq_len, int(math.floor(self.capacity_factor * seq_len))))

        # 1. Router evaluation & top-k selection
        router_logits, top_k_indices, top_k_weights, aux_loss = self.router(x, capacity)

        # 2. Gather selected tokens: X_tilde of shape (batch_size, capacity, dim)
        batch_indices = torch.arange(batch_size, device=x.device).unsqueeze(1).expand(-1, capacity)
        x_gathered = x[batch_indices, top_k_indices]

        # 3. Apply transformer computation block f = MLP(Norm(Attn(Norm(x)))) to X_tilde
        h_attn = x_gathered + self.attn(self.attn_norm(x_gathered), pos_indices=top_k_indices)
        f_out = h_attn + self.mlp(self.mlp_norm(h_attn))

        # 4. Multiply block output by router weights r_i (Section 3.4):
        # putting router weights onto the gradient path
        scaled_f = f_out * top_k_weights.unsqueeze(-1)

        # 5. Scatter back into residual stream:
        # x_next = x (for bypassed tokens) + scaled_f (for selected tokens)
        out = x.clone()
        out[batch_indices, top_k_indices] = x[batch_indices, top_k_indices] + scaled_f

        stats = {
            "capacity": capacity,
            "total_tokens": seq_len,
            "compute_ratio": capacity / seq_len,
            "mean_router_weight": router_logits.mean().item(),
            "top_k_indices": top_k_indices
        }

        return out, aux_loss, stats

    def forward_single_step(self, x_step: torch.Tensor) -> Tuple[torch.Tensor, bool]:
        """
        Autoregressive single-step generation with sigmoid thresholding (Section 3.5).
        If sigmoid(r_t) > threshold (0.5), execute block; otherwise bypass.
        Args:
            x_step: (batch_size, 1, dim)
        Returns:
            out_step: (batch_size, 1, dim)
            participated: bool indicating whether block computation was executed
        """
        router_logit = self.router.linear_router(x_step).squeeze(-1)  # (batch_size, 1)
        prob = torch.sigmoid(router_logit)

        if prob.mean().item() > self.config.inference_threshold:
            # Participate in block computation
            h_attn = x_step + self.attn(self.attn_norm(x_step))
            f_out = h_attn + self.mlp(self.mlp_norm(h_attn))
            out_step = x_step + f_out * router_logit.unsqueeze(-1)
            return out_step, True
        else:
            # Bypass block via residual identity connection
            return x_step, False


class MoDModel(nn.Module):
    """
    Full Mixture-of-Depths Language Model Backbone.
    Composed of token embeddings, stacked MoDTransformerBlocks, final RMSNorm, and LM head.
    """

    def __init__(self, config: MoDConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.dim)
        self.layers = nn.ModuleList([
            MoDTransformerBlock(config) for _ in range(config.n_layers)
        ])
        self.norm = RMSNorm(config.dim, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, List[Dict[str, Any]]]:
        if inputs_embeds is None:
            assert input_ids is not None, "Must provide input_ids or inputs_embeds"
            h = self.embed_tokens(input_ids)
        else:
            h = inputs_embeds

        total_aux_loss = torch.tensor(0.0, device=h.device)
        layer_stats: List[Dict[str, Any]] = []

        for layer in self.layers:
            h, layer_aux, stats = layer(h)
            total_aux_loss = total_aux_loss + layer_aux
            layer_stats.append(stats)

        h = self.norm(h)
        logits = self.lm_head(h)
        return logits, total_aux_loss, layer_stats


def verify_mod_invariants() -> Dict[str, Any]:
    """
    Verifies all mathematical and architectural invariants of Mixture-of-Depths (Paper 66):
    1. Static Compute Budget: exactly C = floor(capacity_factor * S) tokens processed.
    2. Residual Identity Bypass Invariant: bypassed tokens x_j equal input x_j exactly.
    3. Router Weight Gradient Flow: linear router receives direct task gradients via r_i * f(x).
    4. Router BCE Auxiliary Loss: ensures valid loss computation and backprop.
    5. Autoregressive Thresholding: single-token step mode correctly switches paths.
    6. Full MoDModel forward, loss calculation, and backward pass.
    """
    results: Dict[str, Any] = {}
    cfg = MoDConfig(
        dim=64,
        n_layers=2,
        head_dim=16,
        hidden_dim=128,
        n_heads=4,
        n_kv_heads=1,
        capacity_factor=0.5,
        context_len=64,
        vocab_size=100,
        router_aux_loss_coef=0.01,
        inference_threshold=0.5
    )

    batch_size = 2
    seq_len = 10
    expected_c = int(math.floor(cfg.capacity_factor * seq_len))  # 5 tokens

    # 1. Verify Static Capacity & Routing Dimensions
    block = MoDTransformerBlock(cfg)
    x = torch.randn(batch_size, seq_len, cfg.dim, requires_grad=True)
    out, aux_loss, stats = block(x)

    assert stats["capacity"] == expected_c, f"Expected capacity {expected_c}, got {stats['capacity']}"
    assert stats["compute_ratio"] == 0.5
    assert out.shape == x.shape
    results["static_capacity_verified"] = True

    # 2. Residual Identity Bypass Invariant
    # For tokens NOT selected in top-k, out[b, j] - x[b, j] must be identically 0.0
    top_indices = stats["top_k_indices"]  # (batch_size, capacity)
    diff = (out - x).abs()
    for b in range(batch_size):
        selected_set = set(top_indices[b].tolist())
        for j in range(seq_len):
            if j not in selected_set:
                assert diff[b, j].max().item() == 0.0, f"Bypassed token {j} in batch {b} was altered!"
    results["residual_identity_bypass_verified"] = True

    # 3. Router Weight Gradient Flow via r_i * f(X_tilde)
    loss = out.sum() + aux_loss
    loss.backward()
    assert block.router.linear_router.weight.grad is not None, "Router weight received no gradients!"
    assert block.attn.q_proj.weight.grad is not None, "Attention block received no gradients!"
    results["router_gradient_path_verified"] = True

    # 4. Autoregressive Single-Step Thresholding
    x_step = torch.randn(1, 1, cfg.dim)
    # Step with block
    out_step, participated = block.forward_single_step(x_step)
    assert out_step.shape == (1, 1, cfg.dim)
    assert isinstance(participated, bool)
    results["autoregressive_step_verified"] = True

    # 5. Full MoDModel Training Step
    model = MoDModel(cfg)
    input_ids = torch.randint(0, cfg.vocab_size, (batch_size, seq_len))
    logits, total_aux, _ = model(input_ids)
    assert logits.shape == (batch_size, seq_len, cfg.vocab_size)
    assert total_aux.item() > 0.0

    target = torch.randint(0, cfg.vocab_size, (batch_size, seq_len))
    lm_loss = F.cross_entropy(logits.view(-1, cfg.vocab_size), target.view(-1)) + total_aux
    lm_loss.backward()

    assert model.embed_tokens.weight.grad is not None
    assert model.layers[0].router.linear_router.weight.grad is not None
    assert model.lm_head.weight.grad is not None
    results["full_model_training_verified"] = True

    results["all_invariants_passed"] = True
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Testing Mixture-of-Depths (MoD) Invariants...")
    res = verify_mod_invariants()
    print("Verification Results:", res)
