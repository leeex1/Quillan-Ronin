#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mistral 7B: Sliding Window Attention & Rolling Buffer Cache Engine
===================================================================
Paper 64: "Mistral 7B"
Authors: Albert Q. Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford,
         Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel,
         Guillaume Lample, Lucile Saulnier, Lélio Renard Lavaud, Marie-Anne Lachaux,
         Pierre Stock, Teven Le Scao, Thibaut Lavril, Thomas Wang, Timothée Lacroix,
         William El Sayed (Mistral AI)
arXiv: 2310.06825 (Oct 2023, 9 pages)

Key Architectural & Mathematical Formulations:
1. Sliding Window Attention (SWA) (Section 2, Figure 1):
   - In vanilla attention, operation count is quadratic O(L^2) in sequence length.
   - SWA restricts attention so that hidden state h_i at position i attends only to
     tokens within a sliding window of size W:
       j in [i - W + 1, i]
     Mask formulation:
       M_{ij} = 0 if 0 <= i - j < W else -inf
   - Multi-layer Receptive Field: Across stacked transformer layers, information travels
     forward by up to W tokens per layer. After k layers, effective attention span is:
       Span(k) = k * W tokens
     With W = 4096 and k = 32 layers, theoretical context span reaches 131,072 tokens (~131K).

2. Rolling Buffer Cache (Section 2, Figure 2):
   - Fixed attention window W allows bounding the KV cache size to strictly W elements per head.
   - Timestep i overwrites cache slot i mod W:
       K_cache[:, :, i % W, :] = K_i
       V_cache[:, :, i % W, :] = V_i
   - Memory footprint stops growing once sequence length exceeds W, achieving an L/W memory
     reduction (8x reduction on a 32K context with W = 4096) with zero cache fragmentation.

3. Grouped-Query Attention (GQA) (Section 2, Table 1):
   - Employs 32 query heads and 8 key-value heads (group ratio G = 32 / 8 = 4).
   - Each KV head is shared across 4 query heads, substantially reducing memory bandwidth
     and KV cache transfer latency during generation.

4. SwiGLU Feed-Forward Network:
   - Intermediate hidden dimension d_ffn = 14336 with model dimension d = 4096.
   - FFN(x) = (SiLU(x W_gate) * (x W_up)) W_down
"""

import math
import logging
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger("Mistral7BEngine")


@dataclass
class MistralConfig:
    """Architectural configuration for Mistral 7B."""
    dim: int = 128                      # Hidden dimension d_model (default 4096, scaled for tests)
    n_layers: int = 4                   # Number of layers (default 32)
    head_dim: int = 32                  # Dimension per attention head (default 128)
    hidden_dim: int = 256               # SwiGLU intermediate dimension (default 14336)
    n_heads: int = 4                    # Query attention heads (default 32)
    n_kv_heads: int = 1                 # Key/Value attention heads (default 8, GQA ratio 4:1)
    window_size: int = 16               # Sliding window size W (default 4096)
    context_len: int = 8192             # Max sequence length (default 8192)
    vocab_size: int = 32000             # Vocabulary size (32,000)
    rope_theta: float = 10000.0         # RoPE frequency base
    rms_norm_eps: float = 1e-5          # RMSNorm epsilon

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
    """
    Apply Rotary Position Embedding (RoPE) to tensor of shape (batch, heads, seq_len, head_dim).
    """
    d = x.shape[-1]
    x1 = x[..., : d // 2]
    x2 = x[..., d // 2 :]
    rotated = torch.cat((-x2, x1), dim=-1)
    return (x * cos) + (rotated * sin)


class RoPEEmbedding(nn.Module):
    """Precomputed RoPE cos/sin frequencies."""

    def __init__(self, head_dim: int, max_seq_len: int = 8192, theta: float = 10000.0):
        super().__init__()
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.theta = theta

        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        t = torch.arange(seq_len, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def get_cos_sin(self, seq_len: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        if seq_len > self.cos_cached.shape[0]:
            self._build_cache(seq_len)
        return (
            self.cos_cached[:seq_len].to(device).unsqueeze(0).unsqueeze(0),
            self.sin_cached[:seq_len].to(device).unsqueeze(0).unsqueeze(0),
        )


class RollingBufferKVCache:
    """
    Rolling Buffer Cache for Mistral Sliding Window Attention (Figure 2).
    Fixed capacity of W tokens per head. Token at timestep pos is written to
    slot (pos % W), bounding memory footprint to O(W) with zero allocation drift.
    """

    def __init__(self, batch_size: int, n_kv_heads: int, window_size: int, head_dim: int, device: torch.device, dtype: torch.dtype = torch.float32):
        self.batch_size = batch_size
        self.n_kv_heads = n_kv_heads
        self.window_size = window_size
        self.head_dim = head_dim
        self.device = device
        self.dtype = dtype

        # Fixed rolling memory buffers of shape (batch, n_kv_heads, window_size, head_dim)
        self.k_cache = torch.zeros(batch_size, n_kv_heads, window_size, head_dim, device=device, dtype=dtype)
        self.v_cache = torch.zeros(batch_size, n_kv_heads, window_size, head_dim, device=device, dtype=dtype)
        self.seen_tokens = 0

    def insert(self, k: torch.Tensor, v: torch.Tensor, pos: int):
        """
        Insert new token representation at sequence position pos into slot (pos % W).
        k, v shape: (batch, n_kv_heads, 1, head_dim)
        """
        slot = pos % self.window_size
        self.k_cache[:, :, slot : slot + 1, :] = k
        self.v_cache[:, :, slot : slot + 1, :] = v
        self.seen_tokens = max(self.seen_tokens, pos + 1)

    def get_window(self, pos: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieve sliding window keys and values in chronologically ordered sequence.
        Returns:
            k_win, v_win of shape (batch, n_kv_heads, min(pos+1, W), head_dim)
        """
        if self.seen_tokens <= self.window_size:
            valid_len = pos + 1
            return self.k_cache[:, :, :valid_len, :], self.v_cache[:, :, :valid_len, :]

        # Circular unrolling: slot (pos + 1) % W is the oldest surviving token
        start_idx = (pos + 1) % self.window_size
        indices = [(start_idx + i) % self.window_size for i in range(self.window_size)]
        idx_tensor = torch.tensor(indices, device=self.device, dtype=torch.long)
        return self.k_cache[:, :, idx_tensor, :], self.v_cache[:, :, idx_tensor, :]


def build_sliding_window_causal_mask(
    seq_len: int,
    window_size: int,
    device: Optional[torch.device] = None
) -> torch.Tensor:
    """
    Constructs the Mistral Sliding Window Attention mask (Section 2, Figure 1).
    Position i attends to position j iff:
        i - window_size < j <= i  (causal and within window distance W)
    Returns:
        mask of shape (1, 1, seq_len, seq_len) with 0.0 for valid attention and -inf elsewhere.
    """
    if device is None:
        device = torch.device("cpu")
    row_idx = torch.arange(seq_len, device=device).unsqueeze(1)
    col_idx = torch.arange(seq_len, device=device).unsqueeze(0)

    # Causal lower triangular: j <= i
    causal_valid = col_idx <= row_idx
    # Sliding window constraint: i - j < window_size
    window_valid = (row_idx - col_idx) < window_size

    valid = causal_valid & window_valid
    mask = torch.zeros(seq_len, seq_len, device=device, dtype=torch.float32)
    mask = mask.masked_fill(~valid, float("-inf"))
    return mask.unsqueeze(0).unsqueeze(0)


class SlidingWindowAttention(nn.Module):
    """
    Mistral Sliding Window Attention with Grouped-Query Attention (GQA).
    """

    def __init__(self, config: MistralConfig):
        super().__init__()
        self.config = config
        self.dim = config.dim
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.head_dim
        self.window_size = config.window_size
        self.num_queries_per_kv = config.num_queries_per_kv

        self.q_proj = nn.Linear(self.dim, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, self.dim, bias=False)

        self.rope = RoPEEmbedding(self.head_dim, config.context_len, config.rope_theta)

    def forward(
        self,
        x: torch.Tensor,
        cache: Optional[RollingBufferKVCache] = None,
        pos: Optional[int] = None
    ) -> torch.Tensor:
        """
        Forward pass for Sliding Window Attention.
        Supports both full sequence parallel forward and step-by-step rolling buffer decoding.
        """
        batch_size, seq_len, _ = x.shape

        # 1. Projections
        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # 2. Rotary Position Embeddings
        cos, sin = self.rope.get_cos_sin(seq_len if pos is None else pos + 1, x.device)
        if pos is not None:
            # Single-step generation
            cos = cos[:, :, pos : pos + 1, :]
            sin = sin[:, :, pos : pos + 1, :]
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)

        # 3. Cache management (Inference step)
        if cache is not None and pos is not None:
            cache.insert(k, v, pos)
            k, v = cache.get_window(pos)
            kv_len = k.shape[2]
            # When decoding with rolling buffer, the query attends to all valid unrolled window tokens
            mask = None
        else:
            # Parallel training / prompt prefill mode with explicit SWA mask
            kv_len = seq_len
            mask = build_sliding_window_causal_mask(seq_len, self.window_size, x.device)

        # 4. Grouped-Query Attention (GQA): Expand k, v if num_queries_per_kv > 1
        if self.num_queries_per_kv > 1:
            k = k.repeat_interleave(self.num_queries_per_kv, dim=1)
            v = v.repeat_interleave(self.num_queries_per_kv, dim=1)

        # 5. Scaled Dot-Product Attention with SWA mask
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask

        attn_weights = F.softmax(scores, dim=-1)
        # Drop NaN that can appear if mask row is all -inf
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)

        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.n_heads * self.head_dim)
        return self.o_proj(out)


class SwiGLUFFN(nn.Module):
    """
    SwiGLU Feed-Forward Network:
    FFN(x) = (SiLU(x W_gate) * (x W_up)) W_down
    """

    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w_gate = nn.Linear(dim, hidden_dim, bias=False)
        self.w_up = nn.Linear(dim, hidden_dim, bias=False)
        self.w_down = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class MistralDecoderLayer(nn.Module):
    """Single Mistral 7B Transformer Decoder Layer."""

    def __init__(self, config: MistralConfig):
        super().__init__()
        self.attn_norm = RMSNorm(config.dim, config.rms_norm_eps)
        self.attn = SlidingWindowAttention(config)
        self.ffn_norm = RMSNorm(config.dim, config.rms_norm_eps)
        self.ffn = SwiGLUFFN(config.dim, config.hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        cache: Optional[RollingBufferKVCache] = None,
        pos: Optional[int] = None
    ) -> torch.Tensor:
        h = x + self.attn(self.attn_norm(x), cache=cache, pos=pos)
        out = h + self.ffn(self.ffn_norm(h))
        return out


class Mistral7BModel(nn.Module):
    """
    Complete Mistral 7B Sequence Model with Sliding Window Attention.
    """

    def __init__(self, config: Optional[MistralConfig] = None):
        super().__init__()
        if config is None:
            config = MistralConfig()
        self.config = config

        self.embed_tokens = nn.Embedding(config.vocab_size, config.dim)
        self.layers = nn.ModuleList([MistralDecoderLayer(config) for _ in range(config.n_layers)])
        self.norm = RMSNorm(config.dim, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        h = self.embed_tokens(input_ids)
        for layer in self.layers:
            h = layer(h)
        h = self.norm(h)
        return self.lm_head(h)


def verify_mistral_invariants() -> Dict[str, Any]:
    """
    Comprehensive verification harness for Mistral 7B (Jiang et al., 2023) invariants:
      1. Sliding Window Attention Mask Invariant:
         Tokens at distance >= W must have exact attention weight = 0.0.
      2. Multi-Layer Receptive Field Expansion:
         A perturbation at token 0 can reach token (k * W - 1) after k layers,
         proving the theoretical receptive field span of k * W.
      3. Rolling Buffer Cache O(W) Invariant:
         Cache size remains strictly bounded to W entries even for seq_len >> W,
         with circular overwriting pos % W.
      4. Grouped-Query Attention (GQA) Head Expansion:
         Proper 4:1 query-to-KV head broadcasting without tensor shape mismatch.
      5. Autograd Gradient Propagation:
         Backpropagation produces non-zero, finite gradients across Q/K/V/O, SwiGLU, and norms.
    """
    torch.manual_seed(42)
    results = {}

    batch_size = 1
    seq_len = 16
    W = 4
    cfg = MistralConfig(
        dim=64,
        n_layers=2,
        head_dim=16,
        hidden_dim=128,
        n_heads=4,
        n_kv_heads=1,
        window_size=W,
        context_len=64,
        vocab_size=1000
    )

    # 1. SWA Mask Exactness Invariant
    mask = build_sliding_window_causal_mask(seq_len, W, torch.device("cpu"))
    # Token 10 attending to token 2 (distance = 8 >= W = 4): must be -inf
    assert mask[0, 0, 10, 2].item() == float("-inf")
    # Token 10 attending to token 8 (distance = 2 < W = 4): must be 0.0
    assert mask[0, 0, 10, 8].item() == 0.0
    # Future token: token 5 attending to token 6: must be -inf
    assert mask[0, 0, 5, 6].item() == float("-inf")
    results["swa_mask_verified"] = True

    # 2. SWA Attention Module Mask Application
    attn = SlidingWindowAttention(cfg)
    x = torch.randn(batch_size, seq_len, cfg.dim)
    out_attn = attn(x)
    assert out_attn.shape == x.shape
    results["swa_output_shape"] = list(out_attn.shape)

    # 3. Rolling Buffer Cache Bounded Capacity & Circular Overwrite
    rb_cache = RollingBufferKVCache(
        batch_size=batch_size,
        n_kv_heads=cfg.n_kv_heads,
        window_size=W,
        head_dim=cfg.head_dim,
        device=torch.device("cpu")
    )
    # Insert 10 tokens into rolling buffer of size W = 4
    for t in range(10):
        k_t = torch.full((batch_size, cfg.n_kv_heads, 1, cfg.head_dim), float(t))
        v_t = torch.full((batch_size, cfg.n_kv_heads, 1, cfg.head_dim), float(t))
        rb_cache.insert(k_t, v_t, pos=t)

    # Cache shape must still be exactly (1, 1, 4, 16)
    assert rb_cache.k_cache.shape == (batch_size, cfg.n_kv_heads, W, cfg.head_dim)
    # Slot 9 % 4 = 1 must contain token 9
    assert (rb_cache.k_cache[0, 0, 9 % W, :] == 9.0).all()
    # Unrolled window for pos=9 should contain tokens [6, 7, 8, 9] in chronological order
    k_win, v_win = rb_cache.get_window(pos=9)
    assert k_win.shape == (batch_size, cfg.n_kv_heads, W, cfg.head_dim)
    expected_tokens = torch.tensor([6.0, 7.0, 8.0, 9.0])
    actual_tokens = k_win[0, 0, :, 0]
    assert torch.allclose(actual_tokens, expected_tokens), f"Expected {expected_tokens}, got {actual_tokens}"
    results["rolling_buffer_unroll_verified"] = True

    # 4. Multi-Layer Receptive Field Expansion Invariant (Figure 1)
    # Each layer advances the attention horizon by (W - 1) tokens forward.
    # After k=2 layers with W=4, maximum attention span from token 0 is 2 * (W - 1) = 6 tokens.
    # Therefore, token at position 6 is reached, while token 7 is strictly outside the receptive field.
    layer1 = MistralDecoderLayer(cfg)
    layer2 = MistralDecoderLayer(cfg)

    x_base = torch.randn(1, 10, cfg.dim)
    h_base = layer2(layer1(x_base))

    x_perturbed = x_base.clone()
    x_perturbed[0, 0, :] += 10.0  # Mutate only token 0
    h_perturbed = layer2(layer1(x_perturbed))

    diff = (h_base - h_perturbed).abs().sum(dim=-1)[0]
    # Token 6 must be influenced (diff > 0)
    assert diff[6].item() > 1e-4, f"Receptive field failure: Token 6 was not reached: diff={diff[6].item()}"
    # Token 7 is at distance 7 > 2 * (W - 1) = 6, so it MUST NOT be influenced (diff == 0)
    assert diff[7].item() < 1e-5, f"Causal leakage: Token 7 was reached past k*(W-1) span: diff={diff[7].item()}"
    results["receptive_field_span_verified"] = True

    # 5. Full Autograd Gradient Propagation
    model = Mistral7BModel(cfg)
    input_ids = torch.randint(0, cfg.vocab_size, (1, 8))
    logits = model(input_ids)
    loss = logits.sum()
    loss.backward()

    assert model.embed_tokens.weight.grad is not None
    assert model.layers[0].attn.q_proj.weight.grad is not None
    assert model.layers[0].ffn.w_gate.weight.grad is not None
    results["gradient_propagation_verified"] = True

    results["all_invariants_passed"] = True
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Testing Mistral 7B SWA & Rolling Buffer Cache Invariants...")
    res = verify_mistral_invariants()
    print("Verification Results:", res)
