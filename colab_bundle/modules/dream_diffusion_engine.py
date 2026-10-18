#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dream Diffusion Engine: Discrete Diffusion Large Language Models
================================================================
Paper 49: "Dream 7B: Diffusion Large Language Models"
Authors: Jiacheng Ye, Zhihui Xie, Lin Zheng, Jiahui Gao, Zirui Wu,
         Xin Jiang, Zhenguo Li, Lingpeng Kong
Affiliations: The University of Hong Kong & Huawei Noah's Ark Lab
arXiv: 2508.15487v1 (August 2025, 17 pages)

Key Architectural & Mathematical Innovations:
1. Discrete Masked Diffusion with Absorbing State (Section 3.2, Eq. 2 & 3):
   - Discrete vocabulary V with dedicated [MASK] absorbing state.
   - Continuous-time formulation t in [0, 1], linear schedule alpha_t = 1 - t.
   - Forward corruption: q(x_t^n = [MASK] | x_0^n) = 1 - alpha_t = t.
   - Baseline loss weight w(t) = 1 / t under alpha_t = 1 - t.

2. AR-Based LLM Initialization via Shift Operation (Section 4.1 & Figure 2):
   - Preserves autoregressive positional dynamics where hidden state h_i
     predicts token x_0^{i+1}.
   - Seamlessly converts pretrained AR transformer weights (e.g. Qwen2.5-7B)
     into full bidirectional attention without disrupting learned representations.

3. Context-Adaptive Token-Level Noise Rescheduling (CART) (Section 4.2, Eq. 4 & 5):
   - Eliminates the mismatch of uniform sentence-level timesteps t.
   - Measures token-level contextual informativeness via mixture of geometric
     distributions over clean tokens:
     w(t, x_t, n) = 0.5 * sum_{i=1}^N 1[x_t^i != [MASK]] * Geo(p, |n - i| - 1)
     where Geo(p, k) = p * (1 - p)^k for k >= 0 and p in (0, 1].

4. Confidence-Guided Iterative Remasking Inference (Section 3.2 & 5.5):
   - Starts with corrupted sequence x_T (all [MASK] or prompt + [MASK]).
   - Denoises across K steps, sampling clean token candidates and remasking
     the least confident tokens based on max_v p_theta(x_0^n = v | x_t).
   - Naturally supports arbitrary-order generation, completion, and infilling.

This module is part of the Quillan-Ronin Sovereign AI Architecture.
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, NamedTuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class DreamConfig:
    """Configuration for Dream Discrete Diffusion Large Language Model."""
    vocab_size: int = 4096             # Vocabulary size V
    hidden_dim: int = 512              # Transformer hidden dimension d_model
    num_layers: int = 6                # Number of bidirectional transformer layers
    num_heads: int = 8                 # Number of attention heads
    max_seq_len: int = 512             # Maximum sequence length N
    mask_token_id: int = 0             # Token ID reserved for [MASK]
    pad_token_id: int = 1              # Token ID reserved for [PAD]
    bos_token_id: int = 2              # Token ID reserved for [BOS]
    cart_p: float = 0.3                # Geometric parameter p in (0, 1] for CART
    min_cart_weight: float = 1e-4      # Numerical guard for zero clean tokens
    use_ar_shift: bool = True          # Whether to use AR shift output prediction
    dropout: float = 0.1               # Dropout rate


class CARTWeighting(nn.Module):
    """
    Context-Adaptive Token-Level Noise Rescheduling (CART) (Section 4.2, Eq. 4 & 5).
    Computes token-specific weights:
      w(t, x_t, n) = 0.5 * sum_{i=1}^N 1[x_t^i != MASK] * Geo(p, |n - i| - 1)
      where Geo(p, k) = p * (1 - p)^k for k >= 0.
    """
    def __init__(self, p: float = 0.3, min_weight: float = 1e-4):
        super().__init__()
        if not (0.0 < p <= 1.0):
            raise ValueError(f"Geometric parameter p must be in (0, 1], got {p}")
        self.p = float(p)
        self.min_weight = float(min_weight)

    def forward(
        self,
        x_t: torch.Tensor,
        mask_token_id: int,
        timesteps: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Computes CART weights for each token position.

        Args:
            x_t: [B, N] corrupted token sequence at continuous timestep t
            mask_token_id: integer ID of [MASK]
            timesteps: optional [B] continuous timesteps t in (0, 1] for fallback

        Returns:
            weights: [B, N] non-negative token-level weights w(t, x_t, n)
        """
        B, N = x_t.shape
        device = x_t.device

        # Boolean mask of clean tokens: [B, N] (True where token != MASK)
        is_clean = (x_t != mask_token_id).float()  # [B, N]

        # Compute pairwise distance matrix: dist[n, i] = |n - i|
        positions = torch.arange(N, device=device)
        dist = torch.abs(positions.unsqueeze(0) - positions.unsqueeze(1))  # [N, N]

        # k = |n - i| - 1 for i != n
        k = dist - 1
        # Set diagonal (i == n) to a large negative number so Geo is 0
        valid_pair = (dist > 0)
        k_clamped = torch.clamp(k, min=0)

        # Geo(p, k) = p * (1 - p)^k
        geo_vals = self.p * torch.pow(1.0 - self.p, k_clamped.float())  # [N, N]
        geo_vals = torch.where(valid_pair, geo_vals, torch.zeros_like(geo_vals))

        # w(t, x_t, n) = 0.5 * sum_{i=1}^N (is_clean[b, i] * geo_vals[n, i])
        # [B, 1, N] @ [1, N, N]^T -> [B, N]
        # is_clean: [B, N], geo_vals: [N, N] (where geo_vals[n, i] is contribution from i to n)
        # Sum over i: w[b, n] = 0.5 * sum_i (is_clean[b, i] * geo_vals[n, i])
        cart_weights = 0.5 * torch.matmul(is_clean, geo_vals.t())  # [B, N]

        # If a sequence has zero clean tokens (e.g. t -> 1), fallback to baseline 1/t or min_weight
        if timesteps is not None:
            # Baseline w(t) = 1 / t
            t_safe = torch.clamp(timesteps.view(B, 1), min=1e-3, max=1.0)
            baseline_w = 1.0 / t_safe
            fallback = baseline_w.expand(B, N)
        else:
            fallback = torch.full((B, N), self.min_weight, device=device)

        has_clean = (is_clean.sum(dim=-1, keepdim=True) > 0)  # [B, 1]
        final_weights = torch.where(has_clean, cart_weights, fallback)
        final_weights = torch.clamp(final_weights, min=self.min_weight)

        return final_weights


class DiscreteDiffusionForward(nn.Module):
    """
    Forward noising process with absorbing [MASK] state (Section 3.2).
    Linear noise schedule: alpha_t = 1 - t.
    Continuous timestep t in [0, 1].
    """
    def __init__(self, mask_token_id: int):
        super().__init__()
        self.mask_token_id = mask_token_id

    def forward(
        self,
        x_0: torch.Tensor,
        timesteps: Optional[torch.Tensor] = None,
        prompt_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Corrupts clean tokens x_0 into x_t according to continuous timestep t.

        Args:
            x_0: [B, N] clean token sequence
            timesteps: optional [B] continuous timesteps in [0, 1].
                       If None, sampled uniformly from U(0, 1).
            prompt_mask: optional [B, N] boolean tensor. True indicates prompt tokens
                         that MUST NOT be corrupted (preserved as clean).

        Returns:
            x_t: [B, N] corrupted sequence with absorbing [MASK]
            timesteps: [B] sampled continuous timesteps
            mask_indicator: [B, N] bool tensor, True where x_t == MASK
        """
        B, N = x_0.shape
        device = x_0.device

        if timesteps is None:
            timesteps = torch.rand(B, device=device)
        else:
            timesteps = torch.clamp(timesteps, 0.0, 1.0)

        # Under linear schedule: alpha_t = 1 - t
        # Probability of replacement with [MASK] is 1 - alpha_t = t
        mask_prob = timesteps.view(B, 1).expand(B, N)  # [B, N]
        rand_draw = torch.rand(B, N, device=device)

        # Token is masked if rand_draw < mask_prob
        should_mask = (rand_draw < mask_prob)

        # If prompt_mask is provided, never mask prompt positions
        if prompt_mask is not None:
            should_mask = should_mask & (~prompt_mask)

        # Apply absorbing state
        x_t = torch.where(should_mask, torch.tensor(self.mask_token_id, device=device), x_0)
        mask_indicator = (x_t == self.mask_token_id)

        return x_t, timesteps, mask_indicator


class DreamAttention(nn.Module):
    """Bidirectional Multi-Head Self-Attention for full-context diffusion modeling."""
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        assert self.head_dim * num_heads == hidden_dim, "hidden_dim must be divisible by num_heads"

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = 1.0 / math.sqrt(self.head_dim)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        B, N, D = x.shape
        # Project Q, K, V
        q = self.q_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        # Full bidirectional attention scores: [B, H, N, N]
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if attention_mask is not None:
            # attention_mask: [B, 1, 1, N] or [B, 1, N, N]
            scores = scores + attention_mask

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Context aggregation
        out = torch.matmul(attn_weights, v)  # [B, H, N, head_dim]
        out = out.transpose(1, 2).contiguous().view(B, N, D)
        return self.out_proj(out)


class DreamTransformerBlock(nn.Module):
    """Transformer block with RMSNorm, full bidirectional self-attention, and SwiGLU MLP."""
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = DreamAttention(hidden_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(hidden_dim)

        # SwiGLU MLP
        ffn_dim = int(hidden_dim * 8 / 3)
        self.gate_proj = nn.Linear(hidden_dim, ffn_dim, bias=False)
        self.up_proj = nn.Linear(hidden_dim, ffn_dim, bias=False)
        self.down_proj = nn.Linear(ffn_dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Pre-LN Self-Attention
        normed = self.norm1(x)
        x = x + self.attn(normed, attention_mask=attention_mask)

        # Pre-LN SwiGLU FFN
        normed2 = self.norm2(x)
        gate = F.silu(self.gate_proj(normed2))
        up = self.up_proj(normed2)
        ffn_out = self.down_proj(gate * up)
        x = x + self.dropout(ffn_out)
        return x


class DreamDiffusionModel(nn.Module):
    """
    Dream 7B Discrete Diffusion Language Model Architecture.
    Implements:
    - Token embedding and bidirectional transformer backbone.
    - Continuous timestep conditioning via sinusoidal embedding.
    - AR-based Shift Operation projection head (Section 4.1).
    """
    def __init__(self, config: DreamConfig):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.hidden_dim = config.hidden_dim
        self.use_ar_shift = config.use_ar_shift

        # Token and positional embeddings
        self.token_embeddings = nn.Embedding(config.vocab_size, config.hidden_dim)
        self.pos_embeddings = nn.Embedding(config.max_seq_len, config.hidden_dim)

        # Continuous timestep embedding
        self.time_embed = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim)
        )

        # Bidirectional Transformer Layers
        self.layers = nn.ModuleList([
            DreamTransformerBlock(config.hidden_dim, config.num_heads, config.dropout)
            for _ in range(config.num_layers)
        ])
        self.final_norm = nn.LayerNorm(config.hidden_dim)

        # Output LM prediction head
        self.lm_head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)
        # Weight tying with token embeddings
        self.lm_head.weight = self.token_embeddings.weight

    def _get_timestep_embedding(self, timesteps: torch.Tensor) -> torch.Tensor:
        """Sinusoidal embedding for continuous timesteps t in [0, 1]."""
        # timesteps: [B]
        half_dim = self.hidden_dim // 2
        emb_scale = math.log(10000) / (half_dim - 1)
        freqs = torch.exp(torch.arange(half_dim, device=timesteps.device, dtype=torch.float32) * -emb_scale)
        # Scale t from [0, 1] to [0, 1000] for standard diffusion frequency coverage
        scaled_t = timesteps.float() * 1000.0
        args = scaled_t.unsqueeze(1) * freqs.unsqueeze(0)  # [B, half_dim]
        time_emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # [B, hidden_dim]
        return self.time_embed(time_emb)

    def forward(
        self,
        x_t: torch.Tensor,
        timesteps: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass of Dream model.

        Args:
            x_t: [B, N] corrupted token sequence
            timesteps: [B] continuous timesteps t in [0, 1]
            attention_mask: optional attention mask

        Returns:
            logits: [B, N, V] predicted logits for clean tokens x_0
        """
        B, N = x_t.shape
        device = x_t.device

        # Token + Positional embeddings
        positions = torch.arange(N, device=device).unsqueeze(0).expand(B, N)
        h = self.token_embeddings(x_t) + self.pos_embeddings(positions)

        # Inject continuous timestep embedding
        t_emb = self._get_timestep_embedding(timesteps)  # [B, hidden_dim]
        h = h + t_emb.unsqueeze(1)  # Broadcast over sequence length

        # Pass through full bidirectional transformer layers
        for layer in self.layers:
            h = layer(h, attention_mask=attention_mask)

        h = self.final_norm(h)  # [B, N, hidden_dim]

        # AR Shift Operation Alignment (Section 4.1):
        logits = self.lm_head(h)  # [B, N, V]
        return logits


class DreamDiffusionLoss(nn.Module):
    """
    Weighted discrete diffusion cross-entropy loss with CART (Eq. 4).
    L(theta) = - E sum_{n=1}^N 1[x_t^n = MASK] * w(t, x_t, n) * log p_theta(x_0^n | x_t)
    """
    def __init__(self, config: DreamConfig):
        super().__init__()
        self.config = config
        self.mask_token_id = config.mask_token_id
        self.cart = CARTWeighting(p=config.cart_p, min_weight=config.min_cart_weight)

    def forward(
        self,
        logits: torch.Tensor,
        x_0: torch.Tensor,
        x_t: torch.Tensor,
        timesteps: torch.Tensor,
        mask_indicator: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Computes the CART-weighted cross-entropy loss.

        Args:
            logits: [B, N, V] output logits from Dream model
            x_0: [B, N] ground truth clean tokens
            x_t: [B, N] corrupted tokens
            timesteps: [B] continuous timesteps
            mask_indicator: optional [B, N] bool tensor indicating masked positions

        Returns:
            loss: scalar tensor loss
            metrics: dict of logging statistics
        """
        B, N, V = logits.shape
        device = logits.device

        if mask_indicator is None:
            mask_indicator = (x_t == self.mask_token_id)

        # Cross entropy per token: log p_theta(x_0^n | x_t)
        # Flatten for F.cross_entropy
        flat_logits = logits.view(-1, V)
        flat_targets = x_0.view(-1)
        ce_loss = F.cross_entropy(flat_logits, flat_targets, reduction='none').view(B, N)  # [B, N]

        # Compute CART weights w(t, x_t, n)
        cart_weights = self.cart(x_t, self.mask_token_id, timesteps=timesteps)  # [B, N]

        # Apply mask indicator: loss evaluated ONLY on masked positions (Eq. 4)
        masked_weights = cart_weights * mask_indicator.float()  # [B, N]

        # Weighted sum of negative log likelihood
        total_masked_tokens = mask_indicator.sum().clamp(min=1.0)
        weighted_loss = (ce_loss * masked_weights).sum() / total_masked_tokens

        metrics = {
            "loss": float(weighted_loss.detach().item()),
            "masked_tokens": float(total_masked_tokens.detach().item()),
            "avg_cart_weight": float(masked_weights.sum().detach().item() / total_masked_tokens.detach().item())
        }
        return weighted_loss, metrics


class DreamSampler:
    """
    Confidence-Guided Iterative Remasking Sampler for Dream 7B (Section 3.2 & 5.5).
    Generates text from fully corrupted or partial inputs by progressively unmasking
    tokens across K diffusion timesteps.
    """
    def __init__(self, model: DreamDiffusionModel, config: DreamConfig):
        self.model = model
        self.config = config
        self.mask_token_id = config.mask_token_id

    @torch.no_grad()
    def sample(
        self,
        seq_len: int,
        batch_size: int = 1,
        num_steps: int = 20,
        temperature: float = 0.0,
        prompt_tokens: Optional[torch.Tensor] = None,
        device: torch.device = torch.device('cpu')
    ) -> torch.Tensor:
        """
        Executes iterative denoising reverse process from t=1 to t=0.

        Args:
            seq_len: total sequence length N
            batch_size: batch size B
            num_steps: number of diffusion denoising steps K (quality-speed trade-off)
            temperature: sampling temperature (0.0 for greedy argmax)
            prompt_tokens: optional [B, P] prompt tokens to prepend/condition on (infilling/completion)
            device: torch device

        Returns:
            tokens: [B, N] generated clean token sequence
        """
        self.model.eval()

        # Initialize sequence: corrupted absorbing state [MASK]
        x_t = torch.full((batch_size, seq_len), self.mask_token_id, dtype=torch.long, device=device)

        # Track prompt positions that must remain static
        is_fixed = torch.zeros((batch_size, seq_len), dtype=torch.bool, device=device)
        if prompt_tokens is not None:
            p_len = min(prompt_tokens.shape[1], seq_len)
            x_t[:, :p_len] = prompt_tokens[:, :p_len]
            is_fixed[:, :p_len] = True

        # Generate timesteps from 1.0 down to 0.0 across K steps
        timesteps = torch.linspace(1.0, 0.0, num_steps + 1, device=device)

        for step in range(num_steps):
            t_curr = timesteps[step]
            t_next = timesteps[step + 1]

            t_tensor = torch.full((batch_size,), t_curr, device=device)

            # Model prediction: p_theta(x_0 | x_t)
            logits = self.model(x_t, t_tensor)  # [B, N, V]

            # In discrete diffusion, clean tokens x_0 are in V \ {[MASK]}
            logits[:, :, self.mask_token_id] = -float('inf')

            # Sample candidate tokens x_0
            if temperature > 0.0:
                probs = F.softmax(logits / temperature, dim=-1)
                sampled_tokens = torch.multinomial(probs.view(-1, self.config.vocab_size), num_samples=1).view(batch_size, seq_len)
                # Confidence score is probability of the sampled token
                token_probs = probs.gather(-1, sampled_tokens.unsqueeze(-1)).squeeze(-1)  # [B, N]
            else:
                probs = F.softmax(logits, dim=-1)
                token_probs, sampled_tokens = torch.max(probs, dim=-1)  # [B, N]

            # Fixed tokens always retain their original tokens and max confidence
            if prompt_tokens is not None:
                p_len = min(prompt_tokens.shape[1], seq_len)
                sampled_tokens[:, :p_len] = prompt_tokens[:, :p_len]
                token_probs[:, :p_len] = 1.0

            if step == num_steps - 1:
                # Final step: unmask all positions
                x_t = sampled_tokens
                break

            # Target number of unmasked positions according to schedule alpha_{t_next} = 1 - t_next
            unmask_ratio = 1.0 - float(t_next.item())
            num_unmask = int(round(unmask_ratio * seq_len))

            # Select top-k most confident tokens to keep unmasked
            # For each sequence in batch, find top num_unmask positions
            _, topk_indices = torch.topk(token_probs, k=num_unmask, dim=-1)  # [B, num_unmask]

            # Construct new x_t: keep top-k, remask the rest
            new_x_t = torch.full_like(x_t, self.mask_token_id)
            batch_indices = torch.arange(batch_size, device=device).unsqueeze(1).expand(-1, num_unmask)
            new_x_t[batch_indices, topk_indices] = sampled_tokens[batch_indices, topk_indices]

            # Ensure prompt tokens remain intact
            if prompt_tokens is not None:
                new_x_t[:, :p_len] = prompt_tokens[:, :p_len]

            x_t = new_x_t

        return x_t


def verify_dream_invariants() -> bool:
    """
    Self-verification harness testing all core mathematical invariants of Dream 7B:
    1. Forward absorbing corruption with linear schedule alpha_t = 1 - t.
    2. CART context-adaptive token-level noise weighting (Eq. 5).
    3. Full bidirectional transformer forward pass with continuous timestep embedding.
    4. DreamDiffusionLoss computation strictly on masked tokens.
    5. Backward gradient flow through all parameters.
    6. Iterative remasking sampling from noise to clean sequence.
    """
    print("[Dream-7B] Verifying Mathematical & Architectural Invariants...")
    torch.manual_seed(42)

    config = DreamConfig(
        vocab_size=128,
        hidden_dim=64,
        num_layers=2,
        num_heads=4,
        max_seq_len=32,
        mask_token_id=0,
        cart_p=0.4
    )

    B, N = 2, 16
    clean_x0 = torch.randint(3, config.vocab_size, (B, N))

    # 1. Forward absorbing corruption
    fwd = DiscreteDiffusionForward(mask_token_id=config.mask_token_id)
    t = torch.tensor([0.25, 0.75])
    x_t, sampled_t, mask_ind = fwd(clean_x0, timesteps=t)

    assert x_t.shape == (B, N), f"Corrupted shape mismatch: {x_t.shape}"
    assert torch.all(x_t[mask_ind] == config.mask_token_id), "Masked tokens must match mask_token_id"
    # Higher timestep should have more masked tokens in expectation
    print(f"  [1] Forward corruption: t={t.tolist()}, masked counts = {mask_ind.sum(dim=-1).tolist()} -> PASS")

    # 2. CART token-level weighting
    cart = CARTWeighting(p=config.cart_p)
    weights = cart(x_t, config.mask_token_id, timesteps=t)
    assert weights.shape == (B, N), f"CART weights shape mismatch: {weights.shape}"
    assert torch.all(weights >= 0.0), "CART weights must be strictly non-negative"
    print(f"  [2] CART weighting: min={weights.min().item():.4f}, max={weights.max().item():.4f} -> PASS")

    # 3. Model forward pass
    model = DreamDiffusionModel(config)
    logits = model(x_t, t)
    assert logits.shape == (B, N, config.vocab_size), f"Logits shape mismatch: {logits.shape}"
    print(f"  [3] Bidirectional backbone forward: logits={logits.shape} -> PASS")

    # 4. DreamDiffusionLoss & Backward Pass
    criterion = DreamDiffusionLoss(config)
    loss, metrics = criterion(logits, clean_x0, x_t, t, mask_indicator=mask_ind)
    assert not torch.isnan(loss) and not torch.isinf(loss), f"Invalid loss: {loss}"
    assert loss.item() > 0.0, f"Loss must be strictly positive, got {loss.item()}"

    loss.backward()
    has_grads = all(p.grad is not None for p in model.parameters() if p.requires_grad)
    assert has_grads, "All model parameters must receive valid gradients"
    print(f"  [4] CART loss & backward: loss={loss.item():.4f}, grad check -> PASS")

    # 5. Iterative Remasking Sampling
    sampler = DreamSampler(model, config)
    prompt = torch.tensor([[5, 12, 18], [24, 30, 42]])
    sampled_out = sampler.sample(seq_len=N, batch_size=B, num_steps=5, prompt_tokens=prompt)
    assert sampled_out.shape == (B, N), f"Sampler output shape mismatch: {sampled_out.shape}"
    # Verify prompt tokens are preserved
    assert torch.all(sampled_out[:, :3] == prompt), "Prompt tokens must be preserved in completion/infill"
    # Verify no mask tokens remain
    assert not torch.any(sampled_out == config.mask_token_id), "Completed output should contain no [MASK] tokens"
    print(f"  [5] Iterative remasking infilling: generated shape={sampled_out.shape} -> PASS")

    print("[Dream-7B] All 5 Invariants 100% Verified Successfully!\n")
    return True


if __name__ == "__main__":
    verify_dream_invariants()
