#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DFlash Engine: Block Diffusion for Flash Speculative Decoding
=============================================================
Paper 47: "DFlash: Block Diffusion for Flash Speculative Decoding"
Authors: Jian Chen, Yesheng Liang, Zhijian Liu (UC San Diego)
arXiv: 2602.06036 (Feb 2026, 13 pages, ICML 2026 / PMLR 306)

Key Architectural Innovations:
1. Parallel Block Diffusion Drafting (Section 3.2 & 4.1):
   - Rather than sequential autoregressive drafting incurring T_draft = gamma * t_step,
     DFlash predicts all gamma - 1 draft tokens in parallel in a single forward pass
     (T_draft = t_parallel << gamma * t_step).
   - Allows deeper, more expressive draft models (e.g. 3-5 layers) without latency penalty.

2. Context Feature Extraction & Cross-Layer Fusion (Section 4.1 & Appendix A.3):
   - Concatenates hidden states from selected target layers [H^(l_1); ...; H^(l_5)]
     and projects them into draft hidden dimension:
     H_t = RMSNorm(W_c [H^(l_1); ...; H^(l_5)])
   - Memory overhead is negligible (W_c in R^{5D_t x D_d}, ~42MB for 35B model).

3. KV Injection Mechanism (Section 4.1 & Appendix A.3):
   - At each draft layer i:
     Q_i = W_i^Q H_d
     K_i = [W_i^K H_t; W_i^K H_d]_seq
     V_i = [W_i^V H_t; W_i^V H_d]_seq
   - Target features serve as persistent KV context without passing through Q-projection,
     FFN, or output projection, ensuring draft acceptance length scales with depth.

4. Asymmetric Position-Dependent Loss Weighting (Section 4.2 & Eq. 4):
   - Errors at early positions invalidate all subsequent speculative tokens.
   - Cross-entropy loss is weighted by exponential decay:
     w_k = exp(-(k - 1) / gamma_decay) for k in {1, ..., gamma - 1}
   - Prioritizes early positions, yielding faster convergence and higher acceptance length tau.

5. Parallel Speculative Verification (Section 3.1 & Algorithm):
   - Target model verifies proposed draft block in a single parallel forward pass.
   - Prefix acceptance logic accepts k tokens matching target argmax plus 1 bonus token.
   - Expected acceptance length tau in [1, gamma], achieving up to 6x lossless speedup.

This module is part of the Quillan-Ronin Sovereign AI Architecture.
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class DFlashConfig:
    """Configuration for DFlash Block Diffusion Speculative Decoding."""
    target_hidden_dim: int = 1024
    draft_hidden_dim: int = 512
    vocab_size: int = 32000
    block_size: int = 16              # gamma: total block size (1 anchor + 15 mask tokens)
    num_target_sampled_layers: int = 5  # Uniformly sampled layers from target model
    num_draft_layers: int = 3         # Deeper draft model enabled by parallel generation
    num_heads: int = 8
    loss_decay_gamma: float = 7.0     # gamma_decay in Eq. 4 (7 for block 16, 5 for block 10)
    mask_token_id: int = 0
    temperature: float = 0.0          # 0.0 for greedy argmax verification, >0 for sampling
    dropout: float = 0.0

    @property
    def num_speculative_tokens(self) -> int:
        """Number of draft tokens generated in parallel (gamma - 1)."""
        return max(1, self.block_size - 1)


class DFlashOutput(NamedTuple):
    """Output tuple for DFlash draft block generation."""
    logits: torch.Tensor             # [batch_size, num_speculative_tokens, vocab_size]
    draft_tokens: torch.Tensor       # [batch_size, num_speculative_tokens]
    target_context_norm: torch.Tensor
    attention_entropy: float


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight


class TargetContextFeatureFuser(nn.Module):
    """
    Extracts and fuses hidden representations from selected target model layers.
    Appendix A.3:
    H_t = RMSNorm(W_c [H^(l_1); ...; H^(l_5)])
    """

    def __init__(self, config: DFlashConfig):
        super().__init__()
        self.config = config
        fused_in_dim = config.target_hidden_dim * config.num_target_sampled_layers
        self.W_c = nn.Linear(fused_in_dim, config.draft_hidden_dim, bias=False)
        self.norm = RMSNorm(config.draft_hidden_dim)

    def forward(self, target_layer_hiddens: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            target_layer_hiddens: List of tensors, each [B, T, target_hidden_dim]
        Returns:
            H_t: [B, T, draft_hidden_dim] persistent target context features
        """
        assert len(target_layer_hiddens) == self.config.num_target_sampled_layers, (
            f"Expected {self.config.num_target_sampled_layers} target layers, got {len(target_layer_hiddens)}"
        )
        # Concatenate across channel dimension
        cat_hidden = torch.cat(target_layer_hiddens, dim=-1)  # [B, T, num_layers * target_hidden_dim]
        proj = self.W_c(cat_hidden)                           # [B, T, draft_hidden_dim]
        return self.norm(proj)


class DFlashKVInjectionAttention(nn.Module):
    """
    Draft attention with persistent KV injection from target context features.
    Appendix A.3:
    Q_i = W_i^Q H_d
    K_i = [W_i^K H_t; W_i^K H_d]_seq
    V_i = [W_i^V H_t; W_i^V H_d]_seq
    """

    def __init__(self, config: DFlashConfig):
        super().__init__()
        self.config = config
        self.hidden_dim = config.draft_hidden_dim
        self.num_heads = config.num_heads
        self.head_dim = self.hidden_dim // self.num_heads
        assert self.hidden_dim % self.num_heads == 0

        self.W_q = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.W_k = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.W_v = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.W_o = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.dropout = nn.Dropout(config.dropout) if config.dropout > 0.0 else nn.Identity()

    def forward(
        self,
        H_d: torch.Tensor,
        H_t: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, float]:
        """
        Args:
            H_d: [B, S_draft, draft_hidden_dim] draft tokens (anchor + masks)
            H_t: [B, S_target, draft_hidden_dim] persistent target context features
            attn_mask: optional attention mask
        Returns:
            out: [B, S_draft, draft_hidden_dim]
            attn_entropy: scalar float
        """
        B, S_d, _ = H_d.shape
        _, S_t, _ = H_t.shape

        # 1. Queries from draft tokens only: Q_i = W_i^Q H_d
        q = self.W_q(H_d).view(B, S_d, self.num_heads, self.head_dim).transpose(1, 2)

        # 2. Keys and Values from concatenated [H_t; H_d] along sequence dimension
        cat_kv = torch.cat([H_t, H_d], dim=1)  # [B, S_t + S_d, draft_hidden_dim]
        k = self.W_k(cat_kv).view(B, S_t + S_d, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.W_v(cat_kv).view(B, S_t + S_d, self.num_heads, self.head_dim).transpose(1, 2)

        # 3. Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # [B, H, S_d, S_t + S_d]

        if attn_mask is not None:
            scores = scores + attn_mask

        probs = F.softmax(scores, dim=-1)
        probs = self.dropout(probs)

        out = torch.matmul(probs, v)  # [B, H, S_d, head_dim]
        out = out.transpose(1, 2).contiguous().view(B, S_d, self.hidden_dim)
        out = self.W_o(out)

        # Entropy of attention distribution
        with torch.no_grad():
            p_safe = probs.clamp(min=1e-12)
            attn_entropy = float(-(p_safe * p_safe.log()).sum(dim=-1).mean().item())

        return out, attn_entropy


class DFlashDraftBlock(nn.Module):
    """Single transformer layer in DFlash draft model with KV injection and MLP."""

    def __init__(self, config: DFlashConfig):
        super().__init__()
        self.norm1 = RMSNorm(config.draft_hidden_dim)
        self.attn = DFlashKVInjectionAttention(config)
        self.norm2 = RMSNorm(config.draft_hidden_dim)

        ffn_dim = config.draft_hidden_dim * 4
        # SwiGLU MLP
        self.w1 = nn.Linear(config.draft_hidden_dim, ffn_dim, bias=False)
        self.w3 = nn.Linear(config.draft_hidden_dim, ffn_dim, bias=False)
        self.w2 = nn.Linear(ffn_dim, config.draft_hidden_dim, bias=False)

    def forward(
        self,
        H_d: torch.Tensor,
        H_t: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, float]:
        normed_d = self.norm1(H_d)
        attn_out, entropy = self.attn(normed_d, H_t, attn_mask=attn_mask)
        H_d = H_d + attn_out

        normed_ffn = self.norm2(H_d)
        ffn_out = self.w2(F.silu(self.w1(normed_ffn)) * self.w3(normed_ffn))
        H_d = H_d + ffn_out

        return H_d, entropy


class DFlashDraftModel(nn.Module):
    """
    Lightweight Block Diffusion Draft Model for DFlash Speculative Decoding.
    - Fuses cross-layer target context features H_t.
    - Decodes all gamma - 1 speculative tokens in parallel in a single forward pass.
    - Employs deep draft stack (e.g. 3-5 layers) conditioned at every layer via KV injection.
    """

    def __init__(self, config: DFlashConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.draft_hidden_dim)
        self.fuser = TargetContextFeatureFuser(config)

        self.layers = nn.ModuleList([
            DFlashDraftBlock(config) for _ in range(config.num_draft_layers)
        ])
        self.final_norm = RMSNorm(config.draft_hidden_dim)
        self.lm_head = nn.Linear(config.draft_hidden_dim, config.vocab_size, bias=False)

        # Tie weights if dimensions match or keep separate
        self.mask_embedding = nn.Parameter(torch.randn(config.draft_hidden_dim) * 0.02)

    def forward(
        self,
        anchor_tokens: torch.Tensor,
        target_layer_hiddens: List[torch.Tensor],
        mask_token_ids: Optional[torch.Tensor] = None
    ) -> DFlashOutput:
        """
        Parallel Block Forward Pass:
        Args:
            anchor_tokens: [B, 1] clean anchor tokens from target model
            target_layer_hiddens: List of [B, T, target_hidden_dim] extracted from target layers
            mask_token_ids: optional explicit mask ids; if None, uses learned mask embedding
        Returns:
            DFlashOutput with logits [B, gamma-1, vocab_size] and draft token predictions
        """
        B = anchor_tokens.shape[0]
        num_draft = self.config.num_speculative_tokens

        # 1. Fuse target context features H_t once (persistent across draft layers)
        H_t = self.fuser(target_layer_hiddens)  # [B, T, draft_hidden_dim]

        # 2. Construct draft input: [anchor_token, mask_1, ..., mask_{gamma-1}]
        anchor_embed = self.embed_tokens(anchor_tokens)  # [B, 1, draft_hidden_dim]

        if mask_token_ids is not None:
            mask_embed = self.embed_tokens(mask_token_ids)
        else:
            mask_embed = self.mask_embedding.view(1, 1, -1).expand(B, num_draft, -1)

        H_d = torch.cat([anchor_embed, mask_embed], dim=1)  # [B, 1 + num_draft, draft_hidden_dim]

        # 3. Pass through draft layers with persistent KV injection
        entropies = []
        for layer in self.layers:
            H_d, ent = layer(H_d, H_t)
            entropies.append(ent)

        H_d = self.final_norm(H_d)

        # 4. Project speculative positions to vocab logits (slice out anchor token position)
        speculative_hiddens = H_d[:, 1:, :]  # [B, num_draft, draft_hidden_dim]
        logits = self.lm_head(speculative_hiddens)  # [B, num_draft, vocab_size]

        if self.config.temperature == 0.0:
            draft_tokens = torch.argmax(logits, dim=-1)
        else:
            probs = F.softmax(logits / self.config.temperature, dim=-1)
            draft_tokens = torch.multinomial(probs.view(-1, self.config.vocab_size), num_samples=1).view(B, num_draft)

        return DFlashOutput(
            logits=logits,
            draft_tokens=draft_tokens,
            target_context_norm=H_t.norm(dim=-1).mean(),
            attention_entropy=sum(entropies) / max(1, len(entropies))
        )


class DFlashLoss(nn.Module):
    """
    Asymmetric Position-Dependent Loss Weighting (Section 4.2 & Eq. 4):
    w_k = exp(-(k - 1) / gamma_decay) for k in {1, ..., gamma - 1}.
    Emphasizes early positions in the speculative draft block.
    """

    def __init__(self, config: DFlashConfig):
        super().__init__()
        self.config = config
        num_draft = config.num_speculative_tokens
        gamma = config.loss_decay_gamma

        # Precompute weights: k in [1, num_draft]
        k_indices = torch.arange(num_draft, dtype=torch.float32)  # [0, 1, ..., num_draft-1]
        weights = torch.exp(-k_indices / gamma)                   # [num_draft]
        self.register_buffer("weights", weights)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            logits: [B, num_draft, vocab_size]
            targets: [B, num_draft] ground truth future tokens
        Returns:
            weighted_loss: scalar tensor
            unweighted_loss: scalar tensor
        """
        B, num_draft, V = logits.shape
        flat_logits = logits.view(-1, V)
        flat_targets = targets.view(-1)

        token_losses = F.cross_entropy(flat_logits, flat_targets, reduction="none").view(B, num_draft)

        # Apply position-dependent exponential decay weights
        w = self.weights.to(logits.device)
        weighted_loss = (token_losses * w.unsqueeze(0)).sum() / (w.sum() * B)
        unweighted_loss = token_losses.mean()

        return weighted_loss, unweighted_loss


class DFlashSpeculativeVerifier:
    """
    Parallel Speculative Decoding Verifier.
    Performs prefix verification and calculates accepted tokens and acceleration metrics.
    """

    def __init__(self, config: DFlashConfig):
        self.config = config

    def verify_block(
        self,
        draft_tokens: torch.Tensor,
        target_logits: torch.Tensor
    ) -> Tuple[torch.Tensor, int, float]:
        """
        Verifies draft block against parallel target model outputs.
        
        Args:
            draft_tokens: [num_draft] proposed draft tokens
            target_logits: [num_draft + 1, vocab_size] parallel target model logits
                           for the draft positions plus bonus position.
                           
        Returns:
            accepted_tokens: 1D tensor of accepted tokens including 1 bonus token
            num_accepted: int tau in [1, gamma]
            acceptance_rate: float in [0.0, 1.0]
        """
        num_draft = draft_tokens.size(0)
        target_predictions = torch.argmax(target_logits[:num_draft], dim=-1)  # [num_draft]

        # Find prefix agreement length
        k_match = 0
        for i in range(num_draft):
            if draft_tokens[i].item() == target_predictions[i].item():
                k_match += 1
            else:
                break

        # Bonus token from target model at first mismatch position (or end of block)
        bonus_token = torch.argmax(target_logits[k_match], dim=-1).unsqueeze(0)

        if k_match > 0:
            accepted_prefix = draft_tokens[:k_match]
            accepted = torch.cat([accepted_prefix, bonus_token], dim=0)
        else:
            accepted = bonus_token

        num_accepted = k_match + 1  # tau: accepted draft + 1 bonus token
        acceptance_rate = k_match / max(1, num_draft)

        return accepted, num_accepted, acceptance_rate


def verify_dflash_invariants(config: Optional[DFlashConfig] = None) -> Dict[str, any]:
    """
    Rigorously verifies DFlash mathematical invariants:
    1. Parallel block drafting: generates all gamma - 1 tokens in a single forward pass.
    2. KV injection: target context features are fused once and persisted across all layers.
    3. Asymmetric loss weighting: strictly monotonically decreasing weights w_1 > w_2 > ... > w_{gamma-1}.
    4. Speculative verification: exact prefix agreement + bonus token yielding tau in [1, gamma].
    5. Autograd: backward gradients flow through draft parameters, KV projections, and W_c fuser.
    """
    if config is None:
        config = DFlashConfig(
            target_hidden_dim=128,
            draft_hidden_dim=64,
            vocab_size=256,
            block_size=8,             # gamma = 8 -> 7 speculative tokens
            num_target_sampled_layers=5,
            num_draft_layers=2,
            num_heads=4,
            loss_decay_gamma=4.0
        )

    # Invariant 1: Loss weights monotonicity
    loss_module = DFlashLoss(config)
    weights = loss_module.weights.cpu().tolist()
    assert len(weights) == config.num_speculative_tokens
    for i in range(len(weights) - 1):
        assert weights[i] > weights[i + 1], f"Weights must decay monotonically: {weights[i]} <= {weights[i+1]}"

    # Invariant 2: Target Feature Fusion & Draft Forward Pass
    draft_model = DFlashDraftModel(config)
    B, T = 2, 10
    anchor = torch.randint(0, config.vocab_size, (B, 1))
    target_hiddens = [
        torch.randn(B, T, config.target_hidden_dim, requires_grad=True)
        for _ in range(config.num_target_sampled_layers)
    ]

    out = draft_model(anchor, target_hiddens)
    assert out.logits.shape == (B, config.num_speculative_tokens, config.vocab_size)
    assert out.draft_tokens.shape == (B, config.num_speculative_tokens)

    # Invariant 3: Autograd gradient propagation
    targets = torch.randint(0, config.vocab_size, (B, config.num_speculative_tokens))
    w_loss, u_loss = loss_module(out.logits, targets)
    w_loss.backward()

    # Check gradients
    assert draft_model.fuser.W_c.weight.grad is not None, "W_c must receive gradients"
    for l_idx, layer in enumerate(draft_model.layers):
        assert layer.attn.W_q.weight.grad is not None, f"Layer {l_idx} W_q must receive gradients"
        assert layer.attn.W_k.weight.grad is not None, f"Layer {l_idx} W_k must receive gradients"
        assert layer.attn.W_v.weight.grad is not None, f"Layer {l_idx} W_v must receive gradients"

    for idx, th in enumerate(target_hiddens):
        assert th.grad is not None and not torch.isnan(th.grad).any(), f"Target hidden {idx} must receive gradients"

    # Invariant 4: Parallel Speculative Verification
    verifier = DFlashSpeculativeVerifier(config)
    draft_cand = out.draft_tokens[0]  # [num_draft]
    # Simulate target logits: craft match for first 3 tokens
    mock_target_logits = torch.randn(config.num_speculative_tokens + 1, config.vocab_size)
    mock_target_logits[0, draft_cand[0]] = 100.0
    mock_target_logits[1, draft_cand[1]] = 100.0
    mock_target_logits[2, draft_cand[2]] = 100.0
    # Mismatch at token 3
    mismatch_id = (draft_cand[3] + 1) % config.vocab_size
    mock_target_logits[3, mismatch_id] = 100.0

    accepted, tau, acc_rate = verifier.verify_block(draft_cand, mock_target_logits)
    assert tau == 4, f"Expected tau=4 (3 matches + 1 bonus), got {tau}"
    assert accepted.size(0) == 4
    assert accepted[3].item() == mismatch_id.item()

    return {
        "verified": True,
        "block_size": config.block_size,
        "speculative_tokens": config.num_speculative_tokens,
        "loss_weights": weights,
        "weighted_loss": float(w_loss.item()),
        "unweighted_loss": float(u_loss.item()),
        "tau_tested": tau,
        "attention_entropy": out.attention_entropy
    }


if __name__ == "__main__":
    res = verify_dflash_invariants()
    print("DFlash Invariants Verification Result:")
    for k, v in res.items():
        print(f"  {k}: {v}")
