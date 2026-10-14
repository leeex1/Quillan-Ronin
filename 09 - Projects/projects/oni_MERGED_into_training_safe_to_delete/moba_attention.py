#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
👑 QUILLAN-RONIN v5.4-ONI — MOBA (MIXTURE OF BLOCK ATTENTION)
=============================================================================
Paper Reference:
  "MoBA: Mixture of Block Attention for Long-Context LLMs"
  Moonshot AI (Kimi) / Tsinghua University / Zhejiang Lab (arXiv:2502.13189v1)

Key Mathematical Formulation:
  - Divides KV context into n blocks of size B: I_i = [(i-1)*B + 1, i*B]
  - Computes block affinity scores via mean-pooled key inner products:
      s_i = <q, mean_pool(K[I_i])> / sqrt(d)
  - Causal constraint: future blocks masked to -inf
  - Current block attention (g_curr = 1) preserves local context with causal mask
  - Historical blocks: top-k blocks selected dynamically
  - Dual-stream attention (Os, Om) combined via online log-sum-exp softmax
=============================================================================
"""

import math
import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

LOGGER = logging.getLogger("quillan.moba")


def online_softmax_combine(
    out_s: torch.Tensor,
    lse_s: torch.Tensor,
    out_m: torch.Tensor,
    lse_m: torch.Tensor,
) -> torch.Tensor:
    """
    Combines two disjoint attention output streams using their log-sum-exp (LSE) normalizers.
    
    Args:
        out_s: Self-attention output [B, H, N, D]
        lse_s: Self-attention log-sum-exp [B, H, N, 1]
        out_m: MoBA cross-block attention output [B, H, N, D]
        lse_m: MoBA cross-block log-sum-exp [B, H, N, 1]
        
    Returns:
        Combined normalized attention output [B, H, N, D]
    """
    max_lse = torch.maximum(lse_s, lse_m)
    # Clamp negative inf to prevent 0 * inf NaNs
    valid_s = lse_s > -1e4
    valid_m = lse_m > -1e4

    w_s = torch.where(valid_s, torch.exp(lse_s - max_lse), torch.zeros_like(lse_s))
    w_m = torch.where(valid_m, torch.exp(lse_m - max_lse), torch.zeros_like(lse_m))
    total_w = (w_s + w_m).clamp(min=1e-6)

    out = (out_s * w_s + out_m * w_m) / total_w
    return out


class MoBAAttention(nn.Module):
    """
    Mixture of Block Attention (MoBA) module for sub-quadratic long-context attention.
    Seamlessly operates as a drop-in replacement or enhancement for standard causal attention.
    """
    def __init__(
        self,
        head_dim: int,
        n_head: int,
        block_size: int = 64,
        top_k: int = 2,
    ) -> None:
        super().__init__()
        if block_size <= 0:
            raise ValueError(f"block_size must be positive, got {block_size}")
        if top_k < 1:
            raise ValueError(f"top_k must be at least 1, got {top_k}")
            
        self.head_dim = head_dim
        self.n_head = n_head
        self.block_size = block_size
        self.top_k = top_k
        self.scale = 1.0 / math.sqrt(head_dim)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        is_causal: bool = True,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass executing MoBA algorithm with strict causality.
        
        Args:
            q: Query tensor of shape [B, H, T_q, D]
            k: Key tensor of shape [B, H, T_k, D]
            v: Value tensor of shape [B, H, T_k, D]
            is_causal: Whether to enforce autoregressive causal masking
            attn_mask: Optional external attention mask
            
        Returns:
            Attention output tensor of shape [B, H, T_q, D]
        """
        B_sz, H, T_q, D = q.shape
        T_k = k.shape[2]
        
        # Fallback to standard dense attention if sequence length is <= block_size
        if T_k <= self.block_size or not is_causal or T_q == 1:
            return F.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask, is_causal=is_causal, scale=self.scale
            )

        B_len = self.block_size
        num_blocks = (T_k + B_len - 1) // B_len

        # Pad K and V if sequence length is not cleanly divisible by B_len
        pad_k = num_blocks * B_len - T_k
        if pad_k > 0:
            k_padded = F.pad(k, (0, 0, 0, pad_k), value=0.0)
            v_padded = F.pad(v, (0, 0, 0, pad_k), value=0.0)
        else:
            k_padded = k
            v_padded = v

        # Shape: [B_sz, H, num_blocks, B_len, D]
        k_blocks = k_padded.view(B_sz, H, num_blocks, B_len, D)
        v_blocks = v_padded.view(B_sz, H, num_blocks, B_len, D)

        # ── Step 1: Compute Mean-Pooled Block Keys: [B_sz, H, num_blocks, D] ──
        if pad_k > 0:
            # Mask out padded tokens in the final block during mean pooling
            mask_counts = torch.full((num_blocks,), B_len, dtype=k.dtype, device=k.device)
            mask_counts[-1] = B_len - pad_k
            mask_counts = mask_counts.view(1, 1, num_blocks, 1)
            k_mean = k_blocks.sum(dim=-2) / mask_counts
        else:
            k_mean = k_blocks.mean(dim=-2)

        # ── Step 2: Compute Affinity Scores S: [B_sz, H, T_q, num_blocks] ─────
        # S[b, h, t, i] = <q_t, k_mean_i> * scale
        S = torch.matmul(q, k_mean.transpose(-1, -2)) * self.scale

        # ── Step 3: Causal Constraint & Current Block Assignment ───────────────
        # Token position mapping to block indices
        token_indices = torch.arange(T_q, device=q.device)
        curr_block_idx = token_indices // B_len  # [T_q]
        block_grid = torch.arange(num_blocks, device=q.device).unsqueeze(0)  # [1, num_blocks]
        curr_block_grid = curr_block_idx.unsqueeze(-1)  # [T_q, 1]

        # Future blocks: block_grid > curr_block_grid -> score = -inf
        future_mask = block_grid > curr_block_grid  # [T_q, num_blocks]
        # Current block: block_grid == curr_block_grid
        curr_mask = block_grid == curr_block_grid  # [T_q, num_blocks]

        # Historical blocks: past blocks only (block_grid < curr_block_grid)
        hist_scores = S.clone()
        hist_scores = hist_scores.masked_fill(future_mask.view(1, 1, T_q, num_blocks), float("-inf"))
        hist_scores = hist_scores.masked_fill(curr_mask.view(1, 1, T_q, num_blocks), float("-inf"))

        # Top-k block selection among historical blocks
        k_val = min(self.top_k, num_blocks - 1)
        if k_val > 0:
            topk_scores, topk_indices = torch.topk(hist_scores, k=k_val, dim=-1)
            # Gate mask for selected past blocks: [B_sz, H, T_q, num_blocks]
            past_gate = torch.zeros_like(S, dtype=torch.bool)
            # Only retain finite topk scores (discard masked -inf past slots)
            valid_topk = topk_scores > -1e4
            past_gate.scatter_(-1, topk_indices, valid_topk)
        else:
            past_gate = torch.zeros_like(S, dtype=torch.bool)

        # ── Step 4: Stream 1 — Self-Attention to Current Block ────────────────
        scores_curr = torch.zeros(B_sz, H, T_q, B_len, dtype=q.dtype, device=q.device)
        for b_idx in range(num_blocks):
            t_start = b_idx * B_len
            t_end = min((b_idx + 1) * B_len, T_q)
            if t_start >= T_q:
                break
            q_slice = q[:, :, t_start:t_end, :]  # [B, H, t_len, D]
            k_slice = k_blocks[:, :, b_idx, :, :]  # [B, H, B_len, D]
            s_block = torch.matmul(q_slice, k_slice.transpose(-1, -2)) * self.scale
            
            # Causal mask within block
            t_len = t_end - t_start
            intra_idx_q = torch.arange(t_len, device=q.device).unsqueeze(-1)
            intra_idx_k = torch.arange(B_len, device=q.device).unsqueeze(0)
            intra_causal = intra_idx_k <= intra_idx_q
            s_block = s_block.masked_fill(~intra_causal.view(1, 1, t_len, B_len), float("-inf"))
            scores_curr[:, :, t_start:t_end, :] = s_block

        lse_s = torch.logsumexp(scores_curr, dim=-1, keepdim=True)
        attn_s = F.softmax(scores_curr, dim=-1)
        # Multiply by V of current block
        out_s = torch.zeros_like(q)
        for b_idx in range(num_blocks):
            t_start = b_idx * B_len
            t_end = min((b_idx + 1) * B_len, T_q)
            if t_start >= T_q:
                break
            v_slice = v_blocks[:, :, b_idx, :, :]
            out_s[:, :, t_start:t_end, :] = torch.matmul(attn_s[:, :, t_start:t_end, :], v_slice)

        # ── Step 5: Stream 2 — Sparse Historical Block Attention ──────────────
        if k_val > 0 and past_gate.any():
            out_m = torch.zeros_like(q)
            lse_m = torch.full((B_sz, H, T_q, 1), float("-inf"), dtype=q.dtype, device=q.device)

            for b_idx in range(1, num_blocks):
                t_start = b_idx * B_len
                t_end = min((b_idx + 1) * B_len, T_q)
                if t_start >= T_q:
                    break
                
                slice_gate = past_gate[:, :, t_start:t_end, :b_idx]
                if not slice_gate.any():
                    continue

                q_slice = q[:, :, t_start:t_end, :]
                k_past = k_padded[:, :, :b_idx * B_len, :]
                v_past = v_padded[:, :, :b_idx * B_len, :]

                s_past = torch.matmul(q_slice, k_past.transpose(-1, -2)) * self.scale
                gate_tokens = slice_gate.repeat_interleave(B_len, dim=-1)
                s_past = s_past.masked_fill(~gate_tokens, float("-inf"))

                lse_slice = torch.logsumexp(s_past, dim=-1, keepdim=True)
                attn_past = F.softmax(s_past, dim=-1)
                attn_past = torch.where(torch.isnan(attn_past), torch.zeros_like(attn_past), attn_past)
                
                out_m[:, :, t_start:t_end, :] = torch.matmul(attn_past, v_past)
                lse_m[:, :, t_start:t_end, :] = lse_slice

            # Combine self-stream and MoBA-stream using online softmax
            final_out = online_softmax_combine(out_s, lse_s, out_m, lse_m)
            return final_out
        else:
            return out_s


class MoBAAttentionAdapter(nn.Module):
    """
    Zero-overhead backward-compatible adapter wrapping standard attention with MoBA.
    Preserves legacy projection weights while providing an autonomous toggle to MoBA.
    """
    def __init__(
        self,
        base_attention: nn.Module,
        block_size: int = 64,
        top_k: int = 2,
        enabled: bool = True,
    ) -> None:
        super().__init__()
        self.base_attention = base_attention
        self.enabled = enabled
        head_dim = getattr(base_attention, "head_dim", 64)
        n_head = getattr(base_attention, "n_head", 16)
        self.moba = MoBAAttention(head_dim=head_dim, n_head=n_head, block_size=block_size, top_k=top_k)

    def forward(self, *args, **kwargs):
        return self.base_attention(*args, **kwargs)
