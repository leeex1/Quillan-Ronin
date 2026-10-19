#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Memory Attention: Token-Indexed Memory Replacing Dedicated Value Projections
(Jiale Kang, arXiv:2609.28399v1, 23 Sep 2026)

Implements the four foundational pillars of Memory Attention (MA):
1. Memory-Based Value Construction (Section 3.1 & Eq. 7-9):
   Eliminates the dense value projection matrix W_V.
   Constructs values via V = K + Norm(E[s]), where E is a learned layer-specific
   token memory table of size (N, d_v), s are input token IDs, and Norm is head-wise RMSNorm.
2. Inference Memory Folding (Section 3.3 & Eq. 16-17):
   Precomputes E_bar[i] = Norm(E[i]) for all vocabulary tokens i in [1..N].
   Reduces online value construction during inference to simple embedding lookup and
   addition (V = K + E_bar[s]), slashing value-construction arithmetic by Delta F_V = L * S * d_v * (2d - 1).
3. MA-Offload (Section 3.4 & Eq. 20-21):
   Offloads folded memory tables (P_offloaded = b_w * L * N * d_v bytes) to CPU memory / pin_memory,
   streaming only the necessary token slice for the current layer (D_offload = b_w * L * B * d_v bytes per step).
4. MA-Recall (Section 3.5 & Eq. 22-23):
   Eliminates persistent GPU storage of historical value caches (saving exactly 50% of content KV cache memory).
   Retains only historical keys K_1:T and token IDs s_1:T, reconstructing historical values
   on demand: V_1:T = K_1:T + E_bar[s_1:T].
"""

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class MAAnalysisReport:
    """Analytical comparison between Standard Attention and Memory Attention."""
    num_layers: int
    vocab_size: int
    hidden_dim: int
    value_dim: int
    standard_param_count: int
    ma_param_count: int
    online_flop_reduction: float
    offload_bytes_saved_gpu: int
    kv_cache_savings_ratio: float


class MemoryAttentionLayer(nn.Module):
    """
    Memory Attention Layer (Section 3.1 & 3.3).
    Replaces W_V with layer-specific learned token memory E[s].
    """

    def __init__(self,
                 hidden_dim: int = 64,
                 num_heads: int = 4,
                 vocab_size: int = 1000,
                 eps: float = 1e-5):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.vocab_size = vocab_size

        # Query and Key linear projections only; W_V is explicitly omitted
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # Layer-specific token memory table (N x d_v)
        self.token_memory = nn.Embedding(vocab_size, hidden_dim)

        # Per-head RMSNorm for memory tokens (Eq. 7)
        self.mem_norm = nn.RMSNorm(self.head_dim, eps=eps)

        # Pre-folded table buffer for O(1) zero-normalization inference (Eq. 16)
        self.register_buffer("_folded_memory", None, persistent=False)
        self.is_folded = False

    def fold_weights_for_inference(self) -> None:
        """
        Precomputes E_bar = Norm(E) across the full vocabulary table (Eq. 16).
        Enables online value construction with zero normalization FLOPs: V = K + E_bar[s].
        """
        with torch.no_grad():
            w = self.token_memory.weight.view(-1, self.num_heads, self.head_dim)
            normed_w = self.mem_norm(w).view(-1, self.hidden_dim)
            self._folded_memory = normed_w.contiguous()
            self.is_folded = True

    def unfold_weights(self) -> None:
        """Restores un-folded training mode."""
        self._folded_memory = None
        self.is_folded = False

    def construct_values(self, k: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Constructs attention values V from contextual keys K and token memory M.
        Training: V = K + Norm(E[s])
        Inference (folded): V = K + E_bar[s]
        """
        B, T, _ = k.shape
        if self.is_folded and self._folded_memory is not None:
            # Zero-normalization lookup and addition
            mem = F.embedding(token_ids, self._folded_memory)
        else:
            raw_mem = self.token_memory(token_ids)
            mem = self.mem_norm(raw_mem.view(B, T, self.num_heads, self.head_dim)).view(B, T, self.hidden_dim)

        return k + mem

    def forward(self,
                x: torch.Tensor,
                token_ids: torch.Tensor,
                is_causal: bool = True) -> torch.Tensor:
        B, T, C = x.shape
        q = self.q_proj(x)
        k = self.k_proj(x)

        # Construct values directly via MA formulation (Eq. 8)
        v = self.construct_values(k, token_ids)

        # Multi-head attention formatting
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if is_causal and T > 1:
            causal_mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device))
            scores = scores.masked_fill(~causal_mask, float("-inf"))

        weights = F.softmax(scores, dim=-1)
        out = torch.matmul(weights, v)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)


class MAOffloadManager:
    """
    MA-Offload Manager (Section 3.4 & Eq. 20-21).
    Maintains folded memory tables in CPU storage and manages on-demand GPU transfer.
    """

    def __init__(self,
                 num_layers: int,
                 vocab_size: int,
                 hidden_dim: int,
                 bytes_per_element: int = 2):  # FP16 / BF16
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.bytes_per_element = bytes_per_element

        # Host-resident CPU memory tables: L x N x d_v
        self.cpu_tables = torch.randn(num_layers, vocab_size, hidden_dim)
        self.total_offload_bytes = bytes_per_element * num_layers * vocab_size * hidden_dim
        self.total_transfers = 0
        self.total_bytes_transferred = 0

    def prefetch_layer_tokens(self,
                             layer_idx: int,
                             token_ids: torch.Tensor,
                             target_device: torch.device) -> torch.Tensor:
        """
        Transfers only the retrieved rows for the current layer to the GPU (Eq. 21).
        D_offload = b_w * S * d_v
        """
        token_ids_cpu = token_ids.detach().cpu()
        table = self.cpu_tables[layer_idx]
        selected_mem = F.embedding(token_ids_cpu, table).to(target_device)

        transfer_bytes = token_ids.numel() * self.hidden_dim * self.bytes_per_element
        self.total_transfers += 1
        self.total_bytes_transferred += transfer_bytes

        return selected_mem


class MARecallManager:
    """
    MA-Recall Manager (Section 3.5 & Eq. 22-23).
    Trades persistent value-cache storage for on-demand value reconstruction.
    Saves exactly 50% of persistent KV cache storage by retaining only K_1:T and token IDs s_1:T.
    """

    def __init__(self, bytes_per_element: int = 2):
        self.bytes_per_element = bytes_per_element
        self.reconstruction_count = 0

    def calculate_cache_savings(self,
                                num_layers: int,
                                batch_size: int,
                                seq_len: int,
                                hidden_dim: int) -> Dict[str, Any]:
        """
        Conventional KV cache: 2 * b_c * L * B * T * d_v bytes.
        MA-Recall cache: b_c * L * B * T * d_v bytes (50% reduction).
        """
        standard_bytes = 2 * self.bytes_per_element * num_layers * batch_size * seq_len * hidden_dim
        ma_recall_bytes = 1 * self.bytes_per_element * num_layers * batch_size * seq_len * hidden_dim
        return {
            "standard_cache_bytes": standard_bytes,
            "ma_recall_cache_bytes": ma_recall_bytes,
            "bytes_saved": standard_bytes - ma_recall_bytes,
            "reduction_ratio": 0.50
        }

    def reconstruct_historical_values(self,
                                      k_history: torch.Tensor,
                                      token_ids_history: torch.Tensor,
                                      folded_memory: torch.Tensor) -> torch.Tensor:
        """
        Reconstructs full value history on-the-fly: V_1:T = K_1:T + E_bar[s_1:T] (Eq. 22).
        """
        self.reconstruction_count += 1
        mem_history = F.embedding(token_ids_history, folded_memory)
        return k_history + mem_history


class MemoryAttentionEngine:
    """
    Analytical and Operational Engine for Memory Attention (arXiv:2609.28399v1).
    Computes theoretical FLOP reductions, parameter deltas, and validates computational invariants.
    """

    @staticmethod
    def analyze_computational_profile(num_layers: int,
                                      vocab_size: int,
                                      hidden_dim: int,
                                      batch_size: int,
                                      seq_len: int,
                                      bytes_per_element: int = 2) -> MAAnalysisReport:
        d = hidden_dim
        d_v = hidden_dim
        N = vocab_size
        L = num_layers
        S = batch_size * seq_len

        # Trainable parameters
        standard_params = L * d * d_v
        ma_params = L * (N * d_v + d)  # includes RMSNorm scale weights

        # Arithmetic FLOP reduction in value construction (Eq. 18): Delta F_V = L * S * d_v * (2d - 1)
        flop_reduction = float(L * S * d_v * (2 * d - 1))

        # Offloaded bytes
        offload_bytes = bytes_per_element * L * N * d_v

        return MAAnalysisReport(
            num_layers=L,
            vocab_size=N,
            hidden_dim=d,
            value_dim=d_v,
            standard_param_count=standard_params,
            ma_param_count=ma_params,
            online_flop_reduction=flop_reduction,
            offload_bytes_saved_gpu=offload_bytes,
            kv_cache_savings_ratio=0.50
        )
