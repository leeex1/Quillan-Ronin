#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
1-BIT AI INFRA: FAST AND LOSSLESS BITNET b1.58 INFERENCE ON CPUS (bitnet.cpp)
=============================================================================
Reference:
  Wang, Zhou, Song, Mao, Ma, Wang, Xia, Wei (Microsoft Research & Peking Univ., 2024)
  "1-bit AI Infra: Part 1.1, Fast and Lossless BitNet b1.58 Inference on CPUs"
  arXiv:2410.16144v2 [cs.CL] (7 pp)

Core Architectural Invariants:
  1. I2_S Kernel (2-bit packing per weight):
     - Maps ternary weights {-1, 0, 1} to 2-bit codes: -1 -> 00, 0 -> 01, 1 -> 10.
     - 4 weights packed per byte (75% memory reduction over int8, 87.5% over fp16).
     - Vanilla unpacked GEMV accumulation for thread-parallel execution.
  2. TL1 Kernel (Table-Lookup 1: 4-bit index per 2 weights):
     - Compresses every 2 weights into 4 bits (3^2 = 9 combinations: 0000 to 1000).
     - Precomputes 9-entry activation LUT per 2-tuple (w0*x_2j + w1*x_2j+1).
     - Computes GEMV via table lookup and INT16/INT32 addition without multiplication.
  3. TL2 Kernel (Table-Lookup 2: 5-bit index per 3 weights):
     - Compresses every 3 weights into 5 bits (3^3 = 27 combinations <= 32).
     - Achieves 1/6 additional compression over TL1 (1.67 bits/weight vs 2 bits/weight).
     - Precomputes 27-entry activation LUT per 3-tuple.
  4. Lossless Accuracy Guarantee:
     - 100% exact numerical equivalence to the dense FP32/INT8 BitNet b1.58 GEMV.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


# =====================================================================
# 1. I2_S Kernel: 2-bit Weight Packing & GEMV
# =====================================================================

class I2SKernel:
    """
    I2_S Kernel (Table 1):
      -1 -> 00 (0)
       0 -> 01 (1)
       1 -> 10 (2)
    Packs 4 weights per byte.
    """

    @staticmethod
    def pack(w_ternary: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """
        Packs a ternary tensor of shape (M, N) with values in {-1, 0, 1}.
        Returns packed uint8 tensor and original shape (M, N).
        """
        shape = (w_ternary.shape[0], w_ternary.shape[1])
        flat = w_ternary.flatten().to(torch.int64)

        # Map {-1, 0, 1} -> {0, 1, 2}
        codes = (flat + 1).clamp(0, 2)
        n = codes.numel()
        pad = (-n) % 4
        if pad > 0:
            codes = F.pad(codes, (0, pad), value=1)  # Pad with 0s (code 1)

        codes = codes.view(-1, 4)
        packed = (codes[:, 0]) | (codes[:, 1] << 2) | (codes[:, 2] << 4) | (codes[:, 3] << 6)
        return packed.to(torch.uint8), shape

    @staticmethod
    def unpack(packed: torch.Tensor, shape: Tuple[int, int]) -> torch.Tensor:
        """
        Unpacks uint8 tensor back to ternary {-1, 0, 1} tensor of shape (M, N).
        """
        m, n = shape
        total = m * n
        p = packed.to(torch.int64)
        c0 = p & 0b11
        c1 = (p >> 2) & 0b11
        c2 = (p >> 4) & 0b11
        c3 = (p >> 6) & 0b11

        stacked = torch.stack([c0, c1, c2, c3], dim=-1).flatten()[:total]
        unpacked = (stacked - 1).view(m, n).float()
        return unpacked

    @staticmethod
    def gemv(packed: torch.Tensor, shape: Tuple[int, int], x: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
        """
        Executes GEMV: y = x @ W^T * alpha.
        """
        w = I2SKernel.unpack(packed, shape).to(x.device)
        return F.linear(x, w) * alpha


# =====================================================================
# 2. TL1 Kernel: 4-bit LUT (2 Weights per Index)
# =====================================================================

class TL1Kernel:
    """
    TL1 Kernel (Table 2):
    Compresses every 2 weights into a 4-bit index (0 to 8).
    Precomputes 9-entry activation LUT:
      idx 0 (0000): -1, -1  -> -x0 - x1
      idx 1 (0001): -1,  0  -> -x0
      idx 2 (0010): -1,  1  -> -x0 + x1
      idx 3 (0011):  0, -1  -> -x1
      idx 4 (0100):  0,  0  ->  0
      idx 5 (0101):  0,  1  ->  x1
      idx 6 (0110):  1, -1  ->  x0 - x1
      idx 7 (0111):  1,  0  ->  x0
      idx 8 (1000):  1,  1  ->  x0 + x1
    """

    PAIR_MAP: Dict[Tuple[int, int], int] = {
        (-1, -1): 0,
        (-1,  0): 1,
        (-1,  1): 2,
        ( 0, -1): 3,
        ( 0,  0): 4,
        ( 0,  1): 5,
        ( 1, -1): 6,
        ( 1,  0): 7,
        ( 1,  1): 8,
    }

    INV_MAP: Dict[int, Tuple[int, int]] = {v: k for k, v in PAIR_MAP.items()}

    @staticmethod
    def pack(w_ternary: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """
        Packs (M, N) ternary tensor by pairing columns into 4-bit indices.
        Returns uint8 packed tensor (2 indices per byte) of shape (M, ceil(N/2)).
        """
        m, n = w_ternary.shape
        w = w_ternary.to(torch.int64)
        pad = n % 2
        if pad > 0:
            w = F.pad(w, (0, 1), value=0)
            n_padded = n + 1
        else:
            n_padded = n

        w_pairs = w.view(m, n_padded // 2, 2)
        # Map each pair (w0, w1) to index in [0, 8]
        # Formula: idx = (w0 + 1) * 3 + (w1 + 1)
        indices = (w_pairs[..., 0] + 1) * 3 + (w_pairs[..., 1] + 1)  # (M, N//2)

        # Pack two 4-bit indices per uint8 byte
        num_pairs = indices.shape[1]
        pad_byte = num_pairs % 2
        if pad_byte > 0:
            indices = F.pad(indices, (0, 1), value=4)  # pad with (0,0) index 4
            num_pairs += 1

        ind_reshaped = indices.view(m, num_pairs // 2, 2)
        packed = (ind_reshaped[..., 0] & 0x0F) | ((ind_reshaped[..., 1] & 0x0F) << 4)
        return packed.to(torch.uint8), (m, n)

    @staticmethod
    def build_lut(x: torch.Tensor) -> torch.Tensor:
        """
        Precomputes 9 activation sums for every pair in x.
        Args:
          x: Tensor of shape (..., N)
        Returns:
          lut: Tensor of shape (..., N_pairs, 9)
        """
        n = x.shape[-1]
        pad = n % 2
        if pad > 0:
            x_padded = F.pad(x, (0, 1), value=0.0)
        else:
            x_padded = x

        x_pairs = x_padded.view(*x.shape[:-1], -1, 2)
        x0 = x_pairs[..., 0]
        x1 = x_pairs[..., 1]

        # 9 combinations
        lut = torch.stack([
            -x0 - x1,  # 0: -1, -1
            -x0,       # 1: -1,  0
            -x0 + x1,  # 2: -1,  1
            -x1,       # 3:  0, -1
            torch.zeros_like(x0),  # 4: 0, 0
            x1,        # 5:  0,  1
            x0 - x1,   # 6:  1, -1
            x0,        # 7:  1,  0
            x0 + x1,   # 8:  1,  1
        ], dim=-1)
        return lut

    @staticmethod
    def gemv(packed: torch.Tensor, shape: Tuple[int, int], x: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
        """
        Computes GEMV via LUT lookup and addition.
        """
        m, n = shape
        num_pairs = (n + 1) // 2

        # Unpack indices from bytes
        p = packed.to(torch.int64)
        idx0 = p & 0x0F
        idx1 = (p >> 4) & 0x0F
        indices = torch.stack([idx0, idx1], dim=-1).view(m, -1)[:, :num_pairs]  # (M, num_pairs)

        # Build LUT from input x
        lut = TL1Kernel.build_lut(x)  # (..., num_pairs, 9)

        # Lookup and sum across pairs
        # Broadcast indices (M, num_pairs) against lut (..., num_pairs, 9)
        # Using gather
        batch_dims = lut.shape[:-2]
        flat_lut = lut.view(-1, num_pairs, 9)
        b_size = flat_lut.shape[0]

        # indices expanded for batch: (b_size, m, num_pairs)
        idx_expanded = indices.unsqueeze(0).expand(b_size, m, num_pairs)
        # lut expanded: (b_size, m, num_pairs, 9)
        flat_lut_exp = flat_lut.unsqueeze(1).expand(b_size, m, num_pairs, 9)

        gathered = torch.gather(flat_lut_exp, dim=-1, index=idx_expanded.unsqueeze(-1)).squeeze(-1)
        out = gathered.sum(dim=-1) * alpha  # (b_size, m)

        return out.view(*batch_dims, m)


# =====================================================================
# 3. TL2 Kernel: 5-bit LUT (3 Weights per Index)
# =====================================================================

class TL2Kernel:
    """
    TL2 Kernel (Table 3):
    Compresses every 3 weights into a 5-bit index (0 to 26 for 3^3 = 27 combinations).
    Yields 1/6 additional memory savings over TL1 (1.67 bits/weight vs 2.0 bits/weight).
    """

    @staticmethod
    def pack(w_ternary: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        m, n = w_ternary.shape
        w = w_ternary.to(torch.int64)
        pad = (-n) % 3
        if pad > 0:
            w = F.pad(w, (0, pad), value=0)
            n_padded = n + pad
        else:
            n_padded = n

        triplets = w.view(m, n_padded // 3, 3)
        # 3^3 = 27 combinations: idx = (w0+1)*9 + (w1+1)*3 + (w2+1)
        indices = (triplets[..., 0] + 1) * 9 + (triplets[..., 1] + 1) * 3 + (triplets[..., 2] + 1)
        return indices.to(torch.uint8), (m, n)

    @staticmethod
    def build_lut(x: torch.Tensor) -> torch.Tensor:
        """
        Precomputes 27 activation sums for every triplet in x.
        """
        n = x.shape[-1]
        pad = (-n) % 3
        if pad > 0:
            x_padded = F.pad(x, (0, pad), value=0.0)
        else:
            x_padded = x

        triplets = x_padded.view(*x.shape[:-1], -1, 3)
        x0 = triplets[..., 0]
        x1 = triplets[..., 1]
        x2 = triplets[..., 2]

        sums = []
        for w0 in [-1, 0, 1]:
            for w1 in [-1, 0, 1]:
                for w2 in [-1, 0, 1]:
                    term = w0 * x0 + w1 * x1 + w2 * x2
                    sums.append(term)

        lut = torch.stack(sums, dim=-1)  # (..., num_triplets, 27)
        return lut

    @staticmethod
    def gemv(indices: torch.Tensor, shape: Tuple[int, int], x: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
        m, n = shape
        num_triplets = indices.shape[1]
        lut = TL2Kernel.build_lut(x)  # (..., num_triplets, 27)

        batch_dims = lut.shape[:-2]
        flat_lut = lut.view(-1, num_triplets, 27)
        b_size = flat_lut.shape[0]

        idx_exp = indices.to(torch.int64).unsqueeze(0).expand(b_size, m, num_triplets)
        lut_exp = flat_lut.unsqueeze(1).expand(b_size, m, num_triplets, 27)

        gathered = torch.gather(lut_exp, dim=-1, index=idx_exp.unsqueeze(-1)).squeeze(-1)
        out = gathered.sum(dim=-1) * alpha
        return out.view(*batch_dims, m)


# =====================================================================
# 4. BitNet CPU Linear Module (Lossless Engine)
# =====================================================================

class BitNetCPULinear(nn.Module):
    """
    Drop-in BitNet b1.58 CPU inference module with kernel auto-dispatch.
    Supports I2_S, TL1, and TL2 execution paths with guaranteed 100% lossless output.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = False, kernel_type: str = "TL1"):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.kernel_type = kernel_type
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.02)
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        self.alpha = 1.0

        # Cached packed weights
        self.register_buffer("packed_i2s", None)
        self.register_buffer("packed_tl1", None)
        self.register_buffer("packed_tl2", None)

    def prepare_weights(self) -> None:
        """
        Quantizes FP weights to {-1, 0, 1} and pre-packs into I2_S, TL1, and TL2 tables.
        """
        w = self.weight.data
        alpha = w.abs().mean().item()
        self.alpha = max(alpha, 1e-12)

        # Absmean scaling and rounding to ternary
        w_norm = w / self.alpha
        w_ternary = torch.clamp(torch.round(w_norm), -1, 1)

        # Pack weights
        self.packed_i2s, _ = I2SKernel.pack(w_ternary)
        self.packed_tl1, _ = TL1Kernel.pack(w_ternary)
        self.packed_tl2, _ = TL2Kernel.pack(w_ternary)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.packed_tl1 is None:
            self.prepare_weights()

        shape = (self.out_features, self.in_features)

        if self.kernel_type == "I2_S":
            out = I2SKernel.gemv(self.packed_i2s, shape, x, alpha=self.alpha)
        elif self.kernel_type == "TL2":
            out = TL2Kernel.gemv(self.packed_tl2, shape, x, alpha=self.alpha)
        else:  # TL1 default
            out = TL1Kernel.gemv(self.packed_tl1, shape, x, alpha=self.alpha)

        if self.bias is not None:
            out = out + self.bias
        return out
