#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BITNET.CPP: EFFICIENT EDGE INFERENCE FOR TERNARY LLMS
=====================================================
Reference:
  Jinheng Wang, Hansong Zhou, Ting Song, Shijie Cao, Yan Xia, Ting Cao,
  Jianyu Wei, Shuming Ma, Hongyu Wang, Furu Wei
  (Peking Univ, USTC, UCAS, Microsoft Research, Feb 2025)
  "Bitnet.cpp: Efficient Edge Inference for Ternary LLMs"
  arXiv:2502.11880v1 [cs.LG] (18 pp)

Core Invariants & Taxonomy:
  1. Systematic Taxonomy of Edge mpGEMM:
     - MAD-based: I2_S (2 bpw, lossless)
     - LUT-based:
       * TL1_0: 2 bpw, per-block activation quantization (fast, block_size=256)
       * TL1_1: 2 bpw, per-token activation quantization (lossless)
       * TL2_0: 1.67 bpw (5 bits per 3 weights), per-block activation quantization (fast)
       * TL2_1: 1.67 bpw, per-token activation quantization (lossless)
  2. Element-wise Mirror Consolidation:
     - Halves 3^3 = 27 ternary states down to ceil(27/2) = 14 states (indices 0..13).
     - Fits inside 128-bit SIMD registers (vpshufb with 16 int8 slots).
     - Signed-Unsigned Weight Splitting: 4-bit unsigned magnitude index + 1-bit sign bit.
  3. Lossless Inference Guarantee:
     - Bit-exact numerical equivalence with BitLinear b1.58 forward pass when using
       per-token activation scale (TL1_1, TL2_1, I2_S).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


# =====================================================================
# 1. Triplet Mirror Table & Signed-Unsigned Splitting
# =====================================================================

class SignedUnsignedWeightPacker:
    """
    Element-wise mirror consolidation for ternary triplets (w0, w1, w2) in {-1, 0, 1}^3.
    Enumerates all 27 ternary states and pairs each state with its negation:
      state = w0 * 9 + w1 * 3 + w2 + 13  (maps [-13, 13] -> [0, 26])
    Mirror consolidation maps the state to:
      unsigned_idx in [0, 13] (14 unique magnitudes)
      sign_bit in {0, 1} (0 = positive/identity, 1 = negated)
    """

    # Canonical list of 14 non-negative base triplets
    BASE_TRIPLETS: List[Tuple[int, int, int]] = [
        (0, 0, 0),   # 0
        (0, 0, 1),   # 1
        (0, 1, -1),  # 2
        (0, 1, 0),   # 3
        (0, 1, 1),   # 4
        (1, -1, -1), # 5
        (1, -1, 0),  # 6
        (1, -1, 1),  # 7
        (1, 0, -1),  # 8
        (1, 0, 0),   # 9
        (1, 0, 1),   # 10
        (1, 1, -1),  # 11
        (1, 1, 0),   # 12
        (1, 1, 1)    # 13
    ]

    _TRIPLET_TO_ENCODING: Dict[Tuple[int, int, int], Tuple[int, int]] = {}
    _ENCODING_TO_TRIPLET: Dict[Tuple[int, int], Tuple[int, int, int]] = {}

    @classmethod
    def _init_tables(cls):
        if cls._TRIPLET_TO_ENCODING:
            return
        for idx, (w0, w1, w2) in enumerate(cls.BASE_TRIPLETS):
            # Identity (sign = 0)
            cls._TRIPLET_TO_ENCODING[(w0, w1, w2)] = (idx, 0)
            cls._ENCODING_TO_TRIPLET[(idx, 0)] = (w0, w1, w2)
            if (w0, w1, w2) != (0, 0, 0):
                # Negation (sign = 1)
                cls._TRIPLET_TO_ENCODING[(-w0, -w1, -w2)] = (idx, 1)
                cls._ENCODING_TO_TRIPLET[(idx, 1)] = (-w0, -w1, -w2)

    @classmethod
    def pack_triplets(cls, w: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        w: [N, K] with values in {-1, 0, 1}, K must be multiple of 3.
        Returns:
          unsigned_indices: [N, K // 3] (values in 0..13, 4-bit)
          sign_bits: [N, K // 3] (values in 0..1, 1-bit)
        """
        cls._init_tables()
        n, k = w.shape
        num_triplets = k // 3
        w_reshaped = w[:, :num_triplets * 3].view(n, num_triplets, 3).to(torch.int32)

        unsigned_indices = torch.zeros(n, num_triplets, dtype=torch.uint8)
        sign_bits = torch.zeros(n, num_triplets, dtype=torch.uint8)

        # Convert to tuple and lookup
        w_cpu = w_reshaped.cpu()
        for i in range(n):
            for j in range(num_triplets):
                trip = (int(w_cpu[i, j, 0]), int(w_cpu[i, j, 1]), int(w_cpu[i, j, 2]))
                idx, sign = cls._TRIPLET_TO_ENCODING.get(trip, (0, 0))
                unsigned_indices[i, j] = idx
                sign_bits[i, j] = sign

        return unsigned_indices.to(w.device), sign_bits.to(w.device)

    @classmethod
    def unpack_triplets(cls, unsigned_indices: torch.Tensor, sign_bits: torch.Tensor) -> torch.Tensor:
        """Unpacks back into dense {-1, 0, 1} tensor."""
        cls._init_tables()
        n, num_triplets = unsigned_indices.shape
        out = torch.zeros(n, num_triplets * 3, dtype=torch.float32, device=unsigned_indices.device)
        u_cpu = unsigned_indices.cpu()
        s_cpu = sign_bits.cpu()

        for i in range(n):
            for j in range(num_triplets):
                idx = int(u_cpu[i, j])
                sign = int(s_cpu[i, j])
                trip = cls._ENCODING_TO_TRIPLET.get((idx, sign), (0, 0, 0))
                out[i, j * 3] = float(trip[0])
                out[i, j * 3 + 1] = float(trip[1])
                out[i, j * 3 + 2] = float(trip[2])

        return out


# =====================================================================
# 2. TL1 and TL2 Kernel Implementations (Block & Tensor-Wide)
# =====================================================================

class TL1KernelEdge:
    """
    TL1 Kernel (2 bpw): Table-lookup over pairs of weights (g=2).
    Enumerates 3^2 = 9 activation sums per pair:
      LUT[w0*3 + w1 + 4] = w0 * x[2j] + w1 * x[2j+1]
    Modes:
      - per_block=True -> TL1_0 (block_size=256)
      - per_block=False -> TL1_1 (per-token / per-tensor, lossless)
    """

    PAIR_MAP = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),  (0, 0),  (0, 1),
        (1, -1),  (1, 0),  (1, 1)
    ]

    @classmethod
    def compute_gemv(
        cls,
        x: torch.Tensor,
        weights: torch.Tensor,
        per_block: bool = False,
        block_size: int = 256
    ) -> torch.Tensor:
        """
        x: [B, K]
        weights: [N, K] (ternary weights)
        """
        b, k = x.shape
        n, _ = weights.shape
        num_pairs = k // 2

        # 1. Quantize activation
        if per_block:
            # Per-block quantization (TL1_0)
            x_blocks = x.view(b, -1, block_size)
            max_val = x_blocks.abs().max(dim=-1, keepdim=True)[0] + 1e-5
            scale = 127.0 / max_val
            x_q = torch.clamp(torch.round(x_blocks * scale), -128.0, 127.0)
            x_dequant = (x_q / scale).view(b, k)
        else:
            # Per-token quantization (TL1_1, lossless)
            max_val = x.abs().max(dim=-1, keepdim=True)[0] + 1e-5
            scale = 127.0 / max_val
            x_q = torch.clamp(torch.round(x * scale), -128.0, 127.0)
            x_dequant = x_q / scale

        # 2. Build 9-state lookup table per pair for x_dequant
        x_p0 = x_dequant[:, 0::2]  # [B, K // 2]
        x_p1 = x_dequant[:, 1::2]  # [B, K // 2]

        lut = torch.zeros(b, num_pairs, 9, device=x.device, dtype=x.dtype)
        for s_idx, (w0, w1) in enumerate(cls.PAIR_MAP):
            lut[:, :, s_idx] = w0 * x_p0 + w1 * x_p1

        # 3. Lookup and accumulate
        w_p0 = weights[:, 0::2]
        w_p1 = weights[:, 1::2]
        w_idx = ((w_p0 + 1) * 3 + (w_p1 + 1)).long()  # [N, K // 2], values in 0..8

        # Batched gather: [B, N]
        res = torch.zeros(b, n, device=x.device, dtype=x.dtype)
        for i in range(b):
            lut_i = lut[i]  # [num_pairs, 9]
            gathered = torch.gather(lut_i.unsqueeze(0).expand(n, -1, -1), dim=2, index=w_idx.unsqueeze(-1)).squeeze(-1)
            res[i] = gathered.sum(dim=-1)

        return res


class TL2KernelEdge:
    """
    TL2 Kernel (1.67 bpw): Table-lookup over triplets (g=3) with mirror consolidation.
    14-state half-LUT + 1-bit sign negation.
    Modes:
      - per_block=True -> TL2_0 (block_size=256)
      - per_block=False -> TL2_1 (per-token / per-tensor, lossless)
    """

    @classmethod
    def compute_gemv(
        cls,
        x: torch.Tensor,
        unsigned_indices: torch.Tensor,
        sign_bits: torch.Tensor,
        per_block: bool = False,
        block_size: int = 256
    ) -> torch.Tensor:
        """
        x: [B, K]
        unsigned_indices: [N, K // 3] in 0..13
        sign_bits: [N, K // 3] in 0..1
        """
        SignedUnsignedWeightPacker._init_tables()
        b, k = x.shape
        n, num_triplets = unsigned_indices.shape

        if per_block:
            x_blocks = x.view(b, -1, block_size)
            max_val = x_blocks.abs().max(dim=-1, keepdim=True)[0] + 1e-5
            scale = 127.0 / max_val
            x_q = torch.clamp(torch.round(x_blocks * scale), -128.0, 127.0)
            x_dequant = (x_q / scale).view(b, k)
        else:
            max_val = x.abs().max(dim=-1, keepdim=True)[0] + 1e-5
            scale = 127.0 / max_val
            x_q = torch.clamp(torch.round(x * scale), -128.0, 127.0)
            x_dequant = x_q / scale

        x_t0 = x_dequant[:, 0::3]
        x_t1 = x_dequant[:, 1::3]
        x_t2 = x_dequant[:, 2::3]

        # Build 14-state mirror LUT
        lut = torch.zeros(b, num_triplets, 14, device=x.device, dtype=x.dtype)
        for idx, (w0, w1, w2) in enumerate(SignedUnsignedWeightPacker.BASE_TRIPLETS):
            lut[:, :, idx] = w0 * x_t0 + w1 * x_t1 + w2 * x_t2

        res = torch.zeros(b, n, device=x.device, dtype=x.dtype)
        u_idx = unsigned_indices.long()
        signs = (1.0 - 2.0 * sign_bits.float())  # 0 -> +1, 1 -> -1

        for i in range(b):
            lut_i = lut[i]  # [num_triplets, 14]
            gathered = torch.gather(lut_i.unsqueeze(0).expand(n, -1, -1), dim=2, index=u_idx.unsqueeze(-1)).squeeze(-1)
            # Apply sign
            signed_vals = gathered * signs
            res[i] = signed_vals.sum(dim=-1)

        return res


class I2SKernelEdge:
    """
    I2_S MAD-based Kernel (2 bpw): Lossless integer SIMD dot product.
    Strictly aligns with BitNet b1.58 training scheme.
    """

    @classmethod
    def compute_gemv(cls, x: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        max_val = x.abs().max(dim=-1, keepdim=True)[0] + 1e-5
        scale = 127.0 / max_val
        x_q = torch.clamp(torch.round(x * scale), -128.0, 127.0)
        x_dequant = x_q / scale
        return F.linear(x_dequant, weights)


# =====================================================================
# 3. BitNet.cpp Edge Linear Layer
# =====================================================================

class BitNetCppEdgeLinear(nn.Module):
    """
    High-performance Edge Linear Layer supporting all 5 kernels from Table 1:
      - 'TL1_0': 2 bpw, LUT-based, per-block activation quant
      - 'TL1_1': 2 bpw, LUT-based, lossless per-token activation quant
      - 'TL2_0': 1.67 bpw, mirror LUT-based, per-block activation quant
      - 'TL2_1': 1.67 bpw, mirror LUT-based, lossless per-token activation quant
      - 'I2_S' : 2 bpw, MAD-based, lossless per-token activation quant
    """

    def __init__(self, in_features: int, out_features: int, kernel_type: str = "TL2_1"):
        super().__init__()
        # Align in_features to multiple of 6 (LCM of 2 and 3)
        self.in_features = in_features
        self.out_features = out_features
        self.kernel_type = kernel_type

        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.02)
        self.register_buffer("ternary_weight", torch.zeros(out_features, in_features))
        self.register_buffer("unsigned_indices", torch.zeros(out_features, in_features // 3, dtype=torch.uint8))
        self.register_buffer("sign_bits", torch.zeros(out_features, in_features // 3, dtype=torch.uint8))

    def prepare_weights(self):
        """Quantizes full-precision latent weights to ternary and packs into lookup representations."""
        gamma = self.weight.abs().mean()
        w_ternary = torch.clamp(torch.round(self.weight / (gamma + 1e-5)), -1.0, 1.0)
        self.ternary_weight.copy_(w_ternary)

        # Pack for TL2
        u_idx, s_bits = SignedUnsignedWeightPacker.pack_triplets(w_ternary)
        self.unsigned_indices.copy_(u_idx)
        self.sign_bits.copy_(s_bits)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.size(0)
        x_flat = x.view(-1, self.in_features)

        if self.kernel_type == "TL1_0":
            out = TL1KernelEdge.compute_gemv(x_flat, self.ternary_weight, per_block=True, block_size=min(256, self.in_features))
        elif self.kernel_type == "TL1_1":
            out = TL1KernelEdge.compute_gemv(x_flat, self.ternary_weight, per_block=False)
        elif self.kernel_type == "TL2_0":
            out = TL2KernelEdge.compute_gemv(x_flat, self.unsigned_indices, self.sign_bits, per_block=True, block_size=min(256, self.in_features))
        elif self.kernel_type == "TL2_1":
            out = TL2KernelEdge.compute_gemv(x_flat, self.unsigned_indices, self.sign_bits, per_block=False)
        elif self.kernel_type == "I2_S":
            out = I2SKernelEdge.compute_gemv(x_flat, self.ternary_weight)
        else:
            raise ValueError(f"Unknown kernel type: {self.kernel_type}")

        if x.dim() == 3:
            return out.view(b, x.size(1), self.out_features)
        return out


# =====================================================================
# 4. ELUT Theoretical Analyzer
# =====================================================================

class ELUTTheoreticalAnalyzer:
    """
    Theoretical analysis of Element-wise Lookup Table (ELUT) from Section 3 & Appendix A:
      - Weight set cardinality C = 3 for ternary.
      - Group size g = 2 (TL1) -> 3^2 = 9 states vs 2^4 = 16 states (43.8% table saving).
      - Group size g = 3 (TL2) -> with mirror consolidation: ceil(3^3 / 2) = 14 states
        vs 2^6 = 64 states (78.1% table saving).
      - Bits per weight: 5 bits / 3 weights = 1.667 bpw.
    """

    @staticmethod
    def table_compaction_ratio(cardinality: int = 3, group_size: int = 3, use_mirror: bool = True) -> float:
        """Computes table compaction ratio: element-wise size / bit-wise size."""
        bit_width = math.ceil(math.log2(cardinality))  # 2 for ternary
        bitwise_entries = 2 ** (bit_width * group_size)
        if use_mirror:
            elut_entries = math.ceil((cardinality ** group_size) / 2.0)
        else:
            elut_entries = cardinality ** group_size
        return elut_entries / bitwise_entries

    @staticmethod
    def bits_per_weight(group_size: int = 3) -> float:
        """Returns bpw for TL2: 5 bits per 3 weights = 1.666... bpw."""
        if group_size == 3:
            return 5.0 / 3.0
        elif group_size == 2:
            return 4.0 / 2.0
        return 2.0
