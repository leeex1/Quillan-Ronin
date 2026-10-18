"""
BitNet v2: Native 4-bit Activations with Hadamard Transformation for 1-bit LLMs
(arXiv:2504.18415v2, Hongyu Wang*, Shuming Ma*, Furu Wei - Microsoft Research & UCAS).

Mathematical Foundations & Invariants:
--------------------------------------
1. 1.58-bit Weights + Native 4-bit Activations:
   Replaces previous 8-bit activation bottlenecks in BitNet b1.58 and hybrid sparsification
   in BitNet a4.8, unlocking hardware-accelerated 4-bit tensor arithmetic (e.g. GB200 NVFP4 / INT4).

2. H-BitLinear with Online Hadamard Transformation (Eq. 3, 4, 7):
   Addresses severe activation outlier channels in intermediate projections (Wo in attention,
   Wdown in FFN). Applies an orthogonal Walsh-Hadamard transformation Hm prior to quantization:
     Hm = (1 / sqrt(2)) * [[H_{m-1},  H_{m-1}],
                           [H_{m-1}, -H_{m-1}]],  H0 = [1]
     Xr = Hadamard(LN(X))
   Orthogonality: Hm = Hm^T, Hm * Hm = I.
   Grad backprop (Eq. 8): dL/dX = Hadamard(dL/d(Hadamard(X))).
   Reduces activation kurtosis and peak outlier magnitude, reshaping sharp intermediate
   distributions into Gaussian-like distributions suitable for native INT4 quantization.

3. Ternary Weight Quantization (Eq. 1, 2):
   Per-tensor absmean quantization into {-1, 0, 1}:
     Qw(W) = alpha * RoundClip(W / (alpha + eps), -1, 1),  alpha = mean(|W|)

4. Dual Activation Quantization (Eq. 5, 6):
   - 8-bit Activations (Absmax):
       Q_INT8(X) = (gamma / 127) * RoundClip((127 / (gamma + eps)) * X, -128, 127), gamma = max(|X|)
   - 4-bit Activations (Absmean with sqrt(7) scale):
       Q_INT4(X) = (beta / sqrt(7)) * RoundClip((sqrt(7) / (beta + eps)) * X, -8, 7), beta = mean(|X|)

5. Low-Bit Attention & KV Cache (Section 3.1, Tables 2 & 3):
   Supports 4-bit and 3-bit KV cache with 8-bit BOS token protection.
"""

import math
from typing import Dict, Any, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


def fast_hadamard_transform(x: torch.Tensor) -> torch.Tensor:
    """
    Fast Walsh-Hadamard Transform (FWHT) along the last dimension.
    Satisfies Hm = (1 / sqrt(2)) * [[H_{m-1}, H_{m-1}], [H_{m-1}, -H_{m-1}]], H0 = 1.
    Symmetric and orthogonal: H = H^T and H * H = I (O(n log n) complexity).
    """
    orig_shape = x.shape
    n = orig_shape[-1]
    if (n & (n - 1)) != 0 or n <= 0:
        raise ValueError(f"Hadamard transform dimension {n} must be a power of 2.")

    res = x.clone().to(dtype=torch.float32)
    h = 1
    sqrt2_inv = 1.0 / math.sqrt(2.0)
    while h < n:
        res = res.view(-1, n // (2 * h), 2, h)
        u = res[:, :, 0, :].clone()
        v = res[:, :, 1, :].clone()
        res[:, :, 0, :] = (u + v) * sqrt2_inv
        res[:, :, 1, :] = (u - v) * sqrt2_inv
        h *= 2
    return res.view(orig_shape).to(dtype=x.dtype)


class HadamardTransformFunction(torch.autograd.Function):
    """
    Autograd function for online Hadamard transformation.
    Since Hm is orthogonal and symmetric, the backward pass is exact:
      dL/dX = Hadamard(dL/d(Hadamard(X))) (Paper Eq. 8).
    """
    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        return fast_hadamard_transform(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        return fast_hadamard_transform(grad_output)


def hadamard_transform(x: torch.Tensor) -> torch.Tensor:
    return HadamardTransformFunction.apply(x)


class WeightQuantTernarySTE(torch.autograd.Function):
    """
    Ternary weight quantization using absmean scaling:
      Qw(W) = alpha * RoundClip(W / (alpha + eps), -1, 1), alpha = mean(|W|)
    Straight-Through Estimator (STE) passes gradients directly to latent weights.
    """
    @staticmethod
    def forward(ctx, w: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
        alpha = torch.mean(torch.abs(w))
        scale = alpha + eps
        w_scaled = w / scale
        w_q = torch.clamp(torch.round(w_scaled), -1.0, 1.0)
        return w_q * alpha

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        return grad_output.clone(), None


def quantize_weight_ternary(w: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    return WeightQuantTernarySTE.apply(w, eps)


class ActQuantINT8STE(torch.autograd.Function):
    """
    8-bit activation quantization using per-token absmax (Paper Eq. 5):
      Q_INT8(X) = (gamma / 127) * RoundClip((127 / (gamma + eps)) * X, -128, 127)
      gamma = max(|X|) along last dimension.
    """
    @staticmethod
    def forward(ctx, x: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
        gamma = torch.amax(torch.abs(x), dim=-1, keepdim=True)
        scale = gamma + eps
        x_scaled = (127.0 / scale) * x
        x_q = torch.clamp(torch.round(x_scaled), -128.0, 127.0)
        return (scale / 127.0) * x_q

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        return grad_output.clone(), None


def quantize_act_int8(x: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    return ActQuantINT8STE.apply(x, eps)


class ActQuantINT4STE(torch.autograd.Function):
    """
    Native 4-bit activation quantization using per-token absmean (Paper Eq. 6):
      Q_INT4(X) = (beta / sqrt(7)) * RoundClip((sqrt(7) / (beta + eps)) * X, -8, 7)
      beta = mean(|X|) along last dimension.
    """
    @staticmethod
    def forward(ctx, x: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
        beta = torch.mean(torch.abs(x), dim=-1, keepdim=True)
        scale = beta + eps
        sqrt7 = math.sqrt(7.0)
        x_scaled = (sqrt7 / scale) * x
        x_q = torch.clamp(torch.round(x_scaled), -8.0, 7.0)
        return (scale / sqrt7) * x_q

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        return grad_output.clone(), None


def quantize_act_int4(x: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    return ActQuantINT4STE.apply(x, eps)


class BitLinearV2(nn.Module):
    """
    Standard BitLinear layer for inputs with natural Gaussian-like distributions
    (e.g. Wqkv, Wup, Wgate in BitNet v2).
    - RMSNorm on inputs
    - INT8 or INT4 activation quantization
    - Absmean ternary weights
    """
    def __init__(self, in_features: int, out_features: int, act_mode: str = "int4", eps: float = 1e-7):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.act_mode = act_mode.lower()
        self.eps = eps
        
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * (1.0 / math.sqrt(in_features)))
        self.norm = nn.RMSNorm(in_features, eps=eps)

    def set_activation_mode(self, mode: str):
        assert mode.lower() in ("int8", "int4", "fp16", "fp32")
        self.act_mode = mode.lower()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # RMSNorm
        x_norm = self.norm(x)
        
        # Activation Quantization
        if self.act_mode == "int8":
            x_q = quantize_act_int8(x_norm, self.eps)
        elif self.act_mode == "int4":
            x_q = quantize_act_int4(x_norm, self.eps)
        else:
            x_q = x_norm

        # Weight Quantization
        w_q = quantize_weight_ternary(self.weight, self.eps)
        return F.linear(x_q, w_q)


class HBitLinear(nn.Module):
    """
    H-BitLinear layer for intermediate representations with severe outlier channels
    (Wo in Attention, Wdown in FFN, Paper Eq. 7).
    - RMSNorm
    - Online Hadamard transformation: Xr = Hadamard(RMSNorm(X))
    - INT8 or INT4 activation quantization on Gaussianized representation
    - Absmean ternary weights
    """
    def __init__(self, in_features: int, out_features: int, act_mode: str = "int4", eps: float = 1e-7):
        super().__init__()
        if (in_features & (in_features - 1)) != 0 or in_features <= 0:
            raise ValueError(f"H-BitLinear in_features ({in_features}) must be a power of 2.")
            
        self.in_features = in_features
        self.out_features = out_features
        self.act_mode = act_mode.lower()
        self.eps = eps

        self.weight = nn.Parameter(torch.randn(out_features, in_features) * (1.0 / math.sqrt(in_features)))
        self.norm = nn.RMSNorm(in_features, eps=eps)

    def set_activation_mode(self, mode: str):
        assert mode.lower() in ("int8", "int4", "fp16", "fp32")
        self.act_mode = mode.lower()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Normalization
        x_norm = self.norm(x)
        
        # 2. Online Hadamard transformation (Eq. 3, 4, 7)
        x_hadamard = hadamard_transform(x_norm)
        
        # 3. Low-bit Activation Quantization (Eq. 5, 6)
        if self.act_mode == "int8":
            x_q = quantize_act_int8(x_hadamard, self.eps)
        elif self.act_mode == "int4":
            x_q = quantize_act_int4(x_hadamard, self.eps)
        else:
            x_q = x_hadamard

        # 4. Ternary Weight Quantization & Linear Projection
        w_q = quantize_weight_ternary(self.weight, self.eps)
        return F.linear(x_q, w_q)


class BitNetV2Attention(nn.Module):
    """
    Multi-Head Attention module for BitNet v2:
    - Wq, Wk, Wv projections using BitLinearV2 (no Hadamard needed, Gaussian inputs).
    - Wo projection using HBitLinear (Hadamard transform suppresses attention output outliers).
    - Post-RoPE low-bit KV Cache quantization (4-bit or 3-bit) with 8-bit BOS token retention.
    """
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        act_mode: str = "int4",
        kv_cache_bits: Optional[int] = 4,
        eps: float = 1e-7
    ):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.act_mode = act_mode
        self.kv_cache_bits = kv_cache_bits
        self.eps = eps

        self.q_proj = BitLinearV2(embed_dim, embed_dim, act_mode=act_mode, eps=eps)
        self.k_proj = BitLinearV2(embed_dim, embed_dim, act_mode=act_mode, eps=eps)
        self.v_proj = BitLinearV2(embed_dim, embed_dim, act_mode=act_mode, eps=eps)
        # Output projection MUST use H-BitLinear
        self.out_proj = HBitLinear(embed_dim, embed_dim, act_mode=act_mode, eps=eps)

    def set_activation_mode(self, mode: str):
        self.act_mode = mode
        self.q_proj.set_activation_mode(mode)
        self.k_proj.set_activation_mode(mode)
        self.v_proj.set_activation_mode(mode)
        self.out_proj.set_activation_mode(mode)

    def _quantize_kv_head(self, tensor: torch.Tensor, bits: int) -> torch.Tensor:
        """Quantizes key/value states with BOS token (index 0) preserved at 8-bit."""
        B, H, S, D = tensor.shape
        if S == 0 or bits is None or bits >= 16:
            return tensor

        # 8-bit for BOS (index 0)
        bos = tensor[:, :, :1, :]
        bos_q = quantize_act_int8(bos, self.eps)

        if S == 1:
            return bos_q

        # Remaining tokens quantized to target bits (e.g. 4-bit or 3-bit)
        rem = tensor[:, :, 1:, :]
        if bits == 4:
            rem_q = quantize_act_int4(rem, self.eps)
        elif bits == 3:
            # 3-bit quantization: [-4, 3] clip with per-token scale
            gamma = torch.amax(torch.abs(rem), dim=-1, keepdim=True) + self.eps
            scaled = (3.0 / gamma) * rem
            clipped = torch.clamp(torch.round(scaled), -4.0, 3.0)
            rem_q = (gamma / 3.0) * clipped
        else:
            rem_q = quantize_act_int8(rem, self.eps)

        return torch.cat([bos_q, rem_q], dim=2)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, S, D = x.shape
        q = self.q_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        # Quantize KV cache if specified (with 8-bit BOS token protection)
        if self.kv_cache_bits is not None:
            k = self._quantize_kv_head(k, self.kv_cache_bits)
            v = self._quantize_kv_head(v, self.kv_cache_bits)

        # Attention calculation
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask
        attn_weights = F.softmax(scores, dim=-1)
        context = torch.matmul(attn_weights, v)

        # Reshape context back to (B, S, D)
        context = context.transpose(1, 2).contiguous().view(B, S, D)
        out = self.out_proj(context)
        return out, attn_weights


class BitNetV2FFN(nn.Module):
    """
    SwiGLU Feed-Forward Network for BitNet v2:
    - Wgate and Wup use BitLinearV2 (Gaussian inputs).
    - Activation: Swish(x * Wgate) * (x * Wup).
    - Wdown uses HBitLinear (Hadamard transform suppresses SwiGLU outlier channels).
    """
    def __init__(
        self,
        embed_dim: int,
        intermediate_dim: int,
        act_mode: str = "int4",
        eps: float = 1e-7
    ):
        super().__init__()
        if (intermediate_dim & (intermediate_dim - 1)) != 0 or intermediate_dim <= 0:
            raise ValueError(f"intermediate_dim ({intermediate_dim}) must be a power of 2 for H-BitLinear.")
            
        self.embed_dim = embed_dim
        self.intermediate_dim = intermediate_dim
        self.act_mode = act_mode
        self.eps = eps

        self.gate_proj = BitLinearV2(embed_dim, intermediate_dim, act_mode=act_mode, eps=eps)
        self.up_proj = BitLinearV2(embed_dim, intermediate_dim, act_mode=act_mode, eps=eps)
        # Down projection MUST use H-BitLinear
        self.down_proj = HBitLinear(intermediate_dim, embed_dim, act_mode=act_mode, eps=eps)

    def set_activation_mode(self, mode: str):
        self.act_mode = mode
        self.gate_proj.set_activation_mode(mode)
        self.up_proj.set_activation_mode(mode)
        self.down_proj.set_activation_mode(mode)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        inter = gate * up
        return self.down_proj(inter)


class BitNetV2TransformerBlock(nn.Module):
    """
    BitNet v2 Transformer Layer with RMSNorm residual connections,
    H-BitLinear at Wo and Wdown, and native INT4/INT8 activation support.
    """
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        intermediate_dim: int,
        act_mode: str = "int4",
        kv_cache_bits: Optional[int] = 4,
        eps: float = 1e-7
    ):
        super().__init__()
        self.attn = BitNetV2Attention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            act_mode=act_mode,
            kv_cache_bits=kv_cache_bits,
            eps=eps
        )
        self.ffn = BitNetV2FFN(
            embed_dim=embed_dim,
            intermediate_dim=intermediate_dim,
            act_mode=act_mode,
            eps=eps
        )

    def set_activation_mode(self, mode: str):
        self.attn.set_activation_mode(mode)
        self.ffn.set_activation_mode(mode)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        attn_out, _ = self.attn(x, mask=mask)
        x = x + attn_out
        ffn_out = self.ffn(x)
        x = x + ffn_out
        return x


class BitNetV2Model(nn.Module):
    """
    Full BitNet v2 Model implementing LLaMA-like architecture with native 4-bit activations
    and online Hadamard transformation.
    """
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_heads: int,
        intermediate_dim: int,
        num_layers: int,
        act_mode: str = "int4",
        kv_cache_bits: Optional[int] = 4,
        eps: float = 1e-7
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.act_mode = act_mode

        self.tok_embeddings = nn.Embedding(vocab_size, embed_dim)
        self.layers = nn.ModuleList([
            BitNetV2TransformerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                intermediate_dim=intermediate_dim,
                act_mode=act_mode,
                kv_cache_bits=kv_cache_bits,
                eps=eps
            )
            for _ in range(num_layers)
        ])
        self.final_norm = nn.RMSNorm(embed_dim, eps=eps)
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)

    def set_activation_mode(self, mode: str):
        self.act_mode = mode
        for layer in self.layers:
            layer.set_activation_mode(mode)

    def forward(self, input_ids: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = self.tok_embeddings(input_ids)
        for layer in self.layers:
            h = layer(h, mask=mask)
        h = self.final_norm(h)
        return self.lm_head(h)


class OutlierDistributionProfiler:
    """
    Analytical profiler verifying Paper 39 invariants:
    - Kurtosis and maximum outlier attenuation under Hadamard transformation
    - Transition from sharp outlier distributions to Gaussian-like distributions
    """
    @staticmethod
    def compute_kurtosis(x: torch.Tensor) -> float:
        """Excess kurtosis of tensor (Gaussian has excess kurtosis = 0)."""
        x_flat = x.detach().flatten().float()
        mean = torch.mean(x_flat)
        var = torch.var(x_flat, unbiased=False)
        if var < 1e-12:
            return 0.0
        m4 = torch.mean((x_flat - mean) ** 4)
        return (m4 / (var ** 2) - 3.0).item()

    @staticmethod
    def compute_outlier_ratio(x: torch.Tensor, threshold_sigma: float = 3.0) -> float:
        """Fraction of elements exceeding threshold_sigma standard deviations from mean."""
        x_flat = x.detach().flatten().float()
        mean = torch.mean(x_flat)
        std = torch.std(x_flat)
        if std < 1e-12:
            return 0.0
        outliers = torch.abs(x_flat - mean) > (threshold_sigma * std)
        return (torch.sum(outliers).float() / x_flat.numel()).item()

    @staticmethod
    def analyze_hadamard_effect(x_intermediate: torch.Tensor) -> Dict[str, Any]:
        """
        Profiles activations before and after Hadamard transformation.
        Demonstrates reduction in peak activation and smoothing of outlier channels.
        """
        x_had = fast_hadamard_transform(x_intermediate)
        peak_before = torch.max(torch.abs(x_intermediate)).item()
        peak_after = torch.max(torch.abs(x_had)).item()
        kurt_before = OutlierDistributionProfiler.compute_kurtosis(x_intermediate)
        kurt_after = OutlierDistributionProfiler.compute_kurtosis(x_had)
        
        return {
            "peak_before": peak_before,
            "peak_after": peak_after,
            "peak_reduction_ratio": peak_before / (peak_after + 1e-12),
            "kurtosis_before": kurt_before,
            "kurtosis_after": kurt_after,
            "gaussianized": abs(kurt_after) < abs(kurt_before)
        }
