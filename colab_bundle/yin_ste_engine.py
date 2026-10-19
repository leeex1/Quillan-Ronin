"""
Understanding Straight-Through Estimator in Training Activation Quantized Neural Nets
====================================================================================
Mathematical & Computational Invariant Implementation based on:
Yin, Lyu, Zhang, Osher, Qi, and Xin (ICLR 2019 / arXiv:1903.05662)

Key Invariants & Formulations Implemented:
1. Two-Linear-Layer Binary-Activated Network (Section 2, Eq. 1):
   - Model prediction: y(Z, v, w) = v^T sigma(Z w) = sum_{i=1}^m v_i 1_{Z_i^T w > 0}.
   - Ground truth label: y*(Z) = (v*)^T sigma(Z w*).
   - Sample squared loss: l(v, w; Z) = 0.5 * (y(Z, v, w) - y*(Z))^2.
2. Coarse Gradient via STE-Modified Chain Rule (Section 2.1, Eq. 5):
   - grad_v = sigma(Z w) * (y - y*)
   - grad_w coarse = Z^T (mu'(Z w) * v) * (y - y*)
   - Evaluates three canonical STE surrogate derivatives mu'(x):
     a) Identity STE: mu'(x) = 1 (Hinton / Rosenblatt)
     b) ReLU STE: mu'(x) = 1_{x > 0} (Hubara / Cai)
     c) Clipped ReLU STE: mu'(x) = 1_{0 < x < 1} (Yin / Cai)
3. Analytical Population Loss & True Gradient (Section 2.2, Lemmas 1 & 2, Eqs. 6-7):
   - Evaluates exact closed-form expectation over standard Gaussian Z ~ N(0, I):
     theta = arccos(w^T w* / (||w|| ||w*||))
     f(v, w) = 1/8 [ v^T (I_m + 1 1^T) v - 2 v^T ((1 - 2/pi theta) I_m + 1 1^T) v* + (v*)^T (I_m + 1 1^T) v* ]
     grad_w f(v, w) = - (v^T v* / (2 pi ||w||)) * (P_w w* / ||P_w w*||) where P_w = I_n - (w w^T / ||w||^2).
4. Expected Coarse Gradient & Positive Correlation Invariant (Section 3, Lemmas 5, 7, 10):
   - Proves <E[g_mu], grad_w f> >= 0 for valid STEs (ReLU & Clipped ReLU).
   - For ReLU STE: <E[g_relu], grad_w f> = (sin(theta) / (2 (2 pi)^{3/2} ||w||)) * (v^T v*)^2 >= 0.
   - For Identity STE: E[g_id] does NOT vanish at spurious local minimizers (theta=pi), causing instability.
5. General Multi-Bit Uniform Quantization STE Layer:
   - Uniform b-bit activation quantization with selectable STE gradient gating.
"""

import math
from typing import Dict, Optional, Tuple, Literal
import torch
import torch.nn as nn
import torch.nn.functional as F


STEType = Literal["identity", "relu", "clipped_relu"]


class BinarySTEFunction(torch.autograd.Function):
    """
    Autograd function implementing discrete binary step activation sigma(x) = 1_{x > 0}
    with configurable Straight-Through Estimator (STE) in the backward pass.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, ste_type: str = "clipped_relu") -> torch.Tensor:
        ctx.save_for_backward(x)
        ctx.ste_type = ste_type
        # Discrete binary forward pass: 1 if x > 0 else 0
        return (x > 0.0).to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        (x,) = ctx.saved_tensors
        ste_type = ctx.ste_type

        if ste_type == "identity":
            # Identity STE: mu'(x) = 1
            surrogate_grad = torch.ones_like(x)
        elif ste_type == "relu":
            # ReLU STE: mu'(x) = 1_{x > 0}
            surrogate_grad = (x > 0.0).to(x.dtype)
        elif ste_type == "clipped_relu":
            # Clipped ReLU STE: mu'(x) = 1_{0 < x < 1}
            surrogate_grad = ((x > 0.0) & (x < 1.0)).to(x.dtype)
        else:
            raise ValueError(f"Unknown ste_type: {ste_type}")

        grad_input = grad_output * surrogate_grad
        return grad_input, None


def binary_ste(x: torch.Tensor, ste_type: STEType = "clipped_relu") -> torch.Tensor:
    """Convenience functional wrapper for BinarySTEFunction."""
    return BinarySTEFunction.apply(x, ste_type)


class MultiBitSTEFunction(torch.autograd.Function):
    """
    Uniform b-bit activation quantization with STE backward gating:
    Q(x) = round((2^b - 1) * clip(x, 0, 1)) / (2^b - 1)
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, num_bits: int = 2, ste_type: str = "clipped_relu") -> torch.Tensor:
        ctx.save_for_backward(x)
        ctx.ste_type = ste_type
        num_levels = (1 << num_bits) - 1
        x_clipped = torch.clamp(x, 0.0, 1.0)
        q = torch.round(x_clipped * num_levels) / num_levels
        return q

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None, None]:
        (x,) = ctx.saved_tensors
        ste_type = ctx.ste_type

        if ste_type == "identity":
            surrogate_grad = torch.ones_like(x)
        elif ste_type == "relu":
            surrogate_grad = (x > 0.0).to(x.dtype)
        elif ste_type == "clipped_relu":
            surrogate_grad = ((x >= 0.0) & (x <= 1.0)).to(x.dtype)
        else:
            raise ValueError(f"Unknown ste_type: {ste_type}")

        grad_input = grad_output * surrogate_grad
        return grad_input, None, None


def multibit_ste(x: torch.Tensor, num_bits: int = 2, ste_type: STEType = "clipped_relu") -> torch.Tensor:
    """Convenience functional wrapper for MultiBitSTEFunction."""
    return MultiBitSTEFunction.apply(x, num_bits, ste_type)


class TwoLayerBinaryCNN(nn.Module):
    """
    Two-Linear-Layer Convolutional Network with Binary Activation (Section 2, Eq. 1):
    y(Z, v, w) = v^T sigma(Z w)
    where Z in R^{m x n}, w in R^n, v in R^m, and sigma is binary step function.
    """

    def __init__(self, m: int, n: int, ste_type: STEType = "clipped_relu"):
        super().__init__()
        self.m = m
        self.n = n
        self.ste_type = ste_type

        # First layer filter w: [n]
        self.w = nn.Parameter(torch.randn(n))
        # Second layer classifier v: [m]
        self.v = nn.Parameter(torch.randn(m))

    def forward(self, Z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            Z: Input batch of matrices of shape [..., m, n].
        Returns:
            Scalar predictions of shape [...] matching batch dimensions.
        """
        # Z w: shape [..., m]
        Zw = torch.matmul(Z, self.w)
        # Binary activation via STE: shape [..., m]
        acts = binary_ste(Zw, ste_type=self.ste_type)
        # v^T acts: scalar output
        out = torch.matmul(acts, self.v)
        return out


class YinSTETheoreticalAnalytic:
    """
    Exact closed-form mathematical formulas from Yin et al. (ICLR 2019)
    for Gaussian inputs Z ~ N(0, I_n) (Lemmas 1, 2, 4, 5, 9, 10).
    """

    @staticmethod
    def angle_theta(w: torch.Tensor, w_star: torch.Tensor) -> float:
        """Computes theta(w, w*) = arccos(w^T w* / (||w|| ||w*||)) in [0, pi]."""
        dot = torch.dot(w, w_star).item()
        norm_w = torch.norm(w).item()
        norm_star = torch.norm(w_star).item()
        cos_theta = max(-1.0, min(1.0, dot / (norm_w * norm_star + 1e-12)))
        return math.acos(cos_theta)

    @staticmethod
    def population_loss(v: torch.Tensor, w: torch.Tensor, v_star: torch.Tensor, w_star: torch.Tensor) -> float:
        """
        Analytical Population Loss f(v, w) (Lemma 1, Section 2.2):
        f(v, w) = 1/8 [ v^T (I_m + 1 1^T) v - 2 v^T ((1 - 2/pi theta) I_m + 1 1^T) v* + (v*)^T (I_m + 1 1^T) v* ]
        """
        m = v.size(0)
        theta = YinSTETheoreticalAnalytic.angle_theta(w, w_star)
        I_m = torch.eye(m, dtype=v.dtype, device=v.device)
        ones_mat = torch.ones(m, m, dtype=v.dtype, device=v.device)
        M_all = I_m + ones_mat

        term1 = torch.matmul(v, torch.matmul(M_all, v))
        M_theta = (1.0 - (2.0 / math.pi) * theta) * I_m + ones_mat
        term2 = 2.0 * torch.matmul(v, torch.matmul(M_theta, v_star))
        term3 = torch.matmul(v_star, torch.matmul(M_all, v_star))

        return (1.0 / 8.0) * (term1 - term2 + term3).item()

    @staticmethod
    def population_gradient_v(v: torch.Tensor, w: torch.Tensor, v_star: torch.Tensor, w_star: torch.Tensor) -> torch.Tensor:
        """
        Analytical Partial Gradient grad_v f(v, w) (Lemma 2, Eq. 6):
        grad_v f = 1/4 (I_m + 1 1^T) v - 1/4 ((1 - 2/pi theta) I_m + 1 1^T) v*
        """
        m = v.size(0)
        theta = YinSTETheoreticalAnalytic.angle_theta(w, w_star)
        I_m = torch.eye(m, dtype=v.dtype, device=v.device)
        ones_mat = torch.ones(m, m, dtype=v.dtype, device=v.device)
        M_all = I_m + ones_mat
        M_theta = (1.0 - (2.0 / math.pi) * theta) * I_m + ones_mat

        grad_v = 0.25 * (torch.matmul(M_all, v) - torch.matmul(M_theta, v_star))
        return grad_v

    @staticmethod
    def population_gradient_w(v: torch.Tensor, w: torch.Tensor, v_star: torch.Tensor, w_star: torch.Tensor) -> torch.Tensor:
        """
        Analytical Partial Gradient grad_w f(v, w) (Lemma 2, Eq. 7):
        grad_w f = - (v^T v* / (2 pi ||w||)) * (P_w w* / ||P_w w*||)
        where P_w = I_n - (w w^T / ||w||^2) is orthogonal projection onto w^perp.
        """
        norm_w = torch.norm(w)
        dot_v = torch.dot(v, v_star)
        # Projection of w* onto orthogonal complement of w:
        w_unit = w / norm_w
        proj_w_star = w_star - torch.dot(w_star, w_unit) * w_unit
        norm_proj = torch.norm(proj_w_star)

        if norm_proj < 1e-9:
            return torch.zeros_like(w)

        direction = proj_w_star / norm_proj
        coeff = - dot_v / (2.0 * math.pi * norm_w)
        return coeff * direction

    @staticmethod
    def expected_coarse_gradient_relu(v: torch.Tensor, w: torch.Tensor, v_star: torch.Tensor, w_star: torch.Tensor) -> torch.Tensor:
        """
        Analytical Expected Coarse Gradient E_Z[g_relu(v, w; Z)] (Lemma 4, Eq. 11):
        E[g_relu] = (h(v, v*) / (2 sqrt(2 pi))) * (w / ||w||) - (cos(theta/2) (v^T v*) / sqrt(2 pi)) * ((w/||w|| + w*) / ||w/||w|| + w*||)
        where h(v, v*) = ||v||^2 + (1^T v)^2 - (1^T v)(1^T v*) + v^T v*.
        """
        norm_w = torch.norm(w)
        theta = YinSTETheoreticalAnalytic.angle_theta(w, w_star)
        dot_v = torch.dot(v, v_star)
        sum_v = torch.sum(v)
        sum_v_star = torch.sum(v_star)
        norm_v_sq = torch.sum(v * v)

        h_val = norm_v_sq + (sum_v ** 2) - sum_v * sum_v_star + dot_v

        term1 = (h_val / (2.0 * math.sqrt(2.0 * math.pi))) * (w / norm_w)

        w_hat = w / norm_w
        sum_vec = w_hat + w_star
        norm_sum = torch.norm(sum_vec)

        if norm_sum < 1e-9 or theta >= math.pi - 1e-5:
            term2 = torch.zeros_like(w)
        else:
            term2 = (math.cos(theta / 2.0) * dot_v / math.sqrt(2.0 * math.pi)) * (sum_vec / norm_sum)

        return term1 - term2

    @staticmethod
    def expected_coarse_gradient_identity(v: torch.Tensor, w: torch.Tensor, v_star: torch.Tensor, w_star: torch.Tensor) -> torch.Tensor:
        """
        Analytical Expected Coarse Gradient E_Z[g_id(v, w; Z)] (Lemma 9, Eq. 14):
        E[g_id] = (1 / sqrt(2 pi)) * ( ||v||^2 (w / ||w||) - (v^T v*) w* )
        """
        norm_w = torch.norm(w)
        norm_v_sq = torch.sum(v * v)
        dot_v = torch.dot(v, v_star)

        return (1.0 / math.sqrt(2.0 * math.pi)) * (norm_v_sq * (w / norm_w) - dot_v * w_star)

    @staticmethod
    def relu_inner_product_theoretical(v: torch.Tensor, w: torch.Tensor, v_star: torch.Tensor, w_star: torch.Tensor) -> float:
        """
        Analytical Inner Product <E[g_relu], grad_w f> (Lemma 5, Section 3.1):
        <E[g_relu], grad_w f> = (sin(theta) / (2 (2 pi)^{3/2} ||w||)) * (v^T v*)^2 >= 0
        """
        theta = YinSTETheoreticalAnalytic.angle_theta(w, w_star)
        norm_w = torch.norm(w).item()
        dot_v = torch.dot(v, v_star).item()
        return (math.sin(theta) / (2.0 * ((2.0 * math.pi) ** 1.5) * norm_w)) * (dot_v ** 2)
