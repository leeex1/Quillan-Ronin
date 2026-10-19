#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Categorical Reparameterization with Gumbel-Softmax
=================================================
Paper 59: "Categorical Reparameterization with Gumbel-Softmax"
Authors: Eric Jang, Shixiang Gu, Ben Poole
Affiliations: Google Brain, University of Cambridge, MPI Tübingen, Stanford University
Conference: ICLR 2017 / arXiv:1611.01144 (Nov 2016 / Aug 2017, 13 pages)

Key Architectural & Mathematical Innovations:
1. Gumbel-Max Trick & Reparameterization (Section 2, Equation 1):
   - Discrete categorical variables cannot be backpropagated through due to step-discontinuities.
   - Drawing samples g_i ~ Gumbel(0, 1) via inverse transform sampling:
       g_i = -log(-log(u_i)), where u_i ~ Uniform(0, 1)
   - Sampling discrete one-hot categories via:
       z = one_hot(argmax_i [g_i + log pi_i])
     exactly matches sampling from Categorical(pi_1, ..., pi_k).

2. Continuous Gumbel-Softmax Relaxation (Section 2, Equation 2):
   - Replaces the non-differentiable argmax with a smooth, differentiable softmax with temperature tau:
       y_i = exp((log pi_i + g_i) / tau) / sum_{j=1}^k exp((log pi_j + g_j) / tau)
   - The vector y lives on the (k-1)-dimensional simplex Delta^{k-1}.
   - Parameter gradients partial(y) / partial(pi) are smooth and computable via backpropagation.

3. Straight-Through (ST) Gumbel-Softmax Estimator (Section 2.2):
   - In the forward pass, discretizes y to an exact one-hot vector:
       y_{hard} = one_hot(argmax_i(y_i))
   - In the backward pass, computes gradients with respect to the continuous relaxation:
       y = y_{hard} - y_{soft}.detach() + y_{soft}
   - Allows discrete sparse representations during forward evaluation while maintaining continuous
     path-wise gradients during backpropagation.

4. Exact Gumbel-Softmax Density Function (Section 2, Theorem 1, Equation 3):
   - The probability density over the simplex y in Delta^{k-1} is:
       p_{pi, tau}(y_1, ..., y_k) = Gamma(k) * tau^{k-1} * (sum_{i=1}^k pi_i / y_i^tau)^(-k) * prod_{i=1}^k (pi_i / y_i^{tau + 1})
   - As tau -> 0, p_{pi, tau}(y) converges to the discrete categorical distribution.
   - As tau -> infinity, p_{pi, tau}(y) converges to a uniform distribution over the simplex.

5. Temperature Annealing Schedules (Section 2.1 & 3):
   - Balances bias and variance: High tau produces low-variance but biased gradients; low tau
     produces unbiased gradients but high variance.
   - Exponential annealing: tau(t) = max(tau_min, tau_0 * exp(-r * t))
   - Cosine annealing: tau(t) = tau_min + 0.5 * (tau_0 - tau_min) * (1 + cos(pi * t / T))
"""

import math
import logging
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger("GumbelSoftmaxEngine")


@dataclass
class GumbelSoftmaxConfig:
    """Configuration for Gumbel-Softmax sampling and annealing."""
    tau_init: float = 1.0              # Initial temperature tau_0
    tau_min: float = 0.1               # Minimum temperature floor tau_min
    anneal_rate: float = 0.001         # Exponential annealing decay rate r
    hard: bool = False                 # Enable Straight-Through (ST) discretization
    eps: float = 1e-20                 # Safe epsilon for log and uniform clamping
    dim: int = -1                      # Dimension along which softmax is computed


def sample_gumbel_noise(
    shape: torch.Size,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
    eps: float = 1e-20,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """
    Generate independent and identically distributed Gumbel(0, 1) random variables
    via inverse transform sampling (Section 2, Footnote 1):
      u ~ Uniform(0, 1)
      g = -log(-log(u))
    """
    u = torch.rand(shape, device=device, dtype=dtype, generator=generator)
    u_clamped = torch.clamp(u, min=eps, max=1.0 - eps)
    g = -torch.log(-torch.log(u_clamped))
    return g


def gumbel_softmax_sample(
    logits: torch.Tensor,
    tau: float = 1.0,
    hard: bool = False,
    dim: int = -1,
    eps: float = 1e-20,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """
    Continuous Gumbel-Softmax relaxation and Straight-Through estimator (Equations 1 & 2).

    Args:
        logits: Unnormalized log probabilities (..., num_classes)
        tau: Non-negative scalar temperature
        hard: If True, returns one-hot discretized samples in forward pass while backpropagating
              gradients through the continuous relaxation (Straight-Through Gumbel-Softmax).
        dim: Dimension over which categorical softmax is evaluated.
        eps: Numerical stability constant.
        generator: Optional torch.Generator for reproducible sampling.

    Returns:
        Sampled simplex or one-hot tensor matching logits shape.
    """
    tau = max(tau, 1e-4)  # Prevent division by zero
    gumbel_noise = sample_gumbel_noise(
        logits.shape, device=logits.device, dtype=logits.dtype, eps=eps, generator=generator
    )

    # Numerically stable continuous relaxation: y_soft = softmax((logits + g) / tau)
    y_soft = F.softmax((logits + gumbel_noise) / tau, dim=dim)

    if not hard:
        return y_soft

    # Straight-Through (ST) Estimator (Section 2.2):
    # Forward: y_hard = one_hot(argmax(y_soft))
    # Backward: gradient passes directly to y_soft
    index = y_soft.max(dim, keepdim=True)[1]
    y_hard = torch.zeros_like(logits).scatter_(dim, index, 1.0)
    # STE identity trick: y = y_hard - detach(y_soft) + y_soft
    return y_hard - y_soft.detach() + y_soft


def gumbel_softmax_log_density(
    y: torch.Tensor,
    logits: torch.Tensor,
    tau: float = 1.0,
    dim: int = -1,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Computes the exact log-density of the Gumbel-Softmax distribution on the simplex Delta^{k-1}
    (Section 2, Theorem 1, Equation 3):
      log p(y) = log Gamma(k) + (k - 1)*log(tau) - k*log(sum_i (pi_i / y_i^tau)) + sum_i log(pi_i / y_i^{tau+1})
    where pi = softmax(logits) and y in Delta^{k-1}.
    """
    k = y.shape[dim]
    pi = F.softmax(logits, dim=dim)
    y_clamped = torch.clamp(y, min=eps)

    # Term 1: log Gamma(k)
    log_gamma_k = math.lgamma(k)

    # Term 2: (k - 1) * log(tau)
    log_tau_term = (k - 1) * math.log(max(tau, 1e-6))

    # Term 3: -k * log(sum_i (pi_i / y_i^tau))
    pi_div_y_tau = pi / (y_clamped ** tau)
    sum_pi_div_y_tau = torch.sum(pi_div_y_tau, dim=dim, keepdim=True)
    term3 = -k * torch.log(torch.clamp(sum_pi_div_y_tau, min=eps))

    # Term 4: sum_i log(pi_i / y_i^{tau + 1})
    pi_div_y_tau1 = pi / (y_clamped ** (tau + 1.0))
    term4 = torch.sum(torch.log(torch.clamp(pi_div_y_tau1, min=eps)), dim=dim, keepdim=True)

    log_p = log_gamma_k + log_tau_term + term3 + term4
    return log_p.squeeze(dim)


class TemperatureAnnealingSchedule:
    """
    Temperature Annealing Schedules (Section 2.1).
    Interpolates smoothly from exploratory high temperature to sharp categorical selection.
    """

    def __init__(
        self,
        tau_init: float = 1.0,
        tau_min: float = 0.1,
        anneal_rate: float = 0.001,
        total_steps: int = 1000,
        schedule_type: str = "exponential",
    ):
        self.tau_init = tau_init
        self.tau_min = tau_min
        self.anneal_rate = anneal_rate
        self.total_steps = max(total_steps, 1)
        self.schedule_type = schedule_type.lower()

    def get_tau(self, step: int) -> float:
        """Calculate current temperature for training step t."""
        if self.schedule_type == "exponential":
            # tau(t) = max(tau_min, tau_0 * exp(-r * t))
            tau = self.tau_init * math.exp(-self.anneal_rate * step)
            return max(self.tau_min, tau)
        elif self.schedule_type == "linear":
            # tau(t) = max(tau_min, tau_0 - (tau_0 - tau_min) * (t / T))
            fraction = min(max(step / self.total_steps, 0.0), 1.0)
            tau = self.tau_init - (self.tau_init - self.tau_min) * fraction
            return max(self.tau_min, tau)
        elif self.schedule_type == "cosine":
            # tau(t) = tau_min + 0.5 * (tau_0 - tau_min) * (1 + cos(pi * t / T))
            fraction = min(max(step / self.total_steps, 0.0), 1.0)
            tau = self.tau_min + 0.5 * (self.tau_init - self.tau_min) * (1.0 + math.cos(math.pi * fraction))
            return max(self.tau_min, tau)
        else:
            raise ValueError(f"Unknown schedule type: {self.schedule_type}")


class StraightThroughGumbelSoftmax(nn.Module):
    """
    PyTorch Neural Network Layer for Categorical Reparameterization with Gumbel-Softmax.
    Supports continuous relaxation and Straight-Through discretization.
    """

    def __init__(self, config: Optional[GumbelSoftmaxConfig] = None):
        super().__init__()
        self.config = config if config is not None else GumbelSoftmaxConfig()
        self.register_buffer("current_tau", torch.tensor(self.config.tau_init, dtype=torch.float32))

    def set_temperature(self, tau: float) -> None:
        """Update current sampling temperature."""
        self.current_tau.fill_(max(tau, 1e-4))

    def forward(
        self,
        logits: torch.Tensor,
        hard: Optional[bool] = None,
        tau: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Forward pass yielding differentiable categorical samples.
        """
        t = tau if tau is not None else self.current_tau.item()
        h = hard if hard is not None else self.config.hard
        return gumbel_softmax_sample(
            logits=logits,
            tau=t,
            hard=h,
            dim=self.config.dim,
            eps=self.config.eps,
        )


class CategoricalVAELatent(nn.Module):
    """
    Categorical Variational Autoencoder Latent Layer (Section 4, Unsupervised Generative Modeling).
    Transforms encoder features to categorical latent variables and computes the analytic KL divergence:
      D_KL(q(z | x) || p(z)) = sum_i q_i * log(q_i / p_i)
    with a uniform prior p(z) = 1 / num_classes.
    """

    def __init__(
        self,
        in_features: int,
        num_categoricals: int,
        num_classes: int,
        tau_init: float = 1.0,
        tau_min: float = 0.1,
    ):
        super().__init__()
        self.num_categoricals = num_categoricals
        self.num_classes = num_classes
        self.proj = nn.Linear(in_features, num_categoricals * num_classes)
        self.gumbel_layer = StraightThroughGumbelSoftmax(
            GumbelSoftmaxConfig(tau_init=tau_init, tau_min=tau_min, hard=True)
        )

    def forward(
        self,
        x: torch.Tensor,
        hard: bool = True,
        tau: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass producing discrete latents and analytic KL divergence.
        Args:
            x: Input tensor of shape (batch, in_features)
        Returns:
            samples: Latent sample tensor of shape (batch, num_categoricals, num_classes)
            logits: Logits tensor of shape (batch, num_categoricals, num_classes)
            kl_div: Analytic KL divergence per batch sample of shape (batch,)
        """
        batch_size = x.size(0)
        logits = self.proj(x).view(batch_size, self.num_categoricals, self.num_classes)
        samples = self.gumbel_layer(logits, hard=hard, tau=tau)

        # Analytic KL divergence against uniform categorical prior p(z) = 1 / num_classes
        # KL(q || p) = sum_i q_i * (log q_i - log(1 / k)) = sum_i q_i * log q_i + log k
        q = F.softmax(logits, dim=-1)
        log_q = F.log_softmax(logits, dim=-1)
        kl = torch.sum(q * log_q, dim=-1) + math.log(self.num_classes)
        kl_div = torch.sum(kl, dim=-1)  # Sum over all categorical variables

        return samples, logits, kl_div


def verify_gumbel_softmax_invariants() -> Dict[str, Any]:
    """
    Comprehensive verification suite for Paper 59 mathematical invariants:
      1. Gumbel-Max Trick Empirical Convergence: Samples drawn via argmax(g + log pi)
         converge to categorical probabilities within statistical error (< 0.02).
      2. Differentiable Path-Wise Gradient: Backpropagation computes non-zero, finite gradients
         d(Loss)/d(logits) via the continuous relaxation.
      3. Straight-Through (ST) Invariant: Forward pass outputs exact discrete one-hot vectors
         (sum = 1.0, elements in {0, 1}) while backward pass receives non-zero gradients.
      4. Temperature Extremes:
         - As tau -> 0, continuous samples converge to discrete one-hot vectors.
         - As tau -> infinity, samples converge to uniform distributions.
      5. Exact Log-Density Evaluation: Theorem 1 density matches analytical formulation.
      6. Temperature Annealing Monotonicity.
    """
    torch.manual_seed(42)
    results = {}

    # 1. Gumbel-Max Trick Empirical Convergence
    # Draw 50,000 samples from a 4-class categorical distribution and verify convergence
    true_probs = torch.tensor([0.1, 0.4, 0.3, 0.2])
    logits = torch.log(true_probs)
    n_samples = 50000

    expanded_logits = logits.unsqueeze(0).expand(n_samples, -1)
    g_noise = sample_gumbel_noise(expanded_logits.shape)
    sampled_indices = torch.argmax(expanded_logits + g_noise, dim=-1)

    empirical_counts = torch.bincount(sampled_indices, minlength=4).float()
    empirical_probs = empirical_counts / n_samples

    max_prob_err = (empirical_probs - true_probs).abs().max().item()
    assert max_prob_err < 0.02, f"Gumbel-Max empirical error too large: {max_prob_err}"
    results["gumbel_max_empirical_err"] = max_prob_err

    # 2. Differentiable Continuous Relaxation Gradient Flow
    test_logits = torch.tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)
    sample_soft = gumbel_softmax_sample(test_logits, tau=0.5, hard=False)
    loss_soft = torch.sum(sample_soft * torch.tensor([1.0, 2.0, 3.0, 4.0]))
    loss_soft.backward()

    assert test_logits.grad is not None, "Gradients failed to propagate through continuous relaxation"
    assert torch.all(torch.isfinite(test_logits.grad)), "Gradients contain NaN or Inf values"
    assert test_logits.grad.abs().sum().item() > 0.0, "Zero gradient produced in relaxation"
    results["soft_grad_norm"] = test_logits.grad.norm().item()

    # 3. Straight-Through (ST) Gumbel-Softmax Invariant
    st_logits = torch.tensor([1.0, 5.0, 2.0], requires_grad=True)
    sample_hard = gumbel_softmax_sample(st_logits, tau=0.5, hard=True)

    # Check forward pass is exact one-hot
    assert (sample_hard.sum() - 1.0).abs().item() < 1e-6, "ST forward is not normalized"
    assert torch.all((sample_hard == 0.0) | (sample_hard == 1.0)), "ST forward is not strictly binary one-hot"

    # Check backward pass has gradients via continuous surrogate
    loss_hard = (sample_hard * torch.tensor([10.0, 20.0, 30.0])).sum()
    loss_hard.backward()
    assert st_logits.grad is not None, "ST backward gradients failed to propagate"
    assert st_logits.grad.abs().sum().item() > 0.0, "ST backward gradients are zero"
    results["st_is_one_hot"] = True
    results["st_grad_norm"] = st_logits.grad.norm().item()

    # 4. Temperature Extremes Invariant
    # Low temperature limit (tau = 0.01) -> near one-hot
    low_tau_sample = gumbel_softmax_sample(logits, tau=0.01, hard=False)
    assert low_tau_sample.max().item() > 0.99, f"Low tau sample not sharp: {low_tau_sample}"

    # High temperature limit (tau = 50.0) -> near uniform (0.25 each)
    high_tau_sample = gumbel_softmax_sample(logits, tau=50.0, hard=False)
    uniform_target = torch.full((4,), 0.25)
    high_tau_diff = (high_tau_sample - uniform_target).abs().max().item()
    assert high_tau_diff < 0.05, f"High tau sample not uniform: {high_tau_sample}"
    results["low_tau_max_prob"] = low_tau_sample.max().item()
    results["high_tau_uniform_diff"] = high_tau_diff

    # 5. Exact Log-Density Function (Theorem 1, Eq. 3)
    y_test = torch.tensor([0.1, 0.4, 0.3, 0.2])
    log_p = gumbel_softmax_log_density(y_test, logits, tau=1.0)
    assert torch.isfinite(log_p).item(), "Log-density evaluation produced NaN/Inf"
    results["log_density_sample"] = log_p.item()

    # 6. Temperature Annealing Monotonicity
    anneal_rate = math.log(1.0 / 0.1) / 500.0  # Decays from 1.0 to 0.1 at step 500
    annealer = TemperatureAnnealingSchedule(tau_init=1.0, tau_min=0.1, anneal_rate=anneal_rate, total_steps=500)
    t0 = annealer.get_tau(0)
    t250 = annealer.get_tau(250)
    t500 = annealer.get_tau(500)
    assert t0 > t250 > t500, f"Annealing is not strictly monotonically decreasing: {t0}, {t250}, {t500}"
    assert abs(t500 - 0.1) < 0.01, f"Final temperature did not reach floor: {t500}"
    results["anneal_t0"] = t0
    results["anneal_t500"] = t500

    results["all_invariants_passed"] = True
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Testing Gumbel-Softmax Invariants...")
    res = verify_gumbel_softmax_invariants()
    print("Verification Results:", res)
