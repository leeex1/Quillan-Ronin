#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ES AWD Engine: Anchored Weight Decay for Continual Learning in Evolution Strategies
===================================================================================
Paper 54: "Overcoming Forgetting in LLM Fine-Tuning with Evolution Strategies"
Authors: Kajetan Schweighofer, Conor F. Hayes, Roberto Dailey, Risto Miikkulainen, Xin Qiu
Affiliations: Cognizant AI Lab, UT Austin
arXiv: 2605.30148v1 (28 May 2026, 23 pages)

Key Architectural & Mathematical Innovations:
1. Root Cause of ES Forgetting - Weakly Constrained Random Walk (Section 4.3, Eq. 2):
   - ES updates decompose into signal + isotropic noise.
   - The expected cumulative squared parameter drift from random walk along task-unconstrained directions is:
     E[||theta_T - theta_0||_2^2] = (alpha^2 * T * d) / N
     where alpha is learning rate, T is iterations, d is parameter count, N is population size.
   - Explains the massive Frobenius norm explosion observed in Paper 53 (Abdi et al.).

2. Anchored Regularization Objective (Section 5, Eq. 3):
   - Augments the population fitness objective with an element-wise distance penalty from initial weights theta_0:
     max_theta E_{eps ~ N(0, I)} [R(theta + sigma * eps)] - lambda * sum_{i=1}^d l(theta - theta_0)_i
   - Uniform weighting over all parameters reflects generalist prior task preservation without needing
     task-specific Fisher information (EWC).

3. Anchored Weight Decay (AWD) Update Rule (Section 5, Eq. 4 & Algorithm 1):
   - Extends the decomposed ES update with parameter-space shrinkage toward theta_0:
     tilde_theta_t = theta_{t-1} + alpha * (1 / N) * sum_{n=1}^N Z_n * eps_n
     theta_t = tilde_theta_t - alpha * lambda * l'(tilde_theta_t - theta_0)
   - Penalty Variants:
     * L2 Penalty: l(x) = 0.5 * x^2 => l'(x) = x
       theta_t = (1 - alpha * lambda) * tilde_theta_t + (alpha * lambda) * theta_0
     * L1 Penalty (Proximal Step / Soft-Thresholding):
       l'(x) = min(alpha * lambda, |x|) * sign(x)
       theta_t = theta_0 + sign(Delta) * max(0, |Delta| - alpha * lambda)
       where Delta = tilde_theta_t - theta_0.
       Directly zeroes out perturbations smaller than alpha * lambda, restoring ~95% parameter sparsity.

4. Streaming Layer-Wise Host Anchor:
   - Preserves GPU VRAM by storing reference weights theta_0 in pinned CPU host memory and streaming
     them on demand during layer-wise weight updates.

This module is part of the Quillan-Ronin Sovereign AI Architecture.
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Callable, Any

import torch
import torch.nn as nn


@dataclass
class AWDConfig:
    """Configuration for Evolution Strategies with Anchored Weight Decay."""
    population_size: int = 30           # Population size N (default 30 from Table 1 & 2)
    sigma: float = 0.001                # Perturbation noise scale sigma
    learning_rate: float = 0.0005       # Learning rate alpha
    penalty_type: str = "l2"            # "l1" or "l2"
    lambda_reg: float = 1.0             # Regularization penalty factor lambda (e.g. 10.0 for L2, 0.01 for L1)
    seed_base: int = 42                 # Base seed for pseudo-random perturbation generation
    stream_from_cpu: bool = True        # Keep base anchor in pinned host memory to save GPU VRAM


def compute_expected_random_walk_drift(
    step: int,
    num_params: int,
    learning_rate: float,
    population_size: int
) -> float:
    """
    Computes theoretical expected cumulative Frobenius drift E[||theta_T - theta_0||_F]
    induced by isotropic random walk (Paper 54 Eq. 2):
    sqrt(E[||theta_T - theta_0||_2^2]) = sqrt((alpha^2 * T * d) / N)
    """
    if population_size <= 0:
        raise ValueError("Population size N must be positive")
    variance = (learning_rate ** 2) * step * num_params / population_size
    return math.sqrt(variance)


def apply_anchored_decay_step(
    current_param: torch.Tensor,
    base_param: torch.Tensor,
    alpha: float,
    lambda_reg: float,
    penalty_type: str = "l2"
) -> torch.Tensor:
    """
    Applies Anchored Weight Decay (AWD) toward base parameter anchor theta_0 (Paper 54 Eq. 4).

    For L2 penalty (smooth shrinkage):
      theta_t = tilde_theta - alpha * lambda * (tilde_theta - theta_0)
              = (1 - alpha * lambda) * tilde_theta + (alpha * lambda) * theta_0

    For L1 penalty (proximal soft-thresholding operator):
      Delta = tilde_theta - theta_0
      theta_t = theta_0 + sign(Delta) * max(0, |Delta| - alpha * lambda)
    """
    threshold = alpha * lambda_reg
    delta = current_param - base_param

    if penalty_type.lower() == "l2":
        # Smooth contraction toward theta_0
        decay_factor = max(0.0, 1.0 - threshold)
        updated = base_param + decay_factor * delta
        return updated
    elif penalty_type.lower() == "l1":
        # Proximal operator / Soft-thresholding: sets delta exactly to zero if |delta| <= threshold
        abs_delta = torch.abs(delta)
        shrinkage = torch.clamp(abs_delta - threshold, min=0.0)
        updated = base_param + torch.sign(delta) * shrinkage
        return updated
    else:
        raise ValueError(f"Unknown penalty_type: {penalty_type}. Expected 'l1' or 'l2'")


class ESAWDOptimizer:
    """
    Evolution Strategies Optimizer integrated with Anchored Weight Decay (Algorithm 1).
    Performs full-parameter population perturbations and applies AWD regularization
    to mitigate catastrophic forgetting.
    """

    def __init__(
        self,
        model: nn.Module,
        config: Optional[AWDConfig] = None
    ) -> None:
        self.model = model
        self.config = config or AWDConfig()
        self.step_count = 0

        # Store baseline anchor theta_0 in pinned CPU memory to avoid VRAM overhead (Section 5)
        self.base_anchor: Dict[str, torch.Tensor] = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                tensor_copy = param.detach().clone()
                if self.config.stream_from_cpu:
                    tensor_copy = tensor_copy.cpu().pin_memory()
                self.base_anchor[name] = tensor_copy

    def step(self, fitness_fn: Callable[[nn.Module], float]) -> Dict[str, Any]:
        """
        Executes one full iteration of Algorithm 1:
        1. Sample perturbations eps_n ~ N(0, I) for n=1..N
        2. Evaluate perturbed models on target task: r_n <- R(theta + sigma * eps_n)
        3. Z-score reward normalization: z_n <- (r_n - mean) / (std + eps)
        4. Calculate standard ES update: tilde_theta <- theta + alpha * (1/N) * sum z_n * eps_n
        5. Apply Anchored Weight Decay (AWD): theta <- tilde_theta - alpha * lambda * l'(tilde_theta - theta_0)
        """
        self.step_count += 1
        cfg = self.config
        pop_size = cfg.population_size
        alpha = cfg.learning_rate
        sigma = cfg.sigma

        # Collect named trainable parameters
        trainable_params = [(name, p) for name, p in self.model.named_parameters() if p.requires_grad]

        # 1. Sample perturbations and evaluate population rewards
        iter_seed = cfg.seed_base + self.step_count * 10007
        rewards = torch.zeros(pop_size)
        population_epsilons: List[Dict[str, torch.Tensor]] = []

        for n in range(pop_size):
            member_seed = iter_seed + n
            torch.manual_seed(member_seed)
            eps_dict: Dict[str, torch.Tensor] = {}

            # Apply perturbation in-place layer-by-layer
            with torch.no_grad():
                for name, p in trainable_params:
                    eps = torch.randn_like(p)
                    eps_dict[name] = eps
                    p.add_(eps, alpha=sigma)

            # Evaluate fitness
            rewards[n] = float(fitness_fn(self.model))

            # Restore original parameters in-place
            with torch.no_grad():
                for name, p in trainable_params:
                    p.sub_(eps_dict[name], alpha=sigma)

            population_epsilons.append(eps_dict)

        # 2. Z-score reward normalization (Paper 52 & Paper 54 Algorithm 1 Step 6)
        r_mean = rewards.mean()
        r_std = rewards.std() + 1e-8
        z_rewards = (rewards - r_mean) / r_std

        # 3. Calculate ES update & 4. Apply Anchored Weight Decay (Steps 7 & 8)
        total_sq_norm = 0.0
        total_elements = 0
        total_zero_deltas = 0

        with torch.no_grad():
            for name, p in trainable_params:
                # Accumulate population update: (1/N) * sum z_n * eps_n
                accum_grad = torch.zeros_like(p)
                for n in range(pop_size):
                    accum_grad.add_(population_epsilons[n][name], alpha=float(z_rewards[n].item()))
                accum_grad.div_(pop_size)

                # Standard ES update: tilde_theta = theta + alpha * update
                p.add_(accum_grad, alpha=alpha)

                # AWD Step toward base anchor theta_0
                base_p = self.base_anchor[name].to(p.device)
                p.copy_(apply_anchored_decay_step(
                    current_param=p,
                    base_param=base_p,
                    alpha=alpha,
                    lambda_reg=cfg.lambda_reg,
                    penalty_type=cfg.penalty_type
                ))

                delta = p - base_p
                total_sq_norm += float((delta ** 2).sum().item())
                total_elements += delta.numel()
                total_zero_deltas += int((torch.abs(delta) < 1e-6).sum().item())

        frob_drift = math.sqrt(total_sq_norm)
        sparsity_pct = (total_zero_deltas / total_elements * 100.0) if total_elements > 0 else 100.0

        return {
            "step": self.step_count,
            "mean_reward": float(r_mean.item()),
            "frobenius_drift": frob_drift,
            "sparsity_pct": sparsity_pct,
            "penalty_type": cfg.penalty_type,
            "lambda_reg": cfg.lambda_reg
        }


def verify_es_awd_invariants() -> Dict[str, Any]:
    """
    Self-verification function testing all core mathematical invariants of Paper 54:
    1. Theoretical random walk drift scaling equation: E[||theta_T - theta_0||_2^2] = (alpha^2 * T * d) / N.
    2. L2 AWD smooth parameter contraction toward theta_0.
    3. L1 AWD proximal soft-thresholding producing exact zeros for perturbations below threshold.
    4. Controlled optimization benchmark: ES + AWD successfully optimizes target task while
       substantially reducing parameter drift compared to vanilla unanchored ES.
    """
    # Invariant 1: Random walk scaling equation
    # T=100, d=1000, alpha=0.01, N=10 => variance = (1e-4 * 100 * 1000) / 10 = 1.0 => drift = 1.0
    drift_est = compute_expected_random_walk_drift(step=100, num_params=1000, learning_rate=0.01, population_size=10)
    assert abs(drift_est - 1.0) < 1e-6, f"Expected drift 1.0, got {drift_est}"

    # Invariant 2: L2 AWD shrinkage
    w_base = torch.tensor([1.0, 2.0, 3.0])
    w_current = torch.tensor([1.1, 2.2, 3.3])  # delta = [0.1, 0.2, 0.3]
    # alpha=0.01, lambda=10 => threshold = 0.1 => decay factor = 0.9 => delta becomes 0.9 * delta
    w_l2 = apply_anchored_decay_step(w_current, w_base, alpha=0.01, lambda_reg=10.0, penalty_type="l2")
    expected_l2 = w_base + 0.9 * (w_current - w_base)
    assert torch.allclose(w_l2, expected_l2, atol=1e-6)

    # Invariant 3: L1 AWD proximal exact sparsity
    # delta = [0.05, 0.15, -0.08]
    # threshold = alpha * lambda = 0.1
    w_curr_l1 = torch.tensor([1.05, 2.15, 2.92])
    w_l1 = apply_anchored_decay_step(w_curr_l1, w_base, alpha=0.01, lambda_reg=10.0, penalty_type="l1")
    delta_l1 = w_l1 - w_base
    # Element 0: |0.05| <= 0.1 => exactly 0.0
    assert abs(delta_l1[0].item()) < 1e-7, "L1 proximal step must zero out perturbations below threshold"
    # Element 1: 0.15 - 0.1 = 0.05
    assert abs(delta_l1[1].item() - 0.05) < 1e-6
    # Element 2: |-0.08| <= 0.1 => exactly 0.0
    assert abs(delta_l1[2].item()) < 1e-7, "L1 proximal step must zero out perturbations below threshold"

    # Invariant 4: Optimization comparison on synthetic task
    torch.manual_seed(123)
    net_vanilla = nn.Linear(16, 2, bias=False)
    net_awd_l2 = nn.Linear(16, 2, bias=False)
    net_awd_l1 = nn.Linear(16, 2, bias=False)

    # Clone exact initial weights
    w_init = torch.randn(2, 16) * 0.1
    net_vanilla.weight.data.copy_(w_init)
    net_awd_l2.weight.data.copy_(w_init)
    net_awd_l1.weight.data.copy_(w_init)

    target_val = torch.ones(2)
    def task_objective(m: nn.Module) -> float:
        x = torch.ones(1, 16)
        out = m(x).squeeze(0)
        return float(-((out - target_val) ** 2).mean().item())

    # Configure optimizers
    cfg_vanilla = AWDConfig(population_size=10, sigma=0.01, learning_rate=0.02, lambda_reg=0.0, penalty_type="l2")
    cfg_l2 = AWDConfig(population_size=10, sigma=0.01, learning_rate=0.02, lambda_reg=2.0, penalty_type="l2")
    cfg_l1 = AWDConfig(population_size=10, sigma=0.01, learning_rate=0.02, lambda_reg=0.05, penalty_type="l1")

    opt_vanilla = ESAWDOptimizer(net_vanilla, cfg_vanilla)
    opt_l2 = ESAWDOptimizer(net_awd_l2, cfg_l2)
    opt_l1 = ESAWDOptimizer(net_awd_l1, cfg_l1)

    init_score = task_objective(net_vanilla)
    for _ in range(10):
        stat_v = opt_vanilla.step(task_objective)
        stat_l2 = opt_l2.step(task_objective)
        stat_l1 = opt_l1.step(task_objective)

    final_score_v = task_objective(net_vanilla)
    final_score_l2 = task_objective(net_awd_l2)
    final_score_l1 = task_objective(net_awd_l1)

    # All three must improve fitness on target task
    assert final_score_v > init_score
    assert final_score_l2 > init_score
    assert final_score_l1 > init_score

    # AWD L2 must significantly constrain Frobenius drift compared to vanilla unregularized ES
    assert stat_l2["frobenius_drift"] < stat_v["frobenius_drift"], (
        f"AWD L2 drift ({stat_l2['frobenius_drift']}) must be less than Vanilla drift ({stat_v['frobenius_drift']})"
    )

    # AWD L1 must produce higher update sparsity than vanilla ES
    assert stat_l1["sparsity_pct"] > stat_v["sparsity_pct"], (
        f"AWD L1 sparsity ({stat_l1['sparsity_pct']}%) must exceed Vanilla sparsity ({stat_v['sparsity_pct']}%)"
    )

    return {
        "status": "PASS",
        "random_walk_drift_analytical": drift_est,
        "vanilla_drift": stat_v["frobenius_drift"],
        "awd_l2_drift": stat_l2["frobenius_drift"],
        "vanilla_sparsity_pct": stat_v["sparsity_pct"],
        "awd_l1_sparsity_pct": stat_l1["sparsity_pct"]
    }


if __name__ == "__main__":
    res = verify_es_awd_invariants()
    print("=== ES AWD Engine Verification ===")
    for k, v in res.items():
        print(f"  {k}: {v}")
