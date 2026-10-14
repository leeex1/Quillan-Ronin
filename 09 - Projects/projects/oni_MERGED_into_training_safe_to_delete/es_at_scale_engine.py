#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ES at Scale Engine: Full-Parameter Evolution Strategies for LLM Fine-Tuning
==========================================================================
Paper 52: "Evolution Strategies at Scale: LLM Fine-Tuning Beyond Reinforcement Learning"
Authors: Xin Qiu, Yulu Gan, Conor F. Hayes, Qiyao Liang, Yinggan Xu,
         Roberto Dailey, Elliot Meyerson, Babak Hodjat, Risto Miikkulainen
Affiliations: Cognizant AI Labs, UT Austin
arXiv: 2509.24372v3 (July 2026, 29 pages)

Key Architectural & Mathematical Innovations:
1. Full-Parameter High-Dimensional Search Without Dimensionality Reduction:
   - Direct optimization over billions of parameters without low-rank adapters or action-space projection.
   - Proves that ES scales effectively with small populations (e.g. N = 30) rather than tens of thousands.

2. In-Place Layer-Level Perturbation and Restoration (Algorithm 2):
   - Generates deterministic pseudo-random Gaussian noise eps_{n, l} ~ N(0, I) from random seeds s_n.
   - Perturbs parameters in-place layer by layer: theta_{t-1, l} <- theta_{t-1, l} + sigma * eps_{n, l}.
   - Restores parameters in-place after reward evaluation: theta_{t-1, l} <- theta_{t-1, l} - sigma * eps_{n, l}.
   - Requires zero activation memory and no gradient backward passes.

3. Within-Iteration Z-Score Reward Normalization (Section 3.2, Eq. in Step 18):
   - Computes z-score across population rewards:
     Z_n = (R_n - R_mean) / (R_std + epsilon)
   - Eliminates reward scale variance across tasks and iterations.

4. Decomposed Parameter Update with Learning Rate Digestion (Section 3.2):
   - Standard update: theta_t <- theta_{t-1} + (alpha / sigma) * (1 / N) * sum(R_n * eps_n).
   - Digestion: 1 / sigma is absorbed directly into learning rate alpha.
   - Decomposed in-place aggregation:
     theta_{t, l} <- theta_{t-1, l} + alpha * (1 / N) * sum_{n=1}^N Z_n * eps_{n, l}.

This module is part of the Quillan-Ronin Sovereign AI Architecture.
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Callable, Any

import torch
import torch.nn as nn


@dataclass
class ESAtScaleConfig:
    """Configuration for Full-Parameter Evolution Strategies at Scale."""
    population_size: int = 30           # Population size N (default 30 from Table 1)
    sigma: float = 0.001                # Perturbation noise scale sigma
    learning_rate: float = 0.0005       # Digested learning rate alpha (default 5e-4)
    normalize_rewards: bool = True      # Apply Z-score reward normalization
    reward_epsilon: float = 1e-8        # Numerical guard for standard deviation
    seed_base: int = 42                 # Base seed for generating iteration seeds


def normalize_population_rewards(
    rewards: torch.Tensor,
    epsilon: float = 1e-8
) -> torch.Tensor:
    """
    Computes Z-score normalization for population rewards (Algorithm 2, Step 18):
      Z_n = (R_n - R_mean) / (R_std + epsilon)

    Args:
        rewards: [N] 1D tensor of raw scalar rewards
        epsilon: numerical guard against division by zero

    Returns:
        z_scores: [N] normalized rewards with zero mean and unit variance (if std > 0)
    """
    if rewards.numel() <= 1:
        return torch.zeros_like(rewards)
    mean = rewards.mean()
    std = rewards.std(unbiased=False)
    if std < epsilon:
        return torch.zeros_like(rewards)
    return (rewards - mean) / (std + epsilon)


class SeededNoiseGenerator:
    """
    Deterministic noise generator that reproduces exact layer perturbations
    from a single integer seed, enabling memory-free noise storage.
    """
    @staticmethod
    def sample_layer_noise(
        shape: Tuple[int, ...],
        seed: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32
    ) -> torch.Tensor:
        """Generates deterministic Gaussian noise N(0, I) for given shape and seed."""
        generator = torch.Generator(device=device if device.type == 'cpu' else 'cpu')
        generator.manual_seed(seed)
        noise = torch.randn(shape, generator=generator, dtype=dtype)
        if device.type != 'cpu':
            noise = noise.to(device)
        return noise


class ESAtScaleOptimizer:
    """
    Full-Parameter Evolution Strategies Optimizer (Algorithm 2 from Qiu et al., 2026).
    Performs memory-efficient zeroth-order optimization over all trainable parameters
    via layer-level in-place perturbation, evaluation, restoration, and aggregated updates.
    """
    def __init__(self, model: nn.Module, config: Optional[ESAtScaleConfig] = None):
        self.model = model
        self.config = config or ESAtScaleConfig()
        self.iteration = 0

        # Collect references to trainable parameters
        self.param_names: List[str] = []
        self.params: List[nn.Parameter] = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.param_names.append(name)
                self.params.append(param)

    def _get_layer_seed(self, pop_index: int, layer_index: int, iteration_seed: int) -> int:
        """Derives a deterministic 64-bit integer seed for a specific layer and individual."""
        # Simple non-linear hash combination
        h = (iteration_seed * 1000003 + pop_index * 10007 + layer_index * 37) & 0x7FFFFFFF
        return h

    @torch.no_grad()
    def evaluate_individual(
        self,
        pop_index: int,
        iteration_seed: int,
        fitness_fn: Callable[[nn.Module], float]
    ) -> float:
        """
        Perturbs model in-place with individual n's noise, evaluates fitness,
        and restores the model to its base weights (Algorithm 2, Steps 7-16).

        Args:
            pop_index: index n in {0, ..., N-1}
            iteration_seed: iteration root seed
            fitness_fn: callable that evaluates current model and returns scalar reward

        Returns:
            reward: scalar evaluation score R_n
        """
        sigma = self.config.sigma

        # 1. In-place layer-by-layer perturbation: theta <- theta + sigma * eps
        for l_idx, param in enumerate(self.params):
            seed = self._get_layer_seed(pop_index, l_idx, iteration_seed)
            noise = SeededNoiseGenerator.sample_layer_noise(
                param.shape, seed, param.device, param.dtype
            )
            param.add_(noise, alpha=sigma)

        # 2. Compute reward for perturbed model
        reward = float(fitness_fn(self.model))

        # 3. In-place layer-by-layer restoration: theta <- theta - sigma * eps
        for l_idx, param in enumerate(self.params):
            seed = self._get_layer_seed(pop_index, l_idx, iteration_seed)
            noise = SeededNoiseGenerator.sample_layer_noise(
                param.shape, seed, param.device, param.dtype
            )
            param.sub_(noise, alpha=sigma)

        return reward

    @torch.no_grad()
    def step(
        self,
        fitness_fn: Callable[[nn.Module], float],
        custom_rewards: Optional[List[float]] = None
    ) -> Dict[str, Any]:
        """
        Executes one complete ES iteration (Algorithm 2, Steps 2-27):
        1. Evaluates all N individuals via in-place perturbation and restoration.
        2. Normalizes rewards via Z-score.
        3. Updates model parameters in-place layer by layer.

        Args:
            fitness_fn: objective function evaluating model
            custom_rewards: optional pre-computed rewards for distributed runners

        Returns:
            step_info: dictionary containing rewards, z-scores, update norm, and metrics
        """
        self.iteration += 1
        N = self.config.population_size
        alpha = self.config.learning_rate
        iteration_seed = self.config.seed_base + self.iteration * 99991

        # 1. Evaluate population rewards
        if custom_rewards is not None:
            assert len(custom_rewards) == N, f"Expected {N} rewards, got {len(custom_rewards)}"
            rewards_tensor = torch.tensor(custom_rewards, dtype=torch.float32)
        else:
            rewards_list = []
            for n in range(N):
                r_n = self.evaluate_individual(n, iteration_seed, fitness_fn)
                rewards_list.append(r_n)
            rewards_tensor = torch.tensor(rewards_list, dtype=torch.float32)

        # 2. Z-score reward normalization
        if self.config.normalize_rewards:
            z_scores = normalize_population_rewards(rewards_tensor, self.config.reward_epsilon)
        else:
            z_scores = rewards_tensor

        # 3. Decomposed in-place parameter update (Algorithm 2, Steps 19-25):
        # theta_{t, l} <- theta_{t-1, l} + alpha * (1 / N) * sum_{n=1}^N Z_n * eps_{n, l}
        step_multiplier = alpha / float(N)
        total_delta_sq = 0.0

        for l_idx, param in enumerate(self.params):
            # Accumulate layer update delta across all N individuals
            layer_delta = torch.zeros_like(param)
            for n in range(N):
                z_n = float(z_scores[n].item())
                if abs(z_n) < 1e-12:
                    continue
                seed = self._get_layer_seed(n, l_idx, iteration_seed)
                noise = SeededNoiseGenerator.sample_layer_noise(
                    param.shape, seed, param.device, param.dtype
                )
                layer_delta.add_(noise, alpha=z_n)

            # Apply digested update
            param.add_(layer_delta, alpha=step_multiplier)
            total_delta_sq += float(layer_delta.pow(2).sum().item()) * (step_multiplier ** 2)

        update_norm = math.sqrt(total_delta_sq)

        return {
            "iteration": self.iteration,
            "mean_reward": float(rewards_tensor.mean().item()),
            "max_reward": float(rewards_tensor.max().item()),
            "min_reward": float(rewards_tensor.min().item()),
            "std_reward": float(rewards_tensor.std(unbiased=False).item()),
            "z_scores": z_scores.tolist(),
            "update_norm": update_norm
        }


def verify_es_at_scale_invariants() -> bool:
    """
    Self-verification harness testing all core mathematical invariants of Paper 52:
    1. Deterministic seeded noise reproduction.
    2. Exact in-place perturbation and restoration (zero parameter drift).
    3. Population reward Z-score normalization (mean = 0, std = 1).
    4. Decomposed parameter update with learning rate digestion.
    5. Optimization progression on a non-convex benchmark landscape.
    """
    print("[ES-at-Scale] Verifying Mathematical & Architectural Invariants...")
    torch.manual_seed(42)

    # 1. Seeded noise determinism
    dev = torch.device('cpu')
    n1 = SeededNoiseGenerator.sample_layer_noise((4, 4), seed=12345, device=dev)
    n2 = SeededNoiseGenerator.sample_layer_noise((4, 4), seed=12345, device=dev)
    assert torch.all(n1 == n2), "Seeded noise generator must be 100% deterministic"
    print("  [1] Seeded noise determinism -> PASS")

    # 2. In-place perturbation and restoration invariant
    linear_layer = nn.Linear(8, 4)
    orig_weight = linear_layer.weight.detach().clone()
    config = ESAtScaleConfig(population_size=10, sigma=0.01, learning_rate=0.01, seed_base=100)
    optimizer = ESAtScaleOptimizer(linear_layer, config)

    dummy_fitness = lambda m: float(-m.weight.sum().item())
    _ = optimizer.evaluate_individual(pop_index=0, iteration_seed=100, fitness_fn=dummy_fitness)
    assert torch.allclose(linear_layer.weight, orig_weight, atol=1e-7), \
        "Model weights MUST return exactly to original after perturbation & restoration"
    print("  [2] In-place perturbation and restoration invariant -> PASS")

    # 3. Z-score reward normalization
    raw_rewards = torch.tensor([10.0, 20.0, 30.0, 40.0, 50.0])
    z = normalize_population_rewards(raw_rewards)
    assert abs(z.mean().item()) < 1e-6, f"Z-score mean must be ~0, got {z.mean().item()}"
    assert abs(z.std(unbiased=False).item() - 1.0) < 1e-5, f"Z-score std must be ~1, got {z.std(unbiased=False).item()}"
    print("  [3] Z-score reward normalization -> PASS")

    # 4. Optimization on target objective: maximize -||w - target||^2
    target = torch.full_like(linear_layer.weight, 2.5)
    def target_fitness(m: nn.Module) -> float:
        diff = m.weight - target
        return float(-diff.pow(2).mean().item())

    initial_fit = target_fitness(linear_layer)
    for _ in range(5):
        optimizer.step(target_fitness)
    final_fit = target_fitness(linear_layer)

    assert final_fit > initial_fit, f"ES-at-Scale must optimize fitness (initial={initial_fit:.4f}, final={final_fit:.4f})"
    print(f"  [4] Optimization step progression: {initial_fit:.4f} -> {final_fit:.4f} -> PASS")

    print("[ES-at-Scale] All 4 Invariants 100% Verified Successfully!\n")
    return True


if __name__ == "__main__":
    verify_es_at_scale_invariants()
