#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ES Forgetting Engine: Diagnostic & Analytical Framework for Catastrophic Forgetting in ES
========================================================================================
Paper 53: "Evolutionary Strategies lead to Catastrophic Forgetting in LLMs"
Authors: Immanuel Abdi*, Akshat Gupta*, Micah Mok, Alexander Lu, Nicholas Lee, Gopala Anumanchipalli
Affiliation: UC Berkeley
arXiv: 2601.20861v1 (28 Jan 2026, 11 pages)

Key Architectural & Analytical Findings:
1. High-Magnitude Parameter Drift (Frobenius Norm Explosion):
   - The Frobenius norm of model weight delta ||ΔW||_F = ||W_new - W_base||_F grows monotonically
     with training iterations.
   - After 500 iterations, ES updates are three orders of magnitude (~1000x) larger than GRPO updates,
     inducing severe drift away from the base manifold.

2. Global Lack of Update Sparsity:
   - Evaluated under absolute magnitude threshold tau = 10^-6:
     Sparsity(ΔW, tau) = (1 / |ΔW|) * sum(I[|ΔW_ij| < tau]) * 100%
   - GRPO updates exhibit high sparsity (~95%) across attention projections (Q, K, V, O) and MLP blocks,
     localizing modifications to a tiny fraction of critical weights.
   - Standard ES updates are globally dense with near-zero sparsity (<5-10%), perturbing virtually
     all stored parameters uniformly and destroying pre-existing capabilities.

3. Layerwise & Component-Wise Dissection:
   - Tracks parameter drift across key transformer architectural components:
     * Attention projections: Q, K, V (attn_qkv)
     * Attention output projection: O (attn_o)
     * Feed-Forward / MLP projections: Gate, Up, Down (mlp)
     * Layer Normalization: Input & Post-Attention Norm (norm)
   - Only LayerNorm exhibits sparsity in ES, but it accounts for a negligible fraction of total weights.

4. Continual Learning Instability & Pareto Degradation:
   - Even after new-task performance (e.g., Countdown) reaches plateau convergence (iter ~200),
     continued ES fine-tuning leads to an approximately 10% drop in prior task accuracy (e.g., HellaSwag).

This module is part of the Quillan-Ronin Sovereign AI Architecture.
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn as nn


@dataclass
class ESForgettingConfig:
    """Configuration for ES Catastrophic Forgetting Diagnostics."""
    sparsity_threshold: float = 1e-6       # Absolute magnitude threshold tau for parameter sparsity
    norm_ratio_threshold: float = 100.0    # Expected ratio ES_norm / GRPO_norm (>100x - 1000x)
    grpo_target_sparsity: float = 90.0     # Expected sparsity percentage for GRPO (>90%)
    es_max_sparsity: float = 20.0          # Upper bound of expected sparsity for dense ES (<20%)
    forgetting_penalty_coeff: float = 0.05 # Penalty weight for prior task degradation estimation


@dataclass
class ComponentMetrics:
    """Drift and sparsity metrics for a specific architectural component."""
    name: str
    param_count: int
    frobenius_norm: float
    sparsity_pct: float
    mean_abs_delta: float
    max_abs_delta: float


@dataclass
class ESDriftMetrics:
    """Comprehensive snapshot of model parameter drift and update sparsity."""
    step: int
    total_params: int
    total_frobenius_norm: float
    global_sparsity_pct: float
    mean_abs_update: float
    component_metrics: Dict[str, ComponentMetrics] = field(default_factory=dict)
    estimated_prior_retention_pct: float = 100.0


def compute_frobenius_norm(delta: torch.Tensor) -> float:
    """
    Computes Frobenius norm ||ΔW||_F = sqrt(sum(ΔW_ij^2)).
    """
    return float(torch.linalg.norm(delta.float()).item())


def compute_parameter_sparsity(delta: torch.Tensor, tau: float = 1e-6) -> Tuple[float, int, int]:
    """
    Computes sparsity percentage:
    Sparsity(ΔW, tau) = (Number of elements with |ΔW_ij| < tau) / (Total elements) * 100.0

    Returns:
        (sparsity_percentage, count_below_tau, total_elements)
    """
    total = delta.numel()
    if total == 0:
        return 100.0, 0, 0
    below_tau = int((torch.abs(delta) < tau).sum().item())
    sparsity_pct = (below_tau / total) * 100.0
    return sparsity_pct, below_tau, total


def classify_parameter_component(param_name: str) -> str:
    """
    Categorizes a parameter name into canonical transformer architectural groups:
    - 'attn_qkv': Query, Key, Value projections
    - 'attn_o': Attention output projection
    - 'mlp': Feed-forward / Gate / Up / Down projections
    - 'norm': LayerNorm / RMSNorm weights and biases
    - 'other': Embeddings, classifier head, or miscellaneous
    """
    p_lower = param_name.lower()
    if any(k in p_lower for k in ["q_proj", "k_proj", "v_proj", "qkv", "query", "key", "value"]):
        return "attn_qkv"
    elif any(k in p_lower for k in ["o_proj", "out_proj", "dense"]):
        return "attn_o"
    elif any(k in p_lower for k in ["mlp", "ffn", "gate_proj", "up_proj", "down_proj", "fc1", "fc2"]):
        return "mlp"
    elif any(k in p_lower for k in ["norm", "ln", "layernorm", "rmsnorm"]):
        return "norm"
    else:
        return "other"


class ESForgettingDiagnosticTracker:
    """
    Tracks parameter drift (Frobenius norm) and update sparsity across fine-tuning iterations.
    Compares the active model parameters against a frozen baseline checkpoint.
    """

    def __init__(
        self,
        base_model: nn.Module,
        config: Optional[ESForgettingConfig] = None
    ) -> None:
        self.config = config or ESForgettingConfig()
        # Save a detached clone of all base parameters
        self.base_state: Dict[str, torch.Tensor] = {
            name: param.detach().clone().cpu()
            for name, param in base_model.named_parameters()
        }
        self.history: List[ESDriftMetrics] = []

    def inspect_drift(self, current_model: nn.Module, step: int = 0) -> ESDriftMetrics:
        """
        Calculates layerwise and global Frobenius norms and sparsity between the current model
        and the stored base baseline.
        """
        tau = self.config.sparsity_threshold
        comp_deltas: Dict[str, List[torch.Tensor]] = {
            "attn_qkv": [],
            "attn_o": [],
            "mlp": [],
            "norm": [],
            "other": []
        }

        total_sq_norm = 0.0
        total_below_tau = 0
        total_elements = 0
        sum_abs_update = 0.0

        for name, param in current_model.named_parameters():
            if name not in self.base_state:
                continue
            base_p = self.base_state[name].to(param.device)
            delta = (param - base_p).detach()

            # Global aggregation
            sq_norm = float((delta ** 2).sum().item())
            total_sq_norm += sq_norm
            abs_sum = float(torch.abs(delta).sum().item())
            sum_abs_update += abs_sum

            below_cnt = int((torch.abs(delta) < tau).sum().item())
            total_below_tau += below_cnt
            total_elements += delta.numel()

            group = classify_parameter_component(name)
            comp_deltas[group].append(delta)

        global_frob_norm = math.sqrt(total_sq_norm)
        global_sparsity = (total_below_tau / total_elements * 100.0) if total_elements > 0 else 100.0
        mean_abs_update = (sum_abs_update / total_elements) if total_elements > 0 else 0.0

        # Component metrics
        comp_metrics: Dict[str, ComponentMetrics] = {}
        for group, d_list in comp_deltas.items():
            if not d_list:
                continue
            flat_delta = torch.cat([d.reshape(-1) for d in d_list])
            c_norm = compute_frobenius_norm(flat_delta)
            c_spars, _, c_total = compute_parameter_sparsity(flat_delta, tau)
            c_mean_abs = float(torch.abs(flat_delta).mean().item()) if c_total > 0 else 0.0
            c_max_abs = float(torch.abs(flat_delta).max().item()) if c_total > 0 else 0.0

            comp_metrics[group] = ComponentMetrics(
                name=group,
                param_count=c_total,
                frobenius_norm=c_norm,
                sparsity_pct=c_spars,
                mean_abs_delta=c_mean_abs,
                max_abs_delta=c_max_abs
            )

        # Estimate prior task capability retention:
        # Paper 53 Section 3.2: Dense perturbations + high Frobenius drift monotonically decay prior accuracy.
        # Theoretical retention model: Retention = 100% - penalty * ||ΔW||_F * (1 - Sparsity/100)
        density_factor = 1.0 - (global_sparsity / 100.0)
        drift_penalty = self.config.forgetting_penalty_coeff * global_frob_norm * density_factor
        retention_est = max(0.0, min(100.0, 100.0 - drift_penalty))

        snapshot = ESDriftMetrics(
            step=step,
            total_params=total_elements,
            total_frobenius_norm=global_frob_norm,
            global_sparsity_pct=global_sparsity,
            mean_abs_update=mean_abs_update,
            component_metrics=comp_metrics,
            estimated_prior_retention_pct=retention_est
        )

        self.history.append(snapshot)
        return snapshot


class GradientVsESDriftSimulator:
    """
    Simulates the mathematical divergence between gradient-based updates (GRPO)
    and gradient-free population perturbations (ES) over identical optimization steps.

    Mathematical Basis (Paper 53, Sections 3.2 - 3.3):
    - GRPO update: Sparse, low-rank, directional gradient along task subspace:
      ΔW_GRPO = alpha * G, where G is low-rank / sparse (nonzero only on critical path).
    - ES update: Full-parameter pseudo-gradient from isotropic Gaussian perturbations:
      ΔW_ES = (alpha / N) * sum(Z_n * eps_n), where eps_n ~ N(0, I) across all D dimensions.
    - Demonstrates:
      1. ||ΔW_ES||_F >> ||ΔW_GRPO||_F (up to 1000x difference)
      2. Sparsity(ΔW_GRPO, tau) >> Sparsity(ΔW_ES, tau) (~95% vs <10%)
    """

    def __init__(
        self,
        d_in: int = 128,
        d_out: int = 128,
        seed: int = 42
    ) -> None:
        torch.manual_seed(seed)
        self.d_in = d_in
        self.d_out = d_out
        self.w_base = torch.randn(d_out, d_in) * 0.02

    def simulate_steps(
        self,
        steps: int = 100,
        population_size: int = 30,
        sigma: float = 0.001,
        lr_es: float = 0.0005,
        lr_grpo: float = 0.0005,
        tau: float = 1e-6
    ) -> Dict[str, Any]:
        """
        Executes simulated optimization runs for both ES and GRPO and extracts drift metrics.
        """
        w_es = self.w_base.clone()
        w_grpo = self.w_base.clone()

        # Task subspace: only rank-2 active direction for the target task
        task_u = torch.randn(self.d_out, 2)
        task_v = torch.randn(self.d_in, 2)
        task_u, _ = torch.linalg.qr(task_u)
        task_v, _ = torch.linalg.qr(task_v)

        for step in range(steps):
            # GRPO: Directional, highly sparse gradient in task subspace
            # Gradient is concentrated on active singular vectors with small noise
            grad_core = torch.matmul(task_u, task_v.T) * 0.1
            # Add sparse mask: only top 5% weights receive direct gradient update
            flat_grad = grad_core.reshape(-1)
            k = int(flat_grad.numel() * 0.05)
            topk_vals, topk_idx = torch.topk(torch.abs(flat_grad), k)
            sparse_mask = torch.zeros_like(flat_grad)
            sparse_mask[topk_idx] = 1.0
            grad_sparse = (flat_grad * sparse_mask).reshape_as(grad_core)
            w_grpo = w_grpo + lr_grpo * grad_sparse

            # ES: In-place isotropic perturbation across all D dimensions (Algorithm 2 & Section 3.3)
            # eps_n ~ N(0, I)
            epsilons = torch.randn(population_size, self.d_out, self.d_in)
            # Evaluate synthetic reward: alignment with task subspace + Gaussian noise
            rewards = torch.zeros(population_size)
            for n in range(population_size):
                perturbed = w_es + sigma * epsilons[n]
                alignment = (perturbed * grad_core).sum().item()
                rewards[n] = alignment + float(torch.randn(1).item() * 0.01)

            # Z-score normalization
            r_mean = rewards.mean()
            r_std = rewards.std() + 1e-8
            z_rewards = (rewards - r_mean) / r_std

            # Digested update: theta <- theta + alpha * (1/N) * sum(Z_n * eps_n)
            delta_es = lr_es * torch.mean(z_rewards.view(population_size, 1, 1) * epsilons, dim=0)
            w_es = w_es + delta_es

        delta_w_es = w_es - self.w_base
        delta_w_grpo = w_grpo - self.w_base

        frob_es = compute_frobenius_norm(delta_w_es)
        frob_grpo = compute_frobenius_norm(delta_w_grpo)

        spars_es, _, _ = compute_parameter_sparsity(delta_w_es, tau=tau)
        spars_grpo, _, _ = compute_parameter_sparsity(delta_w_grpo, tau=tau)

        return {
            "steps": steps,
            "es_frobenius_norm": frob_es,
            "grpo_frobenius_norm": frob_grpo,
            "norm_ratio_es_to_grpo": frob_es / (frob_grpo + 1e-8),
            "es_sparsity_pct": spars_es,
            "grpo_sparsity_pct": spars_grpo,
            "delta_w_es": delta_w_es,
            "delta_w_grpo": delta_w_grpo
        }


def verify_es_forgetting_invariants() -> Dict[str, Any]:
    """
    Self-verification function testing all core mathematical invariants of Paper 53:
    1. Frobenius norm calculation accuracy.
    2. Sparsity metric under threshold tau = 1e-6.
    3. Component classification across transformer projections.
    4. Divergence between ES and GRPO: ES Frobenius norm >> GRPO and ES sparsity << GRPO.
    """
    # 1. Toy Transformer Module
    class TinyTransformerBlock(nn.Module):
        def __init__(self, d_model: int = 64):
            super().__init__()
            self.q_proj = nn.Linear(d_model, d_model, bias=False)
            self.k_proj = nn.Linear(d_model, d_model, bias=False)
            self.v_proj = nn.Linear(d_model, d_model, bias=False)
            self.o_proj = nn.Linear(d_model, d_model, bias=False)
            self.gate_proj = nn.Linear(d_model, d_model * 2, bias=False)
            self.up_proj = nn.Linear(d_model, d_model * 2, bias=False)
            self.down_proj = nn.Linear(d_model * 2, d_model, bias=False)
            self.ln1 = nn.LayerNorm(d_model)
            self.ln2 = nn.LayerNorm(d_model)

    model = TinyTransformerBlock()
    tracker = ESForgettingDiagnosticTracker(model)

    # 2. Apply simulated dense ES perturbation
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "ln" in name:
                # LayerNorm gets very small perturbation (Paper 53 Section 3.3)
                param.add_(torch.randn_like(param) * 1e-7)
            else:
                param.add_(torch.randn_like(param) * 1e-3)

    metrics = tracker.inspect_drift(model, step=100)

    # Invariant 1: Frobenius norm must be strictly positive
    assert metrics.total_frobenius_norm > 0.0, "Total Frobenius norm must be positive"

    # Invariant 2: Attention and MLP layers must have low sparsity under dense perturbation
    assert metrics.component_metrics["attn_qkv"].sparsity_pct < 20.0, "Attention projections should be dense in ES"
    assert metrics.component_metrics["mlp"].sparsity_pct < 20.0, "MLP projections should be dense in ES"

    # Invariant 3: LayerNorm with < 1e-6 deltas must show high sparsity
    assert metrics.component_metrics["norm"].sparsity_pct > 80.0, "LayerNorm with < tau deltas must show high sparsity"

    # 3. Simulate ES vs GRPO optimization divergence
    sim = GradientVsESDriftSimulator(d_in=64, d_out=64, seed=123)
    res = sim.simulate_steps(steps=50, population_size=20, sigma=0.001, lr_es=0.001, lr_grpo=0.001, tau=1e-6)

    # Invariant 4: ES Frobenius norm must exceed GRPO Frobenius norm
    assert res["es_frobenius_norm"] > res["grpo_frobenius_norm"], (
        f"ES norm ({res['es_frobenius_norm']}) should exceed GRPO norm ({res['grpo_frobenius_norm']})"
    )

    # Invariant 5: GRPO sparsity must substantially exceed ES sparsity
    assert res["grpo_sparsity_pct"] > res["es_sparsity_pct"], (
        f"GRPO sparsity ({res['grpo_sparsity_pct']}%) should exceed ES sparsity ({res['es_sparsity_pct']}%)"
    )

    return {
        "status": "PASS",
        "total_frobenius_norm": metrics.total_frobenius_norm,
        "global_sparsity_pct": metrics.global_sparsity_pct,
        "norm_ratio_es_to_grpo": res["norm_ratio_es_to_grpo"],
        "es_sparsity_pct": res["es_sparsity_pct"],
        "grpo_sparsity_pct": res["grpo_sparsity_pct"],
        "estimated_retention": metrics.estimated_prior_retention_pct
    }


if __name__ == "__main__":
    result = verify_es_forgetting_invariants()
    print("=== ES Forgetting Engine Verification ===")
    for k, v in result.items():
        print(f"  {k}: {v}")
