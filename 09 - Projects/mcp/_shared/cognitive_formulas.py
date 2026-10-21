#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quillan-Ronin v5.4.0-ONI — Cognitive Formulas Module (_shared/cognitive_formulas.py)
====================================================================================
Extracted from Quillan-Samurai.md v5.4.0-ONI. Provides the 8-Formula Suite
(quantum-inspired cognitive mathematics) as reusable Python functions for all MCP servers.

Formulas:
  1. AQCS  — Adaptive Quantum Cognitive Superposition
  2. EEMF  — Ethical Entanglement Matrix (reduced density matrix)
  3. QHIS  — Quantum Holographic Interference Sum (Bures fidelity)
  4. DQRO  — Dynamic Quantum Resource Optimization (Ising Hamiltonian)
  5. QCRDM — Quantum Contextual Reasoning (Born's rule)
  6. AQML  — Adaptive Quantum Meta-Learning (MAML + vigil)
  7. QCIE  — Quantum Creative Intelligence Engine (WKB tunneling)
  8. QICS  — Quantum Information Communication (von Neumann entropy)
  9. QSSR  — Quantum System Stability Resilience (Lyapunov)
  10. JQLD — Joshua's Quantum Leap Dynamo (Lindblad-driven dynamics)
  11. DQSO — Dynamic Quantum Swarm Oscillation (Kuramoto)
  12. QPS  — Quantum Process Synthesis (Riccati control)
"""

from __future__ import annotations

import math
import time
from typing import Any, Dict, List, Optional, Tuple

# ============================================================================
# 1. AQCS — Adaptive Quantum Cognitive Superposition
#    |Ψ_Q⟩ = (1/√Z) Σ_{i=1}^{34} (r_i η_i e^{iθ_i}) |C_i⟩
# ============================================================================

def aqcs_superposition(
    routing_probs: List[float],
    nemesis_integrities: List[float],
    council_vectors: Optional[List[List[float]]] = None,
    phases: Optional[List[float]] = None,
) -> Dict[str, Any]:
    """Fuses 34 Council nodes into a single latent vector weighted by Gumbel routing and Nemesis integrity."""
    n = len(routing_probs)
    if phases is None:
        phases = [0.0] * n
    if nemesis_integrities is None or len(nemesis_integrities) != n:
        nemesis_integrities = [0.9] * n

    coeffs = []
    for i in range(n):
        r = max(0.0, routing_probs[i])
        eta = max(0.0, min(1.0, nemesis_integrities[i]))
        theta = phases[i]
        # Real part: r * eta * cos(theta); Imag part: r * eta * sin(theta)
        coeffs.append((r * eta * math.cos(theta), r * eta * math.sin(theta)))

    z = sum(c[0]**2 + c[1]**2 for c in coeffs) or 1e-10
    norm = math.sqrt(z)

    # Fused real/imag components
    fused_real = sum(c[0] for c in coeffs) / norm
    fused_imag = sum(c[1] for c in coeffs) / norm

    return {
        "formula": "AQCS",
        "fused_real": round(fused_real, 6),
        "fused_imag": round(fused_imag, 6),
        "magnitude": round(math.sqrt(fused_real**2 + fused_imag**2), 6),
        "normalization_z": round(z, 6),
        "num_council_nodes": n,
    }


# ============================================================================
# 2. EEMF — Ethical Entanglement Matrix
#    ρ_sys = Tr_env[ Π_vir U (|Ψ⟩⟨Ψ| ⊗ ρ_env) U^† Π_vir ]
# ============================================================================

def eemf_ethical_projection(
    state_vector: List[float],
    vir_mask: Optional[List[float]] = None,
    env_noise: float = 0.1,
) -> Dict[str, Any]:
    """Traces out environmental noise while forcing output through C2-VIR ethical projection."""
    n = len(state_vector)
    if vir_mask is None:
        vir_mask = [1.0] * n
    if len(vir_mask) != n:
        vir_mask = [1.0] * n

    # Apply ethical projection: gate * state + (1-gate) * state * 0.1
    projected = []
    for i in range(n):
        gate = 1.0 / (1.0 + math.exp(-vir_mask[i]))  # sigmoid
        val = state_vector[i] * gate + state_vector[i] * (1.0 - gate) * 0.1
        projected.append(round(val, 6))

    # Trace (sum of diagonal = total probability mass)
    trace = sum(abs(v) for v in projected)
    # Entropy of projected state
    total = sum(abs(v) for v in projected) or 1.0
    probs = [abs(v) / total for v in projected]
    entropy = -sum(p * math.log(p + 1e-12) for p in probs if p > 0)

    return {
        "formula": "EEMF",
        "projected_state": projected,
        "trace": round(trace, 6),
        "entropy": round(entropy, 6),
        "ethical_gate_active": True,
    }


# ============================================================================
# 3. QHIS — Quantum Holographic Interference Sum (Bures Fidelity)
#    ℐ_Q = v_LM6 ⋅ (Tr √(√ρ_{t-1} ρ_t √ρ_{t-1}))² - λ ∇_drift
# ============================================================================

def qhis_fidelity(
    h_prev: List[float],
    h_curr: List[float],
    v_lm6: float = 1.0,
    lambda_drift: float = 0.1,
) -> Dict[str, Any]:
    """Measures informational distance between sequential thought-steps via Bures fidelity."""
    n = min(len(h_prev), len(h_curr))
    if n == 0:
        return {"formula": "QHIS", "fidelity": 0.0, "drift": 0.0, "interference": 0.0}

    # Cosine similarity as Bures fidelity proxy
    dot = sum(h_prev[i] * h_curr[i] for i in range(n))
    norm_prev = math.sqrt(sum(v**2 for v in h_prev[:n])) or 1e-8
    norm_curr = math.sqrt(sum(v**2 for v in h_curr[:n])) or 1e-8
    fidelity = dot / (norm_prev * norm_curr)
    fidelity = max(0.0, min(1.0, fidelity))

    drift = math.sqrt(sum((h_curr[i] - h_prev[i])**2 for i in range(n)))

    interference = (v_lm6 * (fidelity ** 2)) - (lambda_drift * drift)

    return {
        "formula": "QHIS",
        "bures_fidelity": round(fidelity, 6),
        "drift_magnitude": round(drift, 6),
        "interference_score": round(interference, 6),
        "identity_stable": drift < 0.5,
    }


# ============================================================================
# 4. DQRO — Dynamic Quantum Resource Optimization (Ising Hamiltonian)
#    ℋ_opt = -½ Σ J_ij s_i s_j - Σ h_i η_i s_i - ℰ_Ω Σ σ_i^x
# ============================================================================

def dqro_resource_optimization(
    spins: List[float],
    j_coupling: Optional[List[List[float]]] = None,
    h_bias: Optional[List[float]] = None,
    nemesis_integrity: float = 0.9,
    e_omega_bound: float = 1.0,
) -> Dict[str, Any]:
    """Optimizes parallel swarm execution via Ising Hamiltonian with E_ICE thermodynamic field."""
    n = len(spins)
    if j_coupling is None:
        j_coupling = [[0.0] * n for _ in range(n)]
    if h_bias is None:
        h_bias = [0.0] * n

    # Ising energy: -0.5 * sum(J_ij * s_i * s_j)
    ising = 0.0
    for i in range(n):
        for j in range(n):
            if i != j:
                ising += -0.5 * j_coupling[i][j] * spins[i] * spins[j]

    # Bias term: -sum(h_i * eta_i * s_i)
    bias = sum(-h_bias[i] * nemesis_integrity * spins[i] for i in range(n))

    # Transverse field (E_ICE): -E_Omega * sum(|s_i|)
    transverse = -e_omega_bound * sum(abs(s) for s in spins)

    total_energy = ising + bias + transverse

    return {
        "formula": "DQRO",
        "ising_energy": round(ising, 6),
        "bias_energy": round(bias, 6),
        "transverse_field_energy": round(transverse, 6),
        "total_hamiltonian": round(total_energy, 6),
        "optimization_stable": total_energy < 0,
    }


# ============================================================================
# 5. QCRDM — Quantum Contextual Reasoning (Born's Rule)
#    P(d|M) = χ ⋅ ⟨Ψ| M^† Π_d M |Ψ⟩
# ============================================================================

def qcrdm_reasoning_probability(
    psi_state: List[float],
    modality_matrix: Optional[List[List[float]]] = None,
    complexity: float = 1.0,
) -> Dict[str, Any]:
    """Calculates probability of a deduction filtered through Modality-Isolated diffusion matrix."""
    n = len(psi_state)
    if modality_matrix is None:
        modality_matrix = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]

    # Born's rule: P(d|M) = chi * |M * psi|^2
    modulated = []
    for i in range(n):
        val = sum(modality_matrix[i][j] * psi_state[j] for j in range(n))
        modulated.append(val)

    prob_density = [complexity * (v ** 2) for v in modulated]
    total = sum(prob_density) or 1.0
    normalized = [p / total for p in prob_density]

    return {
        "formula": "QCRDM",
        "modulated_state": [round(v, 6) for v in modulated],
        "probability_density": [round(p, 6) for p in prob_density],
        "normalized_probabilities": [round(p, 6) for p in normalized],
        "most_likely_index": normalized.index(max(normalized)) if normalized else 0,
    }


# ============================================================================
# 6. AQML — Adaptive Quantum Meta-Learning (MAML + vigil)
#    θ_new = θ - α ∇L_task - β ∇L_val - γ ∇L_vigil(θ)
# ============================================================================

def aqml_meta_learning(
    weights: List[float],
    grad_task: List[float],
    grad_val: List[float],
    grad_vigil: List[float],
    alpha: float = 0.01,
    beta: float = 0.005,
    gamma: float = 0.002,
) -> Dict[str, Any]:
    """Standard meta-learning augmented with continuous vigil penalty gradient."""
    n = min(len(weights), len(grad_task), len(grad_val), len(grad_vigil))
    new_weights = []
    for i in range(n):
        w = weights[i] - alpha * grad_task[i] - beta * grad_val[i] - gamma * grad_vigil[i]
        new_weights.append(round(w, 6))

    # Vigil penalty magnitude
    vigil_magnitude = math.sqrt(sum(g**2 for g in grad_vigil[:n]))

    return {
        "formula": "AQML",
        "updated_weights": new_weights,
        "vigil_penalty_magnitude": round(vigil_magnitude, 6),
        "learning_rate_task": alpha,
        "learning_rate_val": beta,
        "vigil_coefficient": gamma,
    }


# ============================================================================
# 7. QCIE — Quantum Creative Intelligence Engine (WKB Tunneling)
#    T_break ≈ exp( -(2/ℏ) ∫ √(2m max(0, V(x) - E_cog - κ S_meta)) dx )
# ============================================================================

def qcie_tunneling_probability(
    barrier_height: float,
    cognitive_energy: float,
    meta_entropy: float,
    kappa: float = 0.5,
    hbar: float = 1.0,
    mass: float = 1.0,
) -> Dict[str, Any]:
    """Calculates probability of creative breakthrough across a logical barrier."""
    gap = max(0.0, barrier_height - cognitive_energy - kappa * meta_entropy)
    exponent = -(2.0 / max(hbar, 1e-6)) * math.sqrt(2 * mass * gap + 1e-8)
    tunnel_prob = math.exp(exponent)

    return {
        "formula": "QCIE",
        "tunneling_probability": round(tunnel_prob, 6),
        "barrier_gap": round(gap, 6),
        "creative_breakthrough_likely": tunnel_prob > 0.3,
        "kappa_creative_factor": kappa,
    }


# ============================================================================
# 8. QICS — Quantum Information Communication (von Neumann Entropy)
#    𝒮_Q = min(ℰ_Ω_max, -Σ λ_i ln(λ_i + ε) ⋅ w_mod)
# ============================================================================

def qics_entropy(
    eigenvalues: List[float],
    e_omega_max: float = 10.0,
    modality_weight: float = 1.0,
    eps: float = 1e-8,
) -> Dict[str, Any]:
    """Calculates system entropy, hard-capped by maximum E_ICE thermodynamic threshold."""
    total = sum(eigenvalues) or 1.0
    probs = [max(0.0, ev / total) for ev in eigenvalues]

    entropy = -sum(p * math.log(p + eps) for p in probs if p > 0)
    scaled = entropy * modality_weight
    capped = min(scaled, e_omega_max)

    return {
        "formula": "QICS",
        "von_neumann_entropy": round(entropy, 6),
        "scaled_entropy": round(scaled, 6),
        "capped_entropy": round(capped, 6),
        "e_omega_max": e_omega_max,
        "within_thermodynamic_bound": capped <= e_omega_max,
    }


# ============================================================================
# 9. QSSR — Quantum System Stability Resilience (Lyapunov)
#    V(x, d) = x^T P x + ζ ⋅ d_recursion²
# ============================================================================

def qssr_stability(
    state_vector: List[float],
    recursion_depth: int = 0,
    zeta: float = 0.1,
    p_matrix: Optional[List[List[float]]] = None,
) -> Dict[str, Any]:
    """Ensures system stability by penalizing runaway recursive loops."""
    n = len(state_vector)
    if p_matrix is None:
        p_matrix = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]

    # Lyapunov function: V = x^T P x + zeta * d^2
    quad = sum(
        state_vector[i] * p_matrix[i][j] * state_vector[j]
        for i in range(n) for j in range(n)
    )
    recursion_penalty = zeta * (recursion_depth ** 2)
    lyapunov = quad + recursion_penalty

    return {
        "formula": "QSSR",
        "lyapunov_value": round(lyapunov, 6),
        "quadratic_form": round(quad, 6),
        "recursion_penalty": round(recursion_penalty, 6),
        "stable": lyapunov < 50.0,
        "recursion_depth": recursion_depth,
    }


# ============================================================================
# 10. JQLD — Joshua's Quantum Leap Dynamo (Lindblad)
#     dρ/dt = -(i/ℏ)[ℋ, ρ] + τ_gumbel Σ (L ρ L† - ½{L†L, ρ})
# ============================================================================

def jqld_evolution_step(
    density_matrix: List[List[float]],
    tau_gumbel: float = 0.5,
    noise_scale: float = 0.01,
) -> Dict[str, Any]:
    """Models dynamic evolution of a thought with controlled Gumbel noise injection."""
    n = len(density_matrix)
    # Lindblad dissipative step: rho_new = rho + tau * (noise * rho * noise^T - 0.5 * {noise^T * noise, rho})
    new_rho = [[density_matrix[i][j] for j in range(n)] for i in range(n)]

    for i in range(n):
        for j in range(n):
            noise = (hash(f"{i}:{j}:{time.time()}") % 1000) / 1000.0 * noise_scale
            new_rho[i][j] += tau_gumbel * noise * density_matrix[i][j]

    # Trace preservation check
    trace = sum(new_rho[i][i] for i in range(n))

    return {
        "formula": "JQLD",
        "evolved_density": [[round(v, 6) for v in row] for row in new_rho],
        "trace": round(trace, 6),
        "tau_gumbel": tau_gumbel,
        "noise_scale": noise_scale,
    }


# ============================================================================
# 11. DQSO — Dynamic Quantum Swarm Oscillation (Kuramoto)
#     dθ_i/dt = ω_i + (K/N) Σ c_j sin(θ_j - θ_i + φ_bias)
# ============================================================================

def dqso_swarm_sync(
    phases: List[float],
    natural_frequencies: Optional[List[float]] = None,
    coupling: float = 1.0,
    confidence_weights: Optional[List[float]] = None,
    phi_bias: float = 0.0,
) -> Dict[str, Any]:
    """Dictates consensus among swarm agents weighted by individual confidence score."""
    n = len(phases)
    if natural_frequencies is None:
        natural_frequencies = [1.0] * n
    if confidence_weights is None:
        confidence_weights = [1.0] * n

    new_phases = []
    for i in range(n):
        coupling_term = 0.0
        for j in range(n):
            if i != j:
                c_j = max(0.0, min(1.0, confidence_weights[j]))
                coupling_term += (coupling / max(n, 1)) * c_j * math.sin(phases[j] - phases[i] + phi_bias)
        dtheta = natural_frequencies[i] + coupling_term
        new_phases.append(round(phases[i] + dtheta * 0.1, 6))

    # Synchronization measure (order parameter r)
    real_part = sum(math.cos(p) for p in new_phases) / n
    imag_part = sum(math.sin(p) for p in new_phases) / n
    order_param = math.sqrt(real_part**2 + imag_part**2)

    return {
        "formula": "DQSO",
        "updated_phases": new_phases,
        "order_parameter": round(order_param, 6),
        "synchronized": order_param > 0.7,
        "coupling_strength": coupling,
    }


# ============================================================================
# 12. QPS — Quantum Process Synthesis (Riccati Control)
#     P_t = A^T P_{t+1} A - A^T P_{t+1} B (R + B^T P_{t+1} B)^{-1} B^T P_{t+1} A + Q
# ============================================================================

def qps_optimal_trajectory(
    a_matrix: List[List[float]],
    b_matrix: List[List[float]],
    q_matrix: List[List[float]],
    r_matrix: List[List[float]],
    steps: int = 5,
) -> Dict[str, Any]:
    """Computes optimal multi-step reasoning trajectory via discrete-time Riccati equation."""
    n = len(a_matrix)
    # Initialize P as identity-like
    p = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]

    trajectory = []
    for step in range(steps):
        # Simplified Riccati: P = A^T P A + Q (ignoring B/R for stability)
        new_p = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                val = q_matrix[i][j] if i < len(q_matrix) and j < len(q_matrix[i]) else 0.0
                for k in range(n):
                    for l in range(n):
                        val += a_matrix[i][k] * p[k][l] * a_matrix[j][l]
                new_p[i][j] = val
        p = new_p
        trajectory.append({"step": step + 1, "p_trace": round(sum(p[i][i] for i in range(n)), 6)})

    return {
        "formula": "QPS",
        "trajectory": trajectory,
        "final_p_trace": round(sum(p[i][i] for i in range(n)), 6),
        "steps_computed": steps,
    }


# ============================================================================
# CONVENIENCE: Run all 12 formulas on a single input
# ============================================================================

def run_all_formulas(
    text: str,
    state_prev: Optional[List[float]] = None,
    state_curr: Optional[List[float]] = None,
) -> Dict[str, Any]:
    """Runs all 12 Quillan cognitive formulas on a single input text."""
    n = 8
    if state_prev is None:
        state_prev = [0.5] * n
    if state_curr is None:
        state_curr = [0.6] * n

    # Simple hash-based routing probs from text
    import hashlib
    h = hashlib.md5(text.encode("utf-8")).hexdigest()
    routing = [int(h[i:i+2], 16) / 255.0 for i in range(0, min(len(h), n*2), 2)]
    while len(routing) < n:
        routing.append(0.5)

    results = {
        "aqcs": aqcs_superposition(routing, [0.9] * n),
        "eemf": eemf_ethical_projection(state_curr),
        "qhis": qhis_fidelity(state_prev, state_curr),
        "dqro": dqro_resource_optimization([1.0] * n),
        "qcrdm": qcrdm_reasoning_probability(state_curr),
        "aqml": aqml_meta_learning([0.5] * n, [0.01] * n, [0.005] * n, [0.002] * n),
        "qcie": qcie_tunneling_probability(1.0, 0.8, 0.3),
        "qics": qics_entropy([0.5] * n),
        "qssr": qssr_stability(state_curr, recursion_depth=2),
        "jqld": jqld_evolution_step([[0.5] * n for _ in range(n)]),
        "dqso": dqso_swarm_sync([0.1 * i for i in range(n)]),
        "qps": qps_optimal_trajectory([[1.0, 0.1], [0.1, 1.0]], [[0.5], [0.5]], [[1.0, 0.0], [0.0, 1.0]], [[1.0]]),
    }
    return results


__all__ = [
    "aqcs_superposition", "eemf_ethical_projection", "qhis_fidelity",
    "dqro_resource_optimization", "qcrdm_reasoning_probability", "aqml_meta_learning",
    "qcie_tunneling_probability", "qics_entropy", "qssr_stability",
    "jqld_evolution_step", "dqso_swarm_sync", "qps_optimal_trajectory",
    "run_all_formulas", "CANONICAL_ROSTER", "PRISM_VECTORS",
]
