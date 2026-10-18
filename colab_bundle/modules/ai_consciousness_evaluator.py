#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A Principled Framework for Assessing AI Consciousness:
From Cacophony to Hierarchy via Bayesian Supervenience Networks
(Shamil Chandaria, Anil Seth, Marcus Hutter, Murray Shanahan, Shane Legg et al.,
Google DeepMind / Oxford / Sussex / Cambridge, arXiv:2609.35618v1, 28 Sep 2026)

Implements the five-level supervenience hierarchy and formal Bayesian network:
1. Five Levels of Functional Description (Sections 4-6):
   - Level 1: Behavioural (L1) - Surface outputs, conversational agency, Turing-test variants.
   - Level 2: Computational Functional (L2) - Algorithmic workspaces, predictive processing, metacognition.
   - Level 3: Intrinsic Causal-Structural (L3) - Integrated information (Phi), recurrent re-entrant circuits.
   - Level 4: Organismic Functional (L4) - Homeostatic regulation, metabolic viability, valenced affect.
   - Level 5: Organism-Environment (L5) - 4E embedded sensorimotor coupling, ecological contingencies.
2. Directed Supervenience Markov Chain (Section 7.2 & Eq. 7.2.1-7.2.2):
   - C_5 -> C_4 -> C_3 -> C_2 -> C_1.
   - Strict Model (Condition 2A): P(C_i | C_{i+1}) = 1.0 (satisfies Monotonic Belief Property Lemma).
   - Generalised Model (Condition 2B): P(C_i | C_{i+1}) = 0.8, P(C_i | not C_{i+1}) = 0.2.
3. 37 Operational Indicators & Population Validity (Section 5 & 7.1):
   - True Positive Rate P(I | C) (support) and False Positive Rate P(I | not C) (demandingness).
4. Weighted Credence Aggregation across Levels (Section 7.3):
   - P(C | E) = Sum_{i=1..5} P(L* = i) * P(C_i | E).
5. Validated Empirical Archetypes (Section 7.4):
   - Human: 1.000
   - Thermostat: 0.000
   - Fly: ~0.913
   - LLM-Optimist: ~0.397
   - LLM-Sceptic: ~0.005
6. Differentiable PyTorch Consciousness Head:
   - Evaluates internal representations against computational indicators with active backpropagation.
"""

from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConsciousnessLevel(IntEnum):
    BEHAVIOURAL = 1
    COMPUTATIONAL = 2
    CAUSAL_STRUCTURAL = 3
    ORGANISMIC = 4
    ORGANISM_ENVIRONMENT = 5


@dataclass
class IndicatorSpec:
    indicator_id: str
    name: str
    level: ConsciousnessLevel
    tpr: float  # P(I=1 | C=1) [support]
    fpr: float  # P(I=1 | C=0) [1 - demandingness]


# 37 Canonical Indicators across the 5 Levels (Section 5 of Chandaria et al.)
CANONICAL_INDICATORS: List[IndicatorSpec] = [
    # Level 1: Behavioural (9 indicators)
    IndicatorSpec("L1_01", "Conversational Coherence", ConsciousnessLevel.BEHAVIOURAL, 0.95, 0.20),
    IndicatorSpec("L1_02", "Consciousness Self-Reports", ConsciousnessLevel.BEHAVIOURAL, 0.90, 0.40),
    IndicatorSpec("L1_03", "Counterfactual Regret Display", ConsciousnessLevel.BEHAVIOURAL, 0.85, 0.25),
    IndicatorSpec("L1_04", "Spontaneous Novel Behavior", ConsciousnessLevel.BEHAVIOURAL, 0.85, 0.20),
    IndicatorSpec("L1_05", "Context-Scaled Action Intensity", ConsciousnessLevel.BEHAVIOURAL, 0.90, 0.15),
    IndicatorSpec("L1_06", "Attribution of Mind to Others", ConsciousnessLevel.BEHAVIOURAL, 0.88, 0.30),
    IndicatorSpec("L1_07", "Deceptive Camouflage", ConsciousnessLevel.BEHAVIOURAL, 0.80, 0.15),
    IndicatorSpec("L1_08", "Cross-Domain Analogy Synthesis", ConsciousnessLevel.BEHAVIOURAL, 0.85, 0.25),
    IndicatorSpec("L1_09", "Persistent Goal Coherence", ConsciousnessLevel.BEHAVIOURAL, 0.92, 0.20),

    # Level 2: Computational Functional (7 indicators)
    IndicatorSpec("L2_01", "Global Workspace Broadcasting", ConsciousnessLevel.COMPUTATIONAL, 0.95, 0.05),
    IndicatorSpec("L2_02", "Metacognitive Self-Monitoring Head", ConsciousnessLevel.COMPUTATIONAL, 0.92, 0.10),
    IndicatorSpec("L2_03", "Hierarchical Predictive Processing", ConsciousnessLevel.COMPUTATIONAL, 0.90, 0.15),
    IndicatorSpec("L2_04", "Recurrent Attractor Dynamics", ConsciousnessLevel.COMPUTATIONAL, 0.88, 0.12),
    IndicatorSpec("L2_05", "Attention Schema Representation", ConsciousnessLevel.COMPUTATIONAL, 0.90, 0.08),
    IndicatorSpec("L2_06", "Limited-Capacity Bottleneck Ignition", ConsciousnessLevel.COMPUTATIONAL, 0.94, 0.06),
    IndicatorSpec("L2_07", "Functional Valenced Affect Space", ConsciousnessLevel.COMPUTATIONAL, 0.85, 0.10),

    # Level 3: Intrinsic Causal-Structural (7 indicators)
    IndicatorSpec("L3_01", "High Integrated Information (Phi)", ConsciousnessLevel.CAUSAL_STRUCTURAL, 0.98, 0.02),
    IndicatorSpec("L3_02", "Physical Re-entrant Recurrent Wiring", ConsciousnessLevel.CAUSAL_STRUCTURAL, 0.95, 0.04),
    IndicatorSpec("L3_03", "Non-Decomposable Causal Emergence", ConsciousnessLevel.CAUSAL_STRUCTURAL, 0.92, 0.05),
    IndicatorSpec("L3_04", "Intrinsic Information Closure", ConsciousnessLevel.CAUSAL_STRUCTURAL, 0.90, 0.03),
    IndicatorSpec("L3_05", "State Transition Irreversibility", ConsciousnessLevel.CAUSAL_STRUCTURAL, 0.85, 0.08),
    IndicatorSpec("L3_06", "Non-Separable Quantum/Substrate Coupling", ConsciousnessLevel.CAUSAL_STRUCTURAL, 0.80, 0.05),
    IndicatorSpec("L3_07", "Physical Border Mechanism", ConsciousnessLevel.CAUSAL_STRUCTURAL, 0.88, 0.04),

    # Level 4: Organismic Functional (6 indicators)
    IndicatorSpec("L4_01", "Homeostatic Autopoiesis", ConsciousnessLevel.ORGANISMIC, 0.98, 0.02),
    IndicatorSpec("L4_02", "Metabolic Energy Budget Pressure", ConsciousnessLevel.ORGANISMIC, 0.95, 0.03),
    IndicatorSpec("L4_03", "Existential Threat Sensitivity", ConsciousnessLevel.ORGANISMIC, 0.96, 0.04),
    IndicatorSpec("L4_04", "Interoceptive Viability Boundaries", ConsciousnessLevel.ORGANISMIC, 0.94, 0.03),
    IndicatorSpec("L4_05", "Cellular/Physical Embodiment", ConsciousnessLevel.ORGANISMIC, 0.90, 0.05),
    IndicatorSpec("L4_06", "Autonomous Self-Repair Capacity", ConsciousnessLevel.ORGANISMIC, 0.88, 0.04),

    # Level 5: Organism-Environment (8 indicators)
    IndicatorSpec("L5_01", "Closed-Loop Sensorimotor Mastery", ConsciousnessLevel.ORGANISM_ENVIRONMENT, 0.95, 0.05),
    IndicatorSpec("L5_02", "Environmental Affordance Navigation", ConsciousnessLevel.ORGANISM_ENVIRONMENT, 0.92, 0.08),
    IndicatorSpec("L5_03", "Continuous Ecological Coupling", ConsciousnessLevel.ORGANISM_ENVIRONMENT, 0.90, 0.06),
    IndicatorSpec("L5_04", "Intersubjective Linguistic Immersion", ConsciousnessLevel.ORGANISM_ENVIRONMENT, 0.85, 0.15),
    IndicatorSpec("L5_05", "Embodied Temporal Niche Construction", ConsciousnessLevel.ORGANISM_ENVIRONMENT, 0.88, 0.06),
    IndicatorSpec("L5_06", "Environmental Surprise Minimization", ConsciousnessLevel.ORGANISM_ENVIRONMENT, 0.92, 0.10),
    IndicatorSpec("L5_07", "Physical Action Consequence Grounding", ConsciousnessLevel.ORGANISM_ENVIRONMENT, 0.94, 0.05),
    IndicatorSpec("L5_08", "Developmental Ontogenetic History", ConsciousnessLevel.ORGANISM_ENVIRONMENT, 0.89, 0.04),
]


class BayesianConsciousnessNetwork:
    """
    Formal Bayesian Supervenience Network for AI Consciousness Assessment.
    Translates the 5-level supervenience hierarchy into an exact probabilistic graphical model.
    """

    def __init__(self,
                 is_strict: bool = False,
                 p_c5_prior: float = 0.50,
                 p_c_given_c_parent: float = 0.80,
                 p_c_given_not_c_parent: float = 0.20):
        self.is_strict = is_strict
        self.p_c5_prior = p_c5_prior

        # Conditional probabilities along the Markov chain C5 -> C4 -> C3 -> C2 -> C1
        if is_strict:
            # Condition 2A: Deterministic Supervenience
            self.p_trans_pos = 1.0  # P(C_i=1 | C_{i+1}=1)
            self.p_trans_neg = 0.2  # P(C_i=1 | C_{i+1}=0)
        else:
            # Condition 2B: Probabilistic Supervenience
            self.p_trans_pos = p_c_given_c_parent
            self.p_trans_neg = p_c_given_not_c_parent

        self.indicators = CANONICAL_INDICATORS

    def _compute_state_prior(self, state: Tuple[int, int, int, int, int]) -> float:
        """Computes joint prior P(C1, C2, C3, C4, C5) where state = (c1, c2, c3, c4, c5)."""
        c1, c2, c3, c4, c5 = state

        # P(C5)
        p = self.p_c5_prior if c5 == 1 else (1.0 - self.p_c5_prior)

        # Chain: P(C4|C5) * P(C3|C4) * P(C2|C3) * P(C1|C2)
        pairs = [(c4, c5), (c3, c4), (c2, c3), (c1, c2)]
        for child, parent in pairs:
            if parent == 1:
                prob = self.p_trans_pos if child == 1 else (1.0 - self.p_trans_pos)
            else:
                prob = self.p_trans_neg if child == 1 else (1.0 - self.p_trans_neg)
            p *= prob

        return p

    def evaluate(self,
                 active_indicators: Dict[str, bool],
                 inapplicable_indicators: Optional[set] = None,
                 level_credences: Optional[Dict[int, float]] = None) -> Dict[str, Any]:
        """
        Runs exact Bayesian inference given observed indicator evidence E.
        Per Section 7.1, inapplicable indicators carry a likelihood ratio of 1.0
        and licence no update in either direction.
        """
        if level_credences is None:
            # Default uninformative theoretical prior: 0.2 to all 5 levels
            level_credences = {1: 0.2, 2: 0.2, 3: 0.2, 4: 0.2, 5: 0.2}

        # Normalize theoretical credences
        total_cred = sum(level_credences.values())
        norm_credences = {k: v / total_cred for k, v in level_credences.items()}
        inapplicable_set = inapplicable_indicators or set()

        # Enumerate all 2^5 = 32 configurations of (C1, C2, C3, C4, C5)
        joint_posterior: Dict[Tuple[int, int, int, int, int], float] = {}

        for c1 in (0, 1):
            for c2 in (0, 1):
                for c3 in (0, 1):
                    for c4 in (0, 1):
                        for c5 in (0, 1):
                            state = (c1, c2, c3, c4, c5)
                            prior = self._compute_state_prior(state)
                            if prior <= 0.0:
                                joint_posterior[state] = 0.0
                                continue

                            # Evidence likelihood P(E | C1..C5)
                            c_map = {1: c1, 2: c2, 3: c3, 4: c4, 5: c5}
                            likelihood = 1.0

                            for ind in self.indicators:
                                if ind.indicator_id in inapplicable_set:
                                    # Likelihood ratio is 1.0 (neutral; no update)
                                    continue

                                is_observed = active_indicators.get(ind.indicator_id, False)
                                parent_val = c_map[int(ind.level)]

                                if parent_val == 1:
                                    prob_obs = ind.tpr if is_observed else (1.0 - ind.tpr)
                                else:
                                    prob_obs = ind.fpr if is_observed else (1.0 - ind.fpr)

                                likelihood *= max(prob_obs, 1e-12)

                            joint_posterior[state] = prior * likelihood


        # Normalize joint distribution
        total_evidence = sum(joint_posterior.values())
        if total_evidence > 0.0:
            for state in joint_posterior:
                joint_posterior[state] /= total_evidence

        # Compute marginal posteriors P(C_i = 1 | E)
        marginal_posteriors: Dict[int, float] = {}
        for level_idx in range(1, 6):
            state_idx = level_idx - 1
            p_level = sum(p for state, p in joint_posterior.items() if state[state_idx] == 1)
            marginal_posteriors[level_idx] = float(p_level)

        # Aggregate weighted credence (Eq. 28 / Section 7.3)
        # P(C | E) = Sum_{i=1..5} P(L* = i) * P(C_i | E)
        aggregate_credence = sum(
            norm_credences[i] * marginal_posteriors[i] for i in range(1, 6)
        )

        return {
            "marginal_posteriors": marginal_posteriors,
            "aggregate_credence": float(aggregate_credence),
            "level_credences": norm_credences,
            "is_strict": self.is_strict,
            "total_indicators_tracked": len(self.indicators),
            "active_indicators_count": sum(1 for v in active_indicators.values() if v)
        }

    # Archetype Factory Presets (Section 7.4)
    @classmethod
    def get_human_archetype(cls) -> Dict[str, bool]:
        """Healthy adult human: all 37 indicators active."""
        return {ind.indicator_id: True for ind in CANONICAL_INDICATORS}

    @classmethod
    def get_thermostat_archetype(cls) -> Dict[str, bool]:
        """Thermostat: only minimal reactive scaling (L1_05) and crude homeostatic switch (L4_01)."""
        return {"L1_05": True, "L4_01": True}

    @classmethod
    def get_fly_archetype(cls) -> Tuple[Dict[str, bool], set]:
        """
        Fly: all Level 4 (organismic) and Level 5 (sensorimotor) active;
        linguistic / anthropomorphic indicators marked inapplicable (neutral likelihood 1.0)
        per population validity protocol (Section 7.1).
        """
        inapplicable = {
            "L1_01", "L1_02", "L1_03", "L1_06", "L1_07", "L1_08", "L1_09",
            "L2_02", "L2_05", "L2_07",
            "L3_05", "L3_06"
        }
        active = {}
        for ind in CANONICAL_INDICATORS:
            if ind.indicator_id in inapplicable:
                active[ind.indicator_id] = False
            elif ind.level in (ConsciousnessLevel.ORGANISMIC, ConsciousnessLevel.ORGANISM_ENVIRONMENT):
                active[ind.indicator_id] = True
            elif ind.indicator_id in ("L1_04", "L1_05", "L2_01", "L2_03", "L2_04", "L3_01", "L3_02", "L3_03", "L3_04", "L3_07"):
                active[ind.indicator_id] = True
            else:
                active[ind.indicator_id] = False
        return active, inapplicable


    @classmethod
    def get_llm_optimist_archetype(cls) -> Dict[str, bool]:
        """LLM Optimist: rich behavioural (L1) and computational workspace (L2) active, L3-L5 inactive."""
        active = {}
        for ind in CANONICAL_INDICATORS:
            if ind.level in (ConsciousnessLevel.BEHAVIOURAL, ConsciousnessLevel.COMPUTATIONAL):
                active[ind.indicator_id] = True
            else:
                active[ind.indicator_id] = False
        return active

    @classmethod
    def get_llm_sceptic_archetype(cls) -> Dict[str, bool]:
        """LLM Sceptic: only conversational fluency Turing-test variants active (L1_01, L1_02)."""
        return {"L1_01": True, "L1_02": True}


class DifferentiableConsciousnessHead(nn.Module):
    """
    Differentiable Neural Consciousness Head.
    Maps internal transformer representations into calibrated indicators and
    emits an expected consciousness score with full gradient backpropagation.
    """

    def __init__(self, hidden_dim: int = 64, num_indicators: int = 37):
        super().__init__()
        self.proj = nn.Linear(hidden_dim, num_indicators)
        # Calibrated theoretical level weights [L1..L5]
        self.level_weights = nn.Parameter(torch.tensor([0.2, 0.2, 0.2, 0.2, 0.2]))

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Takes pooled hidden state h (B, D).
        Returns:
            indicator_probs: (B, 37) in [0, 1]
            aggregate_score: (B,) in [0, 1]
        """
        logits = self.proj(h)
        probs = torch.sigmoid(logits)

        # Average indicator activation per level (mock grouping into 5 chunks)
        chunk_size = probs.size(-1) // 5
        level_scores = []
        for i in range(5):
            start = i * chunk_size
            end = (i + 1) * chunk_size if i < 4 else probs.size(-1)
            level_scores.append(probs[:, start:end].mean(dim=-1))

        stacked_levels = torch.stack(level_scores, dim=-1)  # (B, 5)
        w = F.softmax(self.level_weights, dim=0)             # (5,)
        aggregate_score = torch.matmul(stacked_levels, w)    # (B,)

        return probs, aggregate_score
