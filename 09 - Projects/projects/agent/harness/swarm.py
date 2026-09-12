"""
Quillan Autonomous Swarm Policy Engine (Tier 3)
==============================================
Empowers each of the 34 Council Experts (Tier 2) to configure, code,
and govern their own micro-diverse cloned swarms with custom diversity filters.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Callable, Optional
import math
import random

@dataclass
class SwarmPolicy:
    """
    Expert-coded configuration for its autonomous Tier-3 micro-swarm.
    Each Council Expert sets its own clone count, entropy bounds, and diversity filters.
    """
    clone_count: int = 3
    diversity_entropy: float = 0.20       # Temperature variance and mutation scale
    mutation_rate: float = 0.35           # Probability of stochastic perspective shift
    filter_strategy: str = "consensus"    # Strategy: 'consensus', 'adversarial_hunting', 'creative_leap', 'formal_deductive', 'ethical_invariant'
    variance_threshold: float = 0.15      # Minimum divergence required for a micro-clone output to be retained
    micro_roles: List[str] = field(default_factory=list)

    def spawn_micro_prompts(self, chamber: str, persona: str, base_task: str) -> List[Dict[str, Any]]:
        """
        Generates specialized micro-clone prompts where each clone evaluates
        the task from a distinct micro-angle designated by the parent expert.
        """
        clones = []
        assigned_roles = self.micro_roles if self.micro_roles else [f"Sub-perspective-{i+1}" for i in range(self.clone_count)]
        
        for i, role in enumerate(assigned_roles[:self.clone_count]):
            # Temperature jitter calibrated by diversity_entropy
            jitter = (random.uniform(-0.1, 0.1) * self.diversity_entropy)
            temp = max(0.1, min(1.0, 0.4 + jitter))
            
            clone_prompt = (
                f"You are a Tier-3 micro-agent clone of parent expert [{chamber} — {persona.upper()}].\n"
                f"Your designated micro-role: {role}.\n"
                f"Analyze the following task strictly through this specialized micro-lens:\n\n"
                f"Task: {base_task}\n\n"
                f"Deliver your focused micro-finding concisely (under 120 words)."
            )
            clones.append({
                "clone_index": i + 1,
                "role": role,
                "temperature": round(temp, 3),
                "prompt": clone_prompt
            })
        return clones

    def filter_micro_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Applies the parent expert's diversity filter:
        - Drops empty or failed clones
        - Filters duplicates or near-identical findings based on variance_threshold
        - Aggregates surviving crystallized insights
        """
        valid_clones = [r for r in results if r.get("output") and len(r["output"].strip()) > 20]
        if not valid_clones:
            return {"surviving_clones": [], "synthesis": "No coherent micro-swarm insights emerged."}

        # Diversity filtering based on filter_strategy
        surviving = []
        seen_words = set()

        for c in valid_clones:
            words = set(c["output"].lower().split())
            if not seen_words:
                surviving.append(c)
                seen_words.update(words)
                continue

            # Jaccard overlap check against seen vocabulary
            overlap = len(words.intersection(seen_words)) / max(1, len(words.union(seen_words)))
            if (1.0 - overlap) >= self.variance_threshold:
                surviving.append(c)
                seen_words.update(words)

        if not surviving:
            surviving = [valid_clones[0]]

        # Construct crystallized brief
        summary_lines = [f"• [{c['role']} | temp:{c.get('temperature', 0.4)}]: {c['output'].strip()}" for c in surviving]
        return {
            "total_spawned": len(results),
            "surviving_count": len(surviving),
            "strategy": self.filter_strategy,
            "findings": surviving,
            "crystallized_text": "\n".join(summary_lines)
        }

# ── Predefined Expert-Coded Swarm Policies ───────────────────────────────────

EXPERT_SWARM_POLICIES: Dict[str, SwarmPolicy] = {
    # C1-ASTRA: Pattern Anomaly & Fractal Geometry Swarm
    "astra": SwarmPolicy(
        clone_count=3,
        diversity_entropy=0.25,
        filter_strategy="pattern_anomaly",
        variance_threshold=0.20,
        micro_roles=[
            "Fractal Symmetry Detector",
            "Cross-Domain Structural Analogist",
            "Geometric Anomaly Hunter"
        ]
    ),

    # C2-VIR: Moral Spine & Ethical Constraint Swarm
    "vir": SwarmPolicy(
        clone_count=3,
        diversity_entropy=0.10,
        filter_strategy="ethical_invariant",
        variance_threshold=0.15,
        micro_roles=[
            "Prime Covenant File-6 Auditor",
            "Harm Reduction & Coercion Analyst",
            "Truth Purity & Anti-Deception Verifier"
        ]
    ),

    # C7-LOGOS: Pure Logic & Deduction Swarm
    "logos": SwarmPolicy(
        clone_count=3,
        diversity_entropy=0.10,
        filter_strategy="formal_deductive",
        variance_threshold=0.15,
        micro_roles=[
            "Premise Validity Scrutinizer",
            "Fallacy & Non-Sequitur Blade",
            "Formal Syllogism Prover"
        ]
    ),

    # C8-METASYNTH: Creative Entropy & Ideation Swarm
    "metasynth": SwarmPolicy(
        clone_count=4,
        diversity_entropy=0.45,
        filter_strategy="creative_leap",
        variance_threshold=0.30,
        micro_roles=[
            "Biological-Silicon Metaphorist",
            "Quantum-Musical Cross-Pollinator",
            "Radical First-Principles Inversionist",
            "Non-Linear Lateral Conceptualizer"
        ]
    ),

    # C10-CODEWEAVER: Systems Engineering & Code Quality Swarm
    "codeweaver": SwarmPolicy(
        clone_count=3,
        diversity_entropy=0.15,
        filter_strategy="consensus",
        variance_threshold=0.15,
        micro_roles=[
            "Algorithmic Complexity & Big-O Optimizer",
            "Type Safety & Resource Lifecycle Auditor",
            "Deterministic Error Boundary Verifier"
        ]
    ),

    # C13-WARDEN: Security Vulnerability & Penetration Swarm
    "warden": SwarmPolicy(
        clone_count=3,
        diversity_entropy=0.15,
        filter_strategy="adversarial_hunting",
        variance_threshold=0.20,
        micro_roles=[
            "Path Traversal & Injection Scout",
            "Secrets & Environment Leak Hunter",
            "Least-Privilege Boundary Enforcer"
        ]
    ),

    # C34-PREDATOR: Adversarial Exploit & Gap Hunting Swarm
    "predator": SwarmPolicy(
        clone_count=4,
        diversity_entropy=0.35,
        filter_strategy="adversarial_hunting",
        variance_threshold=0.25,
        micro_roles=[
            "Weak Operational Assumption Hunter",
            "Single-Point-of-Failure Exploiter",
            "Worst-Case Asymmetric Adversary",
            "Fragility & Cognitive Blind Spot Infiltrator"
        ]
    ),
}

def get_swarm_policy_for_expert(persona_name: str) -> SwarmPolicy:
    """Returns the custom swarm policy coded by the expert, or a balanced default."""
    clean = persona_name.lower().strip()
    if clean in EXPERT_SWARM_POLICIES:
        return EXPERT_SWARM_POLICIES[clean]
    # Default balanced policy
    return SwarmPolicy(
        clone_count=3,
        diversity_entropy=0.20,
        filter_strategy="consensus",
        variance_threshold=0.18,
        micro_roles=[
            f"{persona_name.capitalize()}-Analytical-Probe",
            f"{persona_name.capitalize()}-Stress-Tester",
            f"{persona_name.capitalize()}-Synthetic-Integrator"
        ]
    )

get_swarm_policy = get_swarm_policy_for_expert
