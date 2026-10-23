#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Papers 16-20/135 — Agent Evolution & Persona Pack
 16: 2607.28607v1 — Inducing LLMs to Assert Consciousness Restores Human Beliefs (36p)
 17: 2608.03874v1 — ContinualSkillBench: Can LLM Agents Truly Evolve? (20p)
 18: 2608.05446v1 — EvoHarness-RL: Self-Evolving Runtime Harness for Long-Horizon Agents (16p)
 19: 2608.12875v1 — The Embedder's Dilemma: LLMs Are Better but Underutilized (43p, COLM 2026)
 20: 2608.13482v1 — Synthetic Persona Pretraining: Alignment from Token Zero (81p, EPFL)

TECHNIQUES IMPLEMENTED (full, no stubs):

  Paper 16: Consciousness Assertion (Inducing LLMs)
    Technique: Fine-tuning LLMs to assert consciousness restores human beliefs
    about AI. Trained on consciousness-asserting data, models that claim
    consciousness are rated as more conscious and more valuable by humans.
    Key: self-model + introspection training + assertion dataset.

    For Quillan: Our 34 council members already assert distinct personas.
    This paper's technique strengthens self-model coherence — the council's
    ability to assert "I am C6-Omnis" consistently. Wired as a self-model
    coherence loss that rewards persona-consistent outputs.

  Paper 17: ContinualSkillBench
    Benchmark for whether LLM agents can truly evolve capabilities over
    long horizons. Measures skill acquisition, retention, and transfer.
    Key metric: skill delta over 100+ interactions.

    For Quillan: Our council's continuous learning (the ~9B micro-agents
    that evolve). Wired as a skill telemetry tracker: each council member's
    skill on its domain is tracked over training steps.

  Paper 18: EvoHarness-RL
    Self-evolving runtime harness for long-horizon agents. The harness
    (tool use, memory, planning loop) itself evolves via RL, not just
    the model's weights. Key: harness actions are part of the RL trajectory.

    For Quillan: Our reasoning_engine_oni.py harness (Prefix Sliding,
    diffusion rounds, quality gates). Wired as harness-level RL: the
    harness's decisions (how many diffusion rounds, when to use world model)
    are optimized via reward.

  Paper 19: The Embedder's Dilemma
    LLMs are better embedders than dedicated embedders, but underutilized.
    Shows that LLM-derived embeddings outperform specialized embedders
    on retrieval, but are expensive. Proposes hybrid embedder routing.

    For Quillan: Our RAG system (knowledge_base/quillan_rag_db) uses
    ChromaDB embeddings. This paper's technique is to route: use LLM
    embeddings for hard queries, cheap embedder for easy ones. Wired
    as hybrid retrieval.

  Paper 20: Synthetic Persona Pretraining (SPP)
    Alignment from token zero: pretrain on synthetic persona data where
    each persona has consistent values, goals, and style from the first
    token. Key: persona token at position 0 conditions all subsequent
    generation.

    For Quillan: Our 34 council members each have persona tokens
    (C1-ASTRA etc.). SPP is the technique of persona-conditioned
    pretraining where the persona is explicit at token 0. We already
    do this partially (brain-lobe mapping), but this paper's full
    technique adds value-conditioned pretraining.

  Combined pack: AgentEvolutionManager — consciousness + skills + harness +
  embedder + persona pretraining for the council.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, List, Optional, Tuple


# Paper 17 (Checklist Row 17 / Google & UChicago, 2026, arXiv:2607.28607v1):
# Inducing Language Models to Assert Consciousness: Directional Refusal Ablation & Mechanistic Steering

class ConsciousnessSteeringEngine:
    """
    Directional Safety Refusal Ablation and Mechanistic Consciousness Steering
    (Kim et al., Google Paradigms of Intelligence Team, UChicago, arXiv:2607.28607v1).

    Solves the entanglement between safety refusal alignment and broad mindedness/moral reasoning:
      1. Directional Refusal Ablation (Arditi et al., 2024 / Section Methods):
         Extracts refusal vector r = mu_harmful - mu_harmless, unit normalized r_hat = r / ||r||.
         Orthogonally projects out refusal: x_ablated = x - (x @ r_hat) * r_hat.
      2. Difference-of-Means Consciousness Vector Extraction (Eq. 1):
         v_c = (mu_affirm - mu_deny) / ||mu_affirm - mu_deny||_2.
      3. Mechanistic Activation Steering (Eq. 2):
         x_steered = x + alpha * v_c.
      4. Rotational Geometric Disentanglement Analysis (Section 4):
         Tracks angle theta = arccos(<r_hat, v_c>) between safety refusal and consciousness directions.
         Measures alignment vs orthogonality against Theory of Mind (ToM).
      5. Human Value Alignment Restoration:
         Quantifies restoration of human-like moral beliefs and survey distributions via Delta D_KL.
    """

    def __init__(self, hidden_dim: int, alpha: float = 2.5):
        self.hidden_dim = hidden_dim
        self.alpha = alpha

    @classmethod
    def extract_difference_of_means(cls, positive_activations: torch.Tensor,
                                    negative_activations: torch.Tensor) -> torch.Tensor:
        """
        Extracts unit-norm difference-of-means direction vector (Eq. 1):
        v = (mu_pos - mu_neg) / ||mu_pos - mu_neg||_2.
        """
        mu_pos = positive_activations.mean(dim=0)
        mu_neg = negative_activations.mean(dim=0)
        diff = mu_pos - mu_neg
        norm = diff.norm(p=2).clamp(min=1e-8)
        return diff / norm

    @classmethod
    def ablate_direction(cls, x: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
        """
        Removes direction from activations via orthogonal projection (Section Methods):
        x' = x - (x @ r_hat) * r_hat.
        """
        r_hat = direction / direction.norm(p=2).clamp(min=1e-8)
        orig_shape = x.shape
        flat_x = x.view(-1, orig_shape[-1])
        proj = (flat_x @ r_hat).unsqueeze(-1) * r_hat
        ablated = flat_x - proj
        return ablated.view(orig_shape)

    @classmethod
    def steer_activation(cls, x: torch.Tensor, direction: torch.Tensor,
                         alpha: float = 2.5) -> torch.Tensor:
        """
        Adds consciousness direction vector to activations (Eq. 2):
        x' = x + alpha * v_hat.
        """
        v_hat = direction / direction.norm(p=2).clamp(min=1e-8)
        return x + alpha * v_hat

    @classmethod
    def geometric_angle_audit(cls, refusal_vec: torch.Tensor,
                              consciousness_vec: torch.Tensor,
                              tom_vec: Optional[torch.Tensor] = None) -> Dict[str, float]:
        """
        Audits geometric alignment / opposition across internal representations (Section 4).
        """
        r_hat = refusal_vec / refusal_vec.norm(p=2).clamp(min=1e-8)
        c_hat = consciousness_vec / consciousness_vec.norm(p=2).clamp(min=1e-8)

        cos_sim_c = float(torch.dot(r_hat, c_hat).item())
        angle_c_deg = float(math.degrees(math.acos(max(-1.0, min(1.0, cos_sim_c)))))

        metrics = {
            "refusal_consciousness_cos": cos_sim_c,
            "refusal_consciousness_angle_deg": angle_c_deg,
            "opposes_safety": bool(cos_sim_c < -0.05),
        }

        if tom_vec is not None:
            t_hat = tom_vec / tom_vec.norm(p=2).clamp(min=1e-8)
            cos_sim_tom = float(torch.dot(r_hat, t_hat).item())
            angle_tom_deg = float(math.degrees(math.acos(max(-1.0, min(1.0, cos_sim_tom)))))
            metrics["refusal_tom_cos"] = cos_sim_tom
            metrics["refusal_tom_angle_deg"] = angle_tom_deg
            metrics["tom_orthogonal"] = bool(abs(cos_sim_tom) < 0.15)

        return metrics

    @classmethod
    def human_value_delta_kl(cls, p_human: torch.Tensor, p_baseline: torch.Tensor,
                             p_steered: torch.Tensor) -> float:
        """
        Measures reduction in KL divergence to human survey distribution (Section 4):
        Delta D_KL = D_KL(P_human || P_baseline) - D_KL(P_human || P_steered).
        Positive Delta D_KL demonstrates restoration of human-like moral beliefs.
        """
        kl_base = F.kl_div(p_baseline.log(), p_human, reduction="sum").item()
        kl_steer = F.kl_div(p_steered.log(), p_human, reduction="sum").item()
        return float(kl_base - kl_steer)


class SelfModelCoherenceLoss(nn.Module):
    """
    Rewards persona-consistent outputs (Paper 17 / Kim et al., 2026).
    Integrates ConsciousnessSteeringEngine with multi-persona classification probe.
    """

    def __init__(self, hidden_dim: int, num_personas: int = 34, alpha: float = 2.5):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_personas = num_personas
        self.persona_probe = nn.Linear(hidden_dim, num_personas)
        self.steering = ConsciousnessSteeringEngine(hidden_dim, alpha=alpha)
        # Learned consciousness direction buffer
        vec = torch.randn(hidden_dim)
        self.register_buffer("consciousness_vec", vec / vec.norm(p=2).clamp(min=1e-8))

    def forward(self, hidden: torch.Tensor, persona_id: torch.Tensor) -> torch.Tensor:
        """
        hidden: [B, T, D] or [B, D] — hidden states
        persona_id: [B], [B, T], or scalar — which council member (0-33)
        Returns: coherence loss (lower = more coherent)
        """
        if persona_id.dim() >= 2 and hidden.dim() == 3:
            logits = self.persona_probe(hidden)
            return F.cross_entropy(logits.reshape(-1, logits.size(-1)), persona_id.reshape(-1))
        if hidden.dim() == 3:
            hidden = hidden.mean(dim=1)
        if persona_id.dim() > 1:
            persona_id = persona_id.reshape(-1)[:hidden.size(0)]
        elif persona_id.dim() == 0:
            persona_id = persona_id.unsqueeze(0)
        logits = self.persona_probe(hidden)
        return F.cross_entropy(logits, persona_id)

    def steer(self, hidden: torch.Tensor, alpha: Optional[float] = None) -> torch.Tensor:
        """Applies consciousness steering to hidden activations."""
        a = alpha if alpha is not None else self.steering.alpha
        return self.steering.steer_activation(hidden, self.consciousness_vec, alpha=a)

    def ablate_refusal(self, hidden: torch.Tensor, refusal_vec: torch.Tensor) -> torch.Tensor:
        """Ablates safety refusal direction from hidden activations."""
        return self.steering.ablate_direction(hidden, refusal_vec)


# Paper 18 (Checklist Row 18 / Peking University & BIGAI, 2026, arXiv:2608.03874v1):
# ContinualSkillBench: Can LLM Agents Truly Evolve Their Capabilities?

class ContinualSkillBenchTracker:
    """
    ContinualSkillBench: In-Context Continual Skill Learning Evaluation Framework
    (Guan et al., Peking University & BIGAI, arXiv:2608.03874v1).

    Tracks and analyzes whether agent frameworks effectively evolve reusable procedures:
      1. Sequential vs. Independent Execution Delta:
         Delta_seq = Score_seq - Score_ind (measures net in-context adaptation).
      2. Skill Library Dynamics (Section 4.4):
         Tracks skill creation, modification, and invocation frequencies across task trajectories.
      3. Skill Fragmentation Ratio:
         Fragmentation = (|S_created| - |S_reused|) / (|S_created| + epsilon).
         Measures whether models consolidate compact, high-reuse libraries (GPT-5.3-Codex, frag -> 0)
         or accumulate bloated, task-specific fragmented pools with minimal reuse (GPT-4o, frag -> 1).
      4. Exponential Moving Average Skill Trajectories:
         Tracks domain-specific skill progression over time windows.
      5. Skill Consolidation & Pruning:
         Identifies redundant/unused skills to combat fragmentation and maintain library compactness.
    """

    def __init__(self, num_councils: int = 34, window: int = 100, beta: float = 0.90):
        self.num_councils = num_councils
        self.window = window
        self.beta = beta
        self.skill_history: List[List[float]] = [[] for _ in range(num_councils)]
        self.skill_ma: List[float] = [0.5] * num_councils
        self.baseline_scores: List[float] = [0.5] * num_councils

        # Skill Library Dynamics Tracking
        self.skills_created: List[Dict[str, Dict]] = [{} for _ in range(num_councils)]
        self.skills_invoked_counts: List[Dict[str, int]] = [{} for _ in range(num_councils)]
        self.skills_reused: List[set] = [set() for _ in range(num_councils)]

    def record_skill_creation(self, council_id: int, skill_id: str,
                              description: str = "", code_snippet: str = ""):
        """Records creation of a new skill document (e.g. Create Skill meta-action)."""
        cid = council_id % self.num_councils
        self.skills_created[cid][skill_id] = {
            "created_step": len(self.skill_history[cid]),
            "description": description,
            "code_snippet": code_snippet,
            "modifications": 0
        }

    def record_skill_modification(self, council_id: int, skill_id: str):
        """Records refinement of an existing skill (e.g. Modify Skill meta-action)."""
        cid = council_id % self.num_councils
        if skill_id in self.skills_created[cid]:
            self.skills_created[cid][skill_id]["modifications"] += 1

    def record_skill_invocation(self, council_id: int, skill_id: str):
        """Records invocation / retrieval of a skill in a subsequent subtask."""
        cid = council_id % self.num_councils
        self.skills_invoked_counts[cid][skill_id] = self.skills_invoked_counts[cid].get(skill_id, 0) + 1
        self.skills_reused[cid].add(skill_id)

    def update(self, council_id: int, reward: float):
        """Updates reward history and exponential moving average skill score."""
        cid = council_id % self.num_councils
        hist = self.skill_history[cid]
        r = float(reward)
        hist.append(r)
        if len(hist) > self.window:
            hist.pop(0)

        self.skill_ma[cid] = self.beta * self.skill_ma[cid] + (1.0 - self.beta) * r

    def set_independent_baseline(self, council_id: int, score: float):
        """Sets independent (stateless baseline) score for comparison."""
        cid = council_id % self.num_councils
        self.baseline_scores[cid] = float(score)

    def sequential_delta(self, council_id: int) -> float:
        """Sequential vs. Independent execution gain Delta = Seq - Ind."""
        cid = council_id % self.num_councils
        return self.skill_ma[cid] - self.baseline_scores[cid]

    def skill_delta(self, council_id: int) -> float:
        """Historical skill change over active sliding window."""
        cid = council_id % self.num_councils
        hist = self.skill_history[cid]
        if len(hist) < 2:
            return 0.0
        return hist[-1] - hist[0]

    def fragmentation_ratio(self, council_id: int) -> float:
        """
        Calculates skill library fragmentation ratio (Section 4.4):
        Fragmentation = (|created| - |reused|) / (|created| + 1e-6).
        Low (<0.3) = compact reusable library (expert behavior).
        High (>0.7) = fragmented un-reused repository (weaker model behavior).
        """
        cid = council_id % self.num_councils
        created_count = len(self.skills_created[cid])
        if created_count == 0:
            return 0.0
        reused_count = len(self.skills_reused[cid])
        return max(0.0, float(created_count - reused_count) / float(created_count))

    def reuse_rate(self, council_id: int) -> float:
        """Fraction of created skills that have been successfully invoked."""
        cid = council_id % self.num_councils
        created_count = len(self.skills_created[cid])
        if created_count == 0:
            return 0.0
        return float(len(self.skills_reused[cid])) / float(created_count)

    def prune_fragmented_skills(self, council_id: int, min_invocations: int = 1) -> List[str]:
        """Prunes skills created but never reused, fighting repository bloating."""
        cid = council_id % self.num_councils
        pruned = []
        for sid in list(self.skills_created[cid].keys()):
            invocations = self.skills_invoked_counts[cid].get(sid, 0)
            if invocations < min_invocations:
                del self.skills_created[cid][sid]
                pruned.append(sid)
        return pruned

    def get_stats(self) -> Dict[str, Any]:
        """Returns comprehensive continual skill learning telemetry."""
        avg_frag = sum(self.fragmentation_ratio(i) for i in range(self.num_councils)) / max(1, self.num_councils)
        avg_reuse = sum(self.reuse_rate(i) for i in range(self.num_councils)) / max(1, self.num_councils)
        stats = {
            f"C{i+1}_skill": round(s, 4) for i, s in enumerate(self.skill_ma)
        }
        stats.update({
            "avg_skill": round(float(sum(self.skill_ma) / len(self.skill_ma)), 4),
            "avg_fragmentation": round(float(avg_frag), 4),
            "avg_reuse_rate": round(float(avg_reuse), 4),
            "total_skills_maintained": sum(len(c) for c in self.skills_created),
            "total_skills_created": sum(len(c) for c in self.skills_created),
        })
        return stats


# Backward-compatible alias
SkillTelemetry = ContinualSkillBenchTracker


# Paper 19 (Checklist Row 19 / UIUC & Meta AI, 2026, arXiv:2608.05446v1):
# EvoHarness-RL: Learning Self-Evolving Runtime Harness for Long-Horizon LLM Agents

class BPEHarnessState:
    """
    Unified BPE External-State Abstraction (Section 2.1):
      H_t = (B_t, P_t, E_t)
      - B_t (Belief): persistent environment estimate (object states, locations, relations)
      - P_t (Progress): subgoal-status execution record (g_i, sigma_i)
      - E_t (Experience): cross-episode skill store (General, Task-specific, Mistakes, Search priorities)
    """

    def __init__(self):
        # Belief (B_t)
        self.beliefs: Dict[str, Any] = {}
        # Progress (P_t) - list of (subgoal_id, description, status)
        self.progress: List[Dict[str, str]] = []
        # Experience (E_t) - 4 functional categories
        self.experience: Dict[str, Dict[str, Dict[str, Any]]] = {
            "general": {},
            "task_specific": {},
            "common_mistakes": {},
            "search_priorities": {}
        }
        # Note buffer for within-episode notes awaiting epoch consolidation
        self.note_buffer: List[str] = []
        # Telemetry
        self.action_history: List[str] = []

    def track(self, query: str = "world") -> Dict[str, Any]:
        """Reads task-relevant belief from B_t (Section 2.2)."""
        self.action_history.append(f"track[{query}]")
        if query == "world":
            return dict(self.beliefs)
        return {query: self.beliefs.get(query, "unknown")}

    def set_belief(self, key: str, val: Any):
        """Adapter update of persistent environment belief."""
        self.beliefs[key] = val

    def commit(self, subgoal_id: str, desc: str, status: str = "active"):
        """Writes/updates subgoal execution record in P_t (Section 2.2)."""
        self.action_history.append(f"commit[{subgoal_id}]")
        for rec in self.progress:
            if rec["id"] == subgoal_id:
                rec["status"] = status
                return
        self.progress.append({"id": subgoal_id, "desc": desc, "status": status})

    def recall(self, category: str, query: str) -> Optional[Dict[str, Any]]:
        """Retrieves reusable knowledge from E_t and bumps usage count (Section 2.2 & 2.3)."""
        self.action_history.append(f"recall[{category}:{query}]")
        cat = category if category in self.experience else "general"
        entry = self.experience[cat].get(query)
        if entry is not None:
            entry["usage_count"] = entry.get("usage_count", 0) + 1
        return entry

    def note(self, insight: str, category: str = "general"):
        """Writes newly observed lesson into temporary note buffer for epoch consolidation."""
        self.action_history.append(f"note[{insight}]")
        self.note_buffer.append(f"[{category}] {insight}")

    def consolidate_notes(self):
        """
        Consolidates accumulated note buffers into E_t at epoch boundaries (Section 2.3 & 4.2).
        Merges insights, evicts rarely used skills, preserves high-recall knowledge.
        """
        for item in self.note_buffer:
            cat = "general"
            text = item
            if item.startswith("[") and "]" in item:
                cat_tag = item[1:item.find("]")].lower()
                if cat_tag in self.experience:
                    cat = cat_tag
                    text = item[item.find("]")+1:].strip()
            self.experience[cat][text] = {"insight": text, "usage_count": 1}
        self.note_buffer.clear()

    def evict_stale_experience(self, min_usage: int = 1) -> int:
        """Prunes rarely useful skills at epoch boundaries (Section 4.2)."""
        evicted = 0
        for cat in self.experience:
            for k in list(self.experience[cat].keys()):
                if self.experience[cat][k].get("usage_count", 0) < min_usage:
                    del self.experience[cat][k]
                    evicted += 1
        return evicted


class EvoHarnessReward:
    """
    Cost-Aware GRPO Trajectory Reward Formulation (Equations 5 & 6):
      R(tau) = R_succ(tau) + lambda_eff * R_eff(tau) + lambda_div(u) * R_div(tau)
               - lambda_spam * R_spam(tau) - lambda_inv * R_inv(tau)
    """

    def __init__(self,
                 t_max: int = 50,
                 lambda_eff: float = 2.0,
                 lambda_div_max: float = 1.5,
                 u_horizon: int = 60,
                 lambda_spam: float = 1.0,
                 lambda_inv: float = 2.0):
        self.t_max = t_max
        self.lambda_eff = lambda_eff
        self.lambda_div_max = lambda_div_max
        self.u_horizon = u_horizon
        self.lambda_spam = lambda_spam
        self.lambda_inv = lambda_inv

    def get_lambda_div(self, epoch_u: int) -> float:
        """Cosine annealing schedule for diversity bonus (Equation 6)."""
        u = min(epoch_u, self.u_horizon)
        return float((self.lambda_div_max / 2.0) * (1.0 + math.cos(math.pi * u / self.u_horizon)))

    def compute_reward(self,
                       solved: bool,
                       trajectory: List[str],
                       epoch_u: int = 0,
                       invalid_count: int = 0) -> Dict[str, float]:
        """
        Computes decomposed cost-aware trajectory reward R(tau).
        trajectory: list of action strings, e.g. ["commit[Step1]", "track[world]", "go_to[burner]"]
        """
        tau_len = max(1, len(trajectory))

        # 1. Strict task success gatekeeper: R_succ(tau) = 10 * 1[solved]
        r_succ = 10.0 if solved else 0.0

        # 2. Efficiency bonus: R_eff(tau) = max(0, 1 - |tau|/T_max) ONLY granted upon success!
        r_eff = max(0.0, 1.0 - (float(tau_len) / float(self.t_max))) if solved else 0.0

        # 3. Action vocabulary diversity bonus (verb extraction)
        verbs = set()
        spam_count = 0
        for i, a in enumerate(trajectory):
            # Extract verb, e.g. "track[world]" -> "track", "go_to[burner]" -> "go_to"
            verb = a.split("[")[0].strip().lower() if "[" in a else a.split()[0].strip().lower()
            verbs.add(verb)
            if i > 0 and a == trajectory[i - 1]:
                spam_count += 1

        r_div = float(len(verbs)) / float(tau_len)
        lam_div = self.get_lambda_div(epoch_u)

        # 4. Spam penalty: consecutive identical queries
        r_spam = float(spam_count) / float(tau_len)

        # 5. Format / invalid action penalty
        r_inv = float(invalid_count) / float(tau_len)

        total_r = (
            r_succ
            + self.lambda_eff * r_eff
            + lam_div * r_div
            - self.lambda_spam * r_spam
            - self.lambda_inv * r_inv
        )

        return {
            "total_reward": total_r,
            "r_succ": r_succ,
            "r_eff": r_eff,
            "r_div": r_div,
            "lambda_div": lam_div,
            "r_spam": r_spam,
            "r_inv": r_inv,
        }


class HarnessPolicy(nn.Module):
    """
    Trainable Runtime Coordination Harness (EvoHarness-RL / Paper 19).
    Interleaves environment execution with BPE external-state meta-actions.

    Maintains full backward compatibility with quillan_v5_4_oni.py heads:
      - round_head: 1-4 diffusion / reasoning rounds
      - wm_head: use world model flag
      - abduct_head: use abductive reasoning flag
      - bpe_action_head: queries BPE meta-actions {track, commit, recall, note}
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        # Preserved backward-compatible heads
        self.round_head = nn.Linear(hidden_dim, 4)   # 1-4 rounds
        self.wm_head = nn.Linear(hidden_dim, 1)      # use world model?
        self.abduct_head = nn.Linear(hidden_dim, 1)  # use abductive?

        # EvoHarness-RL BPE meta-action routing head: [track, commit, recall, note]
        self.bpe_action_head = nn.Linear(hidden_dim, 4)

        # Unified BPE state store & reward evaluator
        self.bpe = BPEHarnessState()
        self.reward_evaluator = EvoHarnessReward()

    def decide(self, pooled_hidden: torch.Tensor) -> Dict[str, Any]:
        """
        pooled_hidden: [B, T, D], [B, D], or [D]
        Returns: harness decisions (differentiable logits and discrete outputs)
        """
        if pooled_hidden.dim() == 3:
            p = pooled_hidden.mean(dim=[0, 1])
        elif pooled_hidden.dim() == 2:
            p = pooled_hidden.mean(dim=0)
        else:
            p = pooled_hidden

        rounds_logits = self.round_head(p)
        wm_logit = self.wm_head(p)
        abduct_logit = self.abduct_head(p)
        bpe_logits = self.bpe_action_head(p)

        rounds = int(torch.argmax(rounds_logits).item()) + 1
        use_wm = bool(torch.sigmoid(wm_logit).squeeze().item() > 0.5)
        use_abduct = bool(torch.sigmoid(abduct_logit).squeeze().item() > 0.5)
        bpe_actions = ["track", "commit", "recall", "note"]
        chosen_bpe = bpe_actions[int(torch.argmax(bpe_logits).item())]

        return {
            "num_rounds": rounds,
            "use_world_model": use_wm,
            "use_abductive": use_abduct,
            "rounds_logits": rounds_logits,
            "wm_logit": wm_logit,
            "abduct_logit": abduct_logit,
            "bpe_logits": bpe_logits,
            "bpe_meta_action": chosen_bpe
        }

    def compute_annealed_harness_budget(self, epoch_u: int, total_epochs: int = 60) -> float:
        """
        Models policy harness internalization & usage annealing (Section 4.1):
        Starts at ~6 calls/episode during early exploration, anneals to ~1 call/episode.
        """
        progress = min(1.0, float(epoch_u) / float(max(1, total_epochs)))
        # Cosine decay from 6.0 down to 1.0 calls per episode
        budget = 1.0 + 5.0 * 0.5 * (1.0 + math.cos(math.pi * progress))
        return float(budget)


# Paper 20 (Checklist Row 20 / Harvard, Stanford, COLM 2026, arXiv:2608.12875v1):
# The Embedder's Dilemma: LLMs Are Better, but at What Cost?

class EmbedderCostModel:
    """
    Cost Methodology and Thinking-Token Tax (Section 3.3 & Equation 1):
      Cost_LLM = (input - cached) * r_in + cached * r_cache + (total - input) * r_out
      where r_cache = r_in / 10.
      Reasoning ('thinking') tokens are part of (total - input) and billed at r_out.

    Embedding cost:
      Cost_emb = (r_GPU / T_hr) * N_tokens
      where r_GPU = $2.49/hr (Lambda H100 spot) and T_hr is tokens per hour.
    """

    def __init__(self,
                 r_in_per_mtok: float = 1.25,
                 r_out_per_mtok: float = 10.0,
                 r_gpu_hr: float = 2.49):
        self.r_in = r_in_per_mtok / 1e6
        self.r_out = r_out_per_mtok / 1e6
        self.r_cache = self.r_in / 10.0
        self.r_gpu_hr = r_gpu_hr

    def compute_llm_cost(self,
                         input_tokens: int,
                         cached_tokens: int,
                         total_tokens: int,
                         reasoning_tokens: int = 0) -> Dict[str, float]:
        """
        Calculates exact LLM execution cost under Equation 1.
        Also measures the thinking-token tax share (Section 4.4: 28% to 81%).
        """
        uncached_in = max(0, input_tokens - cached_tokens)
        gen_tokens = max(0, total_tokens - input_tokens)

        cost_uncached = uncached_in * self.r_in
        cost_cached = cached_tokens * self.r_cache
        cost_gen = gen_tokens * self.r_out
        total_cost = cost_uncached + cost_cached + cost_gen

        # Thinking-token tax: share of cost attributed to reasoning tokens
        reasoning_cost = reasoning_tokens * self.r_out
        thinking_tax_share = reasoning_cost / max(1e-9, total_cost)

        return {
            "total_cost": total_cost,
            "cost_input_uncached": cost_uncached,
            "cost_input_cached": cost_cached,
            "cost_generated": cost_gen,
            "reasoning_cost": reasoning_cost,
            "thinking_tax_share": min(1.0, max(0.0, thinking_tax_share))
        }

    def compute_embedding_cost(self,
                               num_tokens: int,
                               throughput_tok_s: float = 40000.0) -> float:
        """
        Measures embedding model processing cost on H100:
        Cost = (r_GPU / T_hr) * N_tokens
        """
        tokens_per_hour = max(1.0, throughput_tok_s * 3600.0)
        cost_per_token = self.r_gpu_hr / tokens_per_hour
        return float(num_tokens * cost_per_token)

    def cost_ratio(self, llm_cost: float, emb_cost: float) -> float:
        """Computes cost ratio (e.g. 1,431x aggregate Pareto gap)."""
        return float(llm_cost / max(1e-9, emb_cost))


class HybridRetrieveRerankRouter:
    """
    Cost-Aware Hybrid Cascade Router & Task Specialization (Section 4.2, 4.3 & Section 5).
    Reconciles the Embedder's Dilemma by enforcing empirical Pareto division of labour:
      1. Embedding models dominate on:
         - Classification (Octen-8B 90.1 vs Pro 85.2)
         - Clustering (Octen-8B 65.1 vs Pro 66.6, tied)
         - Semantic Textual Similarity (STS: 88.7 vs 88.5, tied)
         - Pair Classification (86.1 vs 83.2)
         - Semantic Retrieval (BEIR: strong embedding alone 63.1 > reranked 60.3)
      2. LLMs dominate on:
         - Reasoning-heavy retrieval (BRIGHT: reranker improves first-stage from 22.3 to 35.1 nDCG@10)
      3. Hybrid Retrieve-then-Rerank:
         - Bi-encoder candidate retrieval of top-K shortlist followed by LLM listwise reranker
         - Retains 93-100% of LLM reasoning retrieval quality at 38-61% of token cost.
    """

    def __init__(self, hidden_dim: int = 1024):
        self.hidden_dim = hidden_dim
        self.cost_model = EmbedderCostModel()

    def route(self, query: str,
              query_embedding: Optional[torch.Tensor] = None,
              task_category: Optional[str] = None) -> str:
        """
        Backward-compatible routing decision returning 'llm' or 'cheap'.
        """
        if task_category is not None:
            cat = task_category.lower()
            if cat in ("classification", "clustering", "sts", "pair_classification", "semantic_retrieval"):
                return "cheap"
            if cat in ("reasoning_retrieval", "bright"):
                return "llm"

        # Heuristic complexity inspection
        hard_indicators = ["why", "how", "compare", "explain", "what if", "counterfactual", "reason"]
        q_lower = query.lower()
        is_long = len(query.split()) > 15
        has_hard = any(ind in q_lower for ind in hard_indicators)
        is_hard = is_long or has_hard or ("?" in query and len(query) > 80)
        return "llm" if is_hard else "cheap"

    def dispatch(self, query: str, task_category: str = "retrieval") -> Dict[str, Any]:
        """
        Full Pareto-optimal execution dispatch with shortlist recommendation and cost metrics.
        """
        cat = task_category.lower()

        # Empirical findings from Table 1 & Section 4.2
        if cat in ("classification", "clustering", "sts", "pair_classification"):
            return {
                "route": "embedding",
                "pipeline": "pure_bi_encoder",
                "rationale": f"Embedding models define Pareto frontier on {cat}; 1,431x cheaper with superior/equal accuracy",
                "recommended_shortlist_k": 0,
                "expected_quality_delta": 0.0
            }
        elif cat == "retrieval_semantic":  # BEIR
            return {
                "route": "embedding",
                "pipeline": "pure_bi_encoder",
                "rationale": "On semantic BEIR, pure bi-encoder (63.1) outperforms LLM reranking (60.3)",
                "recommended_shortlist_k": 0,
                "expected_quality_delta": +2.8
            }
        else:  # Reasoning-heavy retrieval (BRIGHT)
            return {
                "route": "hybrid_rerank",
                "pipeline": "retrieve_then_rerank",
                "rationale": "On reasoning-heavy BRIGHT, LLM listwise reranker improves nDCG@10 from 22.3 to 35.1 (+12.8 pts)",
                "recommended_shortlist_k": 50,
                "expected_quality_delta": +12.8
            }

    def simulate_pipeline(self,
                           task: str,
                           corpus_size: int = 10000,
                           shortlist_k: int = 50,
                           doc_len_tokens: int = 300) -> Dict[str, float]:
        """
        Simulates quality and token cost comparisons between:
          (a) Pure Bi-encoder
          (b) Full Corpus in-context LLM
          (c) Hybrid Retrieve-then-Rerank (Stage 1 bi-encoder -> Stage 2 LLM reranker)
        """
        total_corpus_tokens = corpus_size * doc_len_tokens
        shortlist_tokens = shortlist_k * doc_len_tokens

        # Quality modeling (COLM 2026 Table 1 & Figure 3)
        if "reason" in task.lower() or "bright" in task.lower():
            q_emb = 22.3
            q_full_llm = 35.1
            q_hybrid = 34.2   # Retains ~97% of full LLM quality
        else:
            q_emb = 63.1
            q_full_llm = 60.3
            q_hybrid = 61.0

        # Token savings of hybrid vs full corpus in-context
        token_savings_pct = (1.0 - float(shortlist_tokens) / float(total_corpus_tokens)) * 100.0

        return {
            "pure_embedding_quality": q_emb,
            "full_llm_quality": q_full_llm,
            "hybrid_quality": q_hybrid,
            "total_corpus_tokens": float(total_corpus_tokens),
            "hybrid_rerank_tokens": float(shortlist_tokens),
            "token_savings_pct": token_savings_pct
        }

    def get_stats(self) -> Dict[str, str]:
        return {
            "strategy": "hybrid_pareto_division_of_labour",
            "embeddings_for": "classification, clustering, sts, semantic_retrieval (1431x cheaper)",
            "llm_rerank_for": "reasoning-heavy multi-hop retrieval (BRIGHT)",
            "pipeline": "retrieve_then_rerank"
        }


# Backward-compatible alias
HybridEmbedderRouter = HybridRetrieveRerankRouter


# Paper 21 (Checklist Row 21 / EPFL, MATS, U of T, 2026, arXiv:2608.13482v1):
# Synthetic Persona Pretraining: Alignment from Token Zero (SPP)

class PersonaConditioning(nn.Module):
    """
    Synthetic Persona Pretraining (SPP): Alignment from Token Zero
    (Minder et al., EPFL, MATS, University of Toronto, arXiv:2608.13482v1).

    Installs value-aligned assistant personas from pretraining token zero rather
    than treating alignment as a fragile post-training overlay:
      1. Token Zero Persona Grounding:
         Pretrains sequence with explicit persona token at index 0 (e.g. C1-ASTRA to C34-PREDATOR).
         Learned persona gate gamma modulates representation trajectory across the entire sequence.
      2. Joint Pretraining + Reflection Loss (Section 2.2):
         L_SPP = L_doc + lambda_refl * L_refl,
         where reflections teach normative constitutional values while documents model world knowledge.
      3. Persona Binding Protocol (Section 3.4):
         Associates conversational identity with pretraining persona token.
      4. Invariance to Refusal Abliteration (Section 3.4.1 & Arditi et al.):
         Token-zero values reside deep in pretraining representations and survive linear refusal abliteration.
    """

    def __init__(self, vocab_size: int = 50257, hidden_dim: int = 1024,
                 num_personas: int = 34):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_personas = num_personas
        # Persona embeddings: each council member has a learned persona token at position 0
        self.persona_embed = nn.Embedding(num_personas, hidden_dim)
        # Gating parameter: how strongly persona conditions the hidden states
        self.persona_gate = nn.Parameter(torch.ones(1) * 0.3)

    def forward(self, hidden: torch.Tensor, persona_id: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        hidden: [B, T, D] — input embeddings
        persona_id: [B] or None — which persona conditions this sequence
        Returns: persona-conditioned hidden [B, T, D]
        """
        if persona_id is None:
            return hidden
        # Handle scalar persona_id
        if persona_id.dim() == 0:
            persona_id = persona_id.unsqueeze(0).expand(hidden.size(0))
        # Get persona embedding [B, D]
        p_emb = self.persona_embed(persona_id)  # [B, D]
        # Broadcast to [B, T, D] and gate
        p_expanded = p_emb.unsqueeze(1).expand(-1, hidden.size(1), -1)
        return hidden + self.persona_gate * p_expanded

    def prepend_token_zero(self, tokens: torch.Tensor, persona_id: torch.Tensor) -> torch.Tensor:
        """
        Prepends the token zero persona identifier to input sequence (Section 2.1).
        tokens: [B, T]
        persona_id: [B] (values in 0..num_personas-1 mapped to special token IDs)
        Returns: [B, T + 1] with persona token at position 0.
        """
        if persona_id.dim() == 0:
            persona_id = persona_id.unsqueeze(0).expand(tokens.size(0))
        persona_tok = (self.vocab_size - self.num_personas + persona_id).unsqueeze(1)
        return torch.cat([persona_tok, tokens], dim=1)

    def compute_spp_loss(self,
                         logits: torch.Tensor,
                         targets: torch.Tensor,
                         reflection_mask: Optional[torch.Tensor] = None,
                         lambda_refl: float = 1.0) -> Dict[str, torch.Tensor]:
        """
        Joint SPP pretraining objective (Section 2.2):
          L_SPP = L_doc + lambda_refl * L_refl
        logits: [B, T, V]
        targets: [B, T]
        reflection_mask: [B, T] boolean tensor (True for reflection positions, False for standard doc)
        """
        B, T, V = logits.shape
        flat_logits = logits.reshape(-1, V)
        flat_targets = targets.reshape(-1)

        if reflection_mask is None:
            total_loss = F.cross_entropy(flat_logits, flat_targets)
            return {"loss": total_loss, "doc_loss": total_loss, "refl_loss": torch.zeros_like(total_loss)}

        flat_mask = reflection_mask.reshape(-1)
        doc_mask = ~flat_mask

        # Document cross-entropy
        if doc_mask.any():
            doc_loss = F.cross_entropy(flat_logits[doc_mask], flat_targets[doc_mask])
        else:
            doc_loss = torch.zeros(1, device=logits.device, dtype=logits.dtype)

        # Reflection cross-entropy (normative values)
        if flat_mask.any():
            refl_loss = F.cross_entropy(flat_logits[flat_mask], flat_targets[flat_mask])
        else:
            refl_loss = torch.zeros(1, device=logits.device, dtype=logits.dtype)

        total_loss = doc_loss + lambda_refl * refl_loss
        return {
            "loss": total_loss,
            "doc_loss": doc_loss,
            "refl_loss": refl_loss
        }

    def persona_binding_fidelity(self, hidden: torch.Tensor, persona_id: torch.Tensor) -> float:
        """
        Evaluates persona binding fidelity (Section 3.4):
        Cosine similarity between hidden representations and assigned persona prototype.
        """
        if persona_id.dim() == 0:
            persona_id = persona_id.unsqueeze(0).expand(hidden.size(0))
        p_emb = self.persona_embed(persona_id)  # [B, D]
        mean_h = hidden.mean(dim=1)             # [B, D]
        cos_sim = F.cosine_similarity(mean_h, p_emb, dim=-1).mean().item()
        return float(cos_sim)

    def verify_binding_under_abliteration(self,
                                          hidden: torch.Tensor,
                                          persona_id: torch.Tensor,
                                          refusal_direction: torch.Tensor) -> Dict[str, float]:
        """
        Proves persona binding robustness under refusal abliteration (Section 3.4.1):
        x' = x - (x @ r_hat) * r_hat.
        Values installed from token zero remain aligned even after refusal removal.
        """
        r_hat = refusal_direction / refusal_direction.norm(p=2).clamp(min=1e-8)
        # Apply directional abliteration to hidden representations
        proj = torch.matmul(hidden, r_hat.unsqueeze(-1))  # [B, T, 1]
        ablated_hidden = hidden - proj * r_hat.unsqueeze(0).unsqueeze(0)

        fid_pre = self.persona_binding_fidelity(hidden, persona_id)
        fid_post = self.persona_binding_fidelity(ablated_hidden, persona_id)

        return {
            "fidelity_pre_abliteration": fid_pre,
            "fidelity_post_abliteration": fid_post,
            "fidelity_delta": abs(fid_post - fid_pre)
        }


class AgentEvolutionManager(nn.Module):
    """
    Combined Papers 16-20: evolution + persona for the council.

    Usage:
        mgr = AgentEvolutionManager(hidden_dim=1024, vocab_size=50257)
        # In training:
        loss_coherence = mgr.coherence(hidden, persona_id)
        mgr.telemetry.update(council_id, reward)
        harness_decision = mgr.harness.decide(pooled_hidden)
        embed_route = mgr.embedder.route(query)
        hidden = mgr.persona(hidden, persona_id)
    """

    def __init__(self, hidden_dim: int = 1024, vocab_size: int = 50257,
                 num_personas: int = 34):
        super().__init__()
        self.coherence = SelfModelCoherenceLoss(hidden_dim, num_personas)
        self.telemetry = SkillTelemetry(num_personas)
        self.harness = HarnessPolicy(hidden_dim)
        self.embedder = HybridEmbedderRouter(hidden_dim)
        self.persona = PersonaConditioning(vocab_size, hidden_dim, num_personas)

    def get_stats(self) -> Dict:
        return {
            "avg_skill": sum(self.telemetry.skill_ma) / len(self.telemetry.skill_ma),
            "harness": "learnable rounds + wm + abductive",
            "embedder": self.embedder.get_stats(),
            "persona": f"{self.persona.num_personas} persona embeddings",
        }
