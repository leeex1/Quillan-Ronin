#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WikiSkill Engine: Compiling Agent Experience into Persistent Knowledge for Skill Evolution
========================================================================================
Formal implementation of WikiSkill (Tang, Rashtchian, Ferng, Tomkins, Juan, Vu; Google Research,
Virginia Tech, 2026 / arXiv:2608.27454).

Mathematical & Architectural Formulations:
------------------------------------------
1. Three-Layer Memory & Codebase Separation (Section 3.1):
   - Raw Execution Layer (raw/): Immutable execution traces tau_i ~ pi(x_i; S_{k-1}) capturing tool
     invocations, stdout/stderr, environment responses, and task outcomes.
   - Persistent Knowledge Layer (wiki/): Continually compounding knowledge base containing:
       * wiki/patterns/*.md: Root-cause failure mode diagnoses, effective strategies, actionable workarounds.
       * wiki/index.md: Catalog of registered patterns.
       * wiki/logs.md: Evolutionary log tracking findings per iteration.
       * wiki/skill-impact.md: Programmatic audit tracker capturing diffs, validation scores, and outcomes.
     Compounding Invariant: The Wiki Layer is NEVER rolled back on validation rejection.
   - Executable Skills Layer (skills/): Active modular filesystem skills S:
       * skills/<name>/SKILL.md: Reusable procedural instructions and code.
       * skills/<name>/PURPOSE.md: Bidirectional map linking the skill back to motivating wiki patterns.

2. Stratified Trace Sampling (Section 3.2.2 & Appendix C):
   T_sample,k subset of T_train,k with bounded budget |T_sample,k| <= 8:
   - Up to 5 failing traces for root-cause diagnosis.
   - Up to 3 passing traces to preserve successful behaviors and prevent regressions.
   - Capped character budget (15,000 characters per trace).

3. Evolutionary Loop & Validation Gating (Algorithm 1):
   - Candidate Skill: S'_k = Apply(S_{k-1}, P_k)
   - Acceptance Rule:
       S_k = S'_k   if R(T_val,k) > R_best
       S_k = S_{k-1} otherwise (rollback skills only; wiki retained)
   - Update Step: W_k = Update(W'_k, P_k, R(T_val,k), a_k) where a_k in {Accepted, Rejected}

4. Optimizer API Call Complexity (Table 7 & Section D.2):
   - WikiSkill: C_WikiSkill = (1 + T_ReAct) * (N_train / B).
     In full-batch mode (B = N_train): C = 1 + T_ReAct (O(1) w.r.t. training set size N_train).
   - Trace2Skill: C_Trace2Skill = N_train + (1 + 1/(c - 1)) * (N_train / B) + 1 (O(N_train)).
   - EvoSkill: C_EvoSkill = 2 * (N_train / B) (O(N_train / B)).
   - SkillOpt: C_SkillOpt = K_opt * (N_train / B) (O(N_train / B)).

5. Paired Bootstrap Significance Testing (Appendix C):
   Resamples test instances with replacement over B_iter = 1000 iterations to evaluate
   empirical p-values and statistical ties (p >= 0.05).
"""

import math
import copy
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set, Any


@dataclass
class ExecutionTrace:
    """Immutable trace of an agent executing a task in the environment (raw/ layer)."""
    task_id: str
    prompt: str
    tool_calls: List[Dict[str, Any]]
    output: str
    success: bool
    reward: float
    raw_log: str = ""
    max_char_cap: int = 15000

    def get_capped_log(self) -> str:
        """Returns the execution log capped at max_char_cap per Appendix C specifications."""
        text = self.raw_log if self.raw_log else f"Prompt: {self.prompt}\nTools: {self.tool_calls}\nOutput: {self.output}"
        if len(text) > self.max_char_cap:
            return text[:self.max_char_cap] + "\n[TRUNCATED_AT_15K_CHARS]"
        return text


@dataclass
class WikiPattern:
    """Structured knowledge pattern compiled into wiki/patterns/*.md."""
    pattern_id: str
    title: str
    category: str  # "failure_mode" or "successful_strategy" or "workaround"
    root_cause: str
    recommended_action: str
    evidence_traces: List[str]
    recurrence_count: int = 1
    iteration_created: int = 0
    iteration_last_updated: int = 0

    def to_markdown(self) -> str:
        return (
            f"# Pattern: {self.title} ({self.pattern_id})\n"
            f"- Category: {self.category}\n"
            f"- Recurrence Count: {self.recurrence_count}\n"
            f"- First Seen: Iteration {self.iteration_created} | Last Seen: Iteration {self.iteration_last_updated}\n\n"
            f"## Root Cause Analysis\n{self.root_cause}\n\n"
            f"## Actionable Solution / Workaround\n{self.recommended_action}\n\n"
            f"## Evidence Traces\n" + "\n".join(f"- `{t}`" for t in self.evidence_traces)
        )


@dataclass
class SkillImpactRecord:
    """Historical record in wiki/skill-impact.md tracking proposal outcomes."""
    iteration: int
    proposal_id: str
    target_skill: str
    action: str  # "create" or "patch"
    diff_summary: str
    val_score: float
    previous_best_score: float
    outcome: str  # "Accepted" or "Rejected"


@dataclass
class AgentSkill:
    """Executable modular skill in skills/ directory."""
    name: str
    skill_content: str  # Content of SKILL.md
    purpose_content: str  # Content of PURPOSE.md (mapping to wiki patterns)
    motivating_patterns: List[str] = field(default_factory=list)
    version: int = 1


class PersistentWiki:
    """
    Persistent Knowledge Base (wiki/ layer).
    Invariant: Knowledge compounds monotonically across all iterations.
    The wiki is NEVER rolled back on candidate skill rejection.
    """

    def __init__(self):
        self.patterns: Dict[str, WikiPattern] = {}
        self.logs: List[str] = []
        self.skill_impact_tracker: List[SkillImpactRecord] = []

    def add_or_update_pattern(self, pattern: WikiPattern) -> None:
        if pattern.pattern_id in self.patterns:
            existing = self.patterns[pattern.pattern_id]
            existing.recurrence_count += 1
            existing.iteration_last_updated = pattern.iteration_last_updated
            existing.root_cause = pattern.root_cause
            existing.recommended_action = pattern.recommended_action
            for t in pattern.evidence_traces:
                if t not in existing.evidence_traces:
                    existing.evidence_traces.append(t)
        else:
            self.patterns[pattern.pattern_id] = copy.deepcopy(pattern)

    def log_iteration(self, iter_idx: int, summary: str) -> None:
        entry = f"### Iteration {iter_idx} Summary\n{summary}\n"
        self.logs.append(entry)

    def record_impact(self, record: SkillImpactRecord) -> None:
        self.skill_impact_tracker.append(record)

    def get_rejected_proposals(self) -> Set[str]:
        """Returns target skills and actions that were rejected to prevent repeating failed interventions."""
        rejected = set()
        for rec in self.skill_impact_tracker:
            if rec.outcome == "Rejected":
                rejected.add(f"{rec.target_skill}:{rec.diff_summary}")
        return rejected

    def get_index_markdown(self) -> str:
        lines = ["# Wiki Pattern Catalog (wiki/index.md)", "| ID | Title | Category | Recurrence | Last Updated |", "| --- | --- | --- | --- | --- |"]
        for pid, p in sorted(self.patterns.items()):
            lines.append(f"| {pid} | {p.title} | {p.category} | {p.recurrence_count} | Iteration {p.iteration_last_updated} |")
        return "\n".join(lines)


class SkillSet:
    """Active executable skill set (skills/ layer)."""

    def __init__(self):
        self.skills: Dict[str, AgentSkill] = {}

    def clone(self) -> "SkillSet":
        new_set = SkillSet()
        for k, v in self.skills.items():
            new_set.skills[k] = copy.deepcopy(v)
        return new_set

    def inject_skills_into_prompt(self) -> str:
        """Injects full content of active skills directly into system prompt (Section 3.2.1)."""
        if not self.skills:
            return ""
        sections = ["## Active Domain Skills"]
        for name, skill in sorted(self.skills.items()):
            sections.append(f"### Skill: {name}\n{skill.skill_content}\n")
        return "\n".join(sections)


@dataclass
class SkillProposal:
    """Atomic skill proposal synthesized by Skill Proposer (M_P)."""
    proposal_id: str
    action: str  # "create" or "patch"
    target_skill: str
    new_content: str
    diff_summary: str
    motivating_patterns: List[str]


def sample_stratified_traces(
    traces: List[ExecutionTrace],
    max_fails: int = 5,
    max_passes: int = 3,
    rng: Optional[random.Random] = None
) -> List[ExecutionTrace]:
    """
    Stratified trace sampling operator (Section 3.2.2 & Appendix C).
    Samples up to max_fails failing traces and up to max_passes passing traces (total <= 8).
    """
    if rng is None:
        rng = random.Random(42)

    fails = [t for t in traces if not t.success]
    passes = [t for t in traces if t.success]

    sampled_fails = rng.sample(fails, min(len(fails), max_fails)) if fails else []
    sampled_passes = rng.sample(passes, min(len(passes), max_passes)) if passes else []

    return sampled_fails + sampled_passes


class WikiMaintainerAgent:
    """
    Wiki Maintainer agent (M_WM) (Section 3.2.2).
    Consolidates observations from sampled traces into the persistent wiki.
    """

    def consolidate(
        self,
        wiki: PersistentWiki,
        sampled_traces: List[ExecutionTrace],
        iter_idx: int
    ) -> None:
        new_patterns_count = 0
        updated_patterns_count = 0

        for trace in sampled_traces:
            if not trace.success:
                # Diagnose root cause of failure
                pid = f"PAT-FAIL-{abs(hash(trace.prompt)) % 10000:04d}"
                pattern = WikiPattern(
                    pattern_id=pid,
                    title=f"Failure in {trace.task_id}",
                    category="failure_mode",
                    root_cause=f"Model failed during execution of: {trace.prompt[:100]}. Trace output: {trace.output[:150]}",
                    recommended_action="Introduce explicit step-by-step constraint check and error handling.",
                    evidence_traces=[trace.task_id],
                    iteration_created=iter_idx,
                    iteration_last_updated=iter_idx
                )
                if pid in wiki.patterns:
                    updated_patterns_count += 1
                else:
                    new_patterns_count += 1
                wiki.add_or_update_pattern(pattern)
            else:
                # Extract successful strategy to prevent regressions
                pid = f"PAT-PASS-{abs(hash(trace.prompt)) % 10000:04d}"
                pattern = WikiPattern(
                    pattern_id=pid,
                    title=f"Verified Solution Strategy for {trace.task_id}",
                    category="successful_strategy",
                    root_cause="None (Task succeeded).",
                    recommended_action=f"Preserve workflow: {trace.output[:150]}",
                    evidence_traces=[trace.task_id],
                    iteration_created=iter_idx,
                    iteration_last_updated=iter_idx
                )
                if pid in wiki.patterns:
                    updated_patterns_count += 1
                else:
                    new_patterns_count += 1
                wiki.add_or_update_pattern(pattern)

        summary = (
            f"Analyzed {len(sampled_traces)} sampled traces. "
            f"Created {new_patterns_count} new patterns, updated {updated_patterns_count} existing patterns. "
            f"Active pattern catalog size: {len(wiki.patterns)}."
        )
        wiki.log_iteration(iter_idx, summary)


class SkillProposerAgent:
    """
    Wiki-Informed Skill Proposer agent (M_P) (Section 3.2.3).
    Inspects persistent wiki patterns and execution history to synthesize atomic skill proposals.
    """

    def propose(
        self,
        wiki: PersistentWiki,
        current_skills: SkillSet,
        traces: List[ExecutionTrace],
        iter_idx: int
    ) -> Optional[SkillProposal]:
        rejected = wiki.get_rejected_proposals()

        # Prioritize recurring failure patterns
        sorted_patterns = sorted(
            wiki.patterns.values(),
            key=lambda p: (p.category == "failure_mode", p.recurrence_count),
            reverse=True
        )

        for pat in sorted_patterns:
            skill_name = f"skill_{pat.pattern_id.lower().replace('-', '_')}"
            diff_summary = f"Rule enforcement from {pat.pattern_id}"
            signature = f"{skill_name}:{diff_summary}"

            # Avoid repeating failed modifications (Historical Awareness Invariant)
            if signature in rejected:
                continue

            if skill_name in current_skills.skills:
                # Incremental patch
                existing = current_skills.skills[skill_name]
                new_content = existing.skill_content + f"\n- Additional Rule (Iter {iter_idx}): {pat.recommended_action}"
                return SkillProposal(
                    proposal_id=f"PROP-{iter_idx:03d}",
                    action="patch",
                    target_skill=skill_name,
                    new_content=new_content,
                    diff_summary=diff_summary,
                    motivating_patterns=[pat.pattern_id]
                )
            else:
                # Create new skill
                content = (
                    f"# Skill: {skill_name}\n\n"
                    f"## Purpose\nMitigate failure patterns identified in wiki.\n\n"
                    f"## Instructions\n1. Check inputs thoroughly.\n2. {pat.recommended_action}\n"
                )
                return SkillProposal(
                    proposal_id=f"PROP-{iter_idx:03d}",
                    action="create",
                    target_skill=skill_name,
                    new_content=content,
                    diff_summary=diff_summary,
                    motivating_patterns=[pat.pattern_id]
                )

        return None


def apply_proposal(skills: SkillSet, proposal: SkillProposal) -> SkillSet:
    """Applies atomic proposal to generate candidate skill set S'_k."""
    candidate = skills.clone()
    if proposal.action == "create":
        candidate.skills[proposal.target_skill] = AgentSkill(
            name=proposal.target_skill,
            skill_content=proposal.new_content,
            purpose_content=f"Motivated by patterns: {', '.join(proposal.motivating_patterns)}",
            motivating_patterns=proposal.motivating_patterns,
            version=1
        )
    elif proposal.action == "patch":
        if proposal.target_skill in candidate.skills:
            skill = candidate.skills[proposal.target_skill]
            skill.skill_content = proposal.new_content
            skill.version += 1
            for p in proposal.motivating_patterns:
                if p not in skill.motivating_patterns:
                    skill.motivating_patterns.append(p)
    return candidate


class WikiSkillEvolutionLoop:
    """
    Complete WikiSkill Evolution Loop (Algorithm 1, Appendix A.1).
    Orchestrates training rollouts, stratified sampling, wiki consolidation,
    skill proposal, validation gating, and compounding knowledge retention.
    """

    def __init__(self, val_evaluator_fn):
        self.val_evaluator_fn = val_evaluator_fn
        self.wiki_maintainer = WikiMaintainerAgent()
        self.skill_proposer = SkillProposerAgent()

    def run_evolution(
        self,
        training_rollout_fn,
        num_iterations: int = 5,
        target_score: float = 1.0
    ) -> Tuple[SkillSet, PersistentWiki, float]:
        # Line 1: Initialize S_0 <- empty, W_0 <- empty
        active_skills = SkillSet()
        wiki = PersistentWiki()

        # Line 2: Baseline validation
        r_best = self.val_evaluator_fn(active_skills)

        for k in range(1, num_iterations + 1):
            # Line 4: Early termination check
            if r_best >= target_score:
                break

            # Line 7: Inference rollouts on training set
            t_train = training_rollout_fn(active_skills)

            # Line 8: Stratified trace sampling
            t_sample = sample_stratified_traces(t_train, max_fails=5, max_passes=3)

            # Line 9: Wiki Maintenance
            self.wiki_maintainer.consolidate(wiki, t_sample, iter_idx=k)

            # Line 10: Skill Proposal
            proposal = self.skill_proposer.propose(wiki, active_skills, t_train, iter_idx=k)
            if proposal is None:
                continue

            # Line 11: Apply proposal
            candidate_skills = apply_proposal(active_skills, proposal)

            # Line 12: Validate candidate skills
            val_score = self.val_evaluator_fn(candidate_skills)

            # Line 13-17: Gating and Rollback
            if val_score > r_best:
                outcome = "Accepted"
                active_skills = candidate_skills
                r_best = val_score
            else:
                outcome = "Rejected"
                # Candidate skills rolled back (active_skills remains S_{k-1}); wiki retained!

            # Line 18: Update Wiki Log (skill-impact.md)
            impact_record = SkillImpactRecord(
                iteration=k,
                proposal_id=proposal.proposal_id,
                target_skill=proposal.target_skill,
                action=proposal.action,
                diff_summary=proposal.diff_summary,
                val_score=val_score,
                previous_best_score=r_best if outcome == "Accepted" else r_best,
                outcome=outcome
            )
            wiki.record_impact(impact_record)

        return active_skills, wiki, r_best


# ==============================================================================
# Complexity & Statistical Significance Analysis Models
# ==============================================================================

class OptimizerComplexityModel:
    """
    Formal Optimizer API Call Complexity (Table 7 & Appendix D.2).
    Quantifies the number of optimizer LLM calls per evolution iteration across frameworks.
    """

    @staticmethod
    def wikiskill_complexity(n_train: int, batch_size: int, t_react: int) -> float:
        """C_WikiSkill = (1 + T_ReAct) * (N_train / B). When B = N_train, C = 1 + T_ReAct."""
        steps = math.ceil(n_train / batch_size)
        return float((1 + t_react) * steps)

    @staticmethod
    def trace2skill_complexity(n_train: int, batch_size: int, branch_factor: int) -> float:
        """C_Trace2Skill = N_train + (1 + 1/(c-1)) * (N_train / B) + 1."""
        batches = math.ceil(n_train / batch_size)
        reduction_factor = 1.0 + 1.0 / (branch_factor - 1)
        return float(n_train + reduction_factor * batches + 1)

    @staticmethod
    def evoskill_complexity(n_train: int, batch_size: int) -> float:
        """C_EvoSkill = 2 * (N_train / B)."""
        batches = math.ceil(n_train / batch_size)
        return float(2 * batches)

    @staticmethod
    def skillopt_complexity(n_train: int, batch_size: int, k_opt: int) -> float:
        """C_SkillOpt = K_opt * (N_train / B)."""
        batches = math.ceil(n_train / batch_size)
        return float(k_opt * batches)


def paired_bootstrap_test(
    scores_a: List[float],
    scores_b: List[float],
    num_iterations: int = 1000,
    alpha: float = 0.05,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Paired bootstrap significance test with paired resampling (Appendix C).
    Tests null hypothesis H0: E[score_a] <= E[score_b].
    Returns empirical p-value, mean delta, 95% confidence interval, and whether a tie occurs.
    """
    if len(scores_a) != len(scores_b):
        raise ValueError(f"Sample lengths must match for paired bootstrap: {len(scores_a)} vs {len(scores_b)}")

    n = len(scores_a)
    rng = random.Random(seed)
    deltas = [a - b for a, b in zip(scores_a, scores_b)]
    observed_mean_delta = sum(deltas) / n

    bootstrap_deltas = []
    for _ in range(num_iterations):
        sample = [rng.choice(deltas) for _ in range(n)]
        bootstrap_deltas.append(sum(sample) / n)

    bootstrap_deltas.sort()
    # Empirical p-value: fraction of bootstrap samples where candidate margin <= 0
    p_value = sum(1 for d in bootstrap_deltas if d <= 0) / num_iterations

    # 95% two-sided confidence interval
    lower_idx = int((alpha / 2.0) * num_iterations)
    upper_idx = int((1.0 - alpha / 2.0) * num_iterations)
    ci_lower = bootstrap_deltas[lower_idx]
    ci_upper = bootstrap_deltas[upper_idx]

    is_statistically_significant = (p_value < alpha) and (ci_lower > 0)
    is_statistical_tie = not is_statistically_significant

    return {
        "observed_mean_delta": observed_mean_delta,
        "p_value": p_value,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "is_significant": is_statistically_significant,
        "is_statistical_tie": is_statistical_tie
    }
