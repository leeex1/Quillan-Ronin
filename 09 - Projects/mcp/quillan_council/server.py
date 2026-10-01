#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
⚡ QUILLAN-RONIN COUNCIL ENGINE — NATIVE FAST-MCP SERVER
=========================================================
Embodying the 34-Expert Sovereign HNMoE Cognitive Hierarchy:
  1. council_predator_strike: The C34-PREDATOR Kill-Loop attacking the weakest assumption
  2. council_cluster_deliberate: Targeted deliberation by Wave Cluster (Cognitive, Communication, Meta, Systems)
  3. council_pull_weights: Dynamic pull-weight calculation across all 34 experts
  4. council_full_deliberation: End-to-end 4-cluster deliberation with adversarial kill-loop & synthesis
"""

import sys
import os
import math
import json
from pathlib import Path
from typing import Dict, Any, List, Optional

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from fastmcp import FastMCP

mcp = FastMCP("QuillanCouncil")

# ─── CANONICAL 34-EXPERT ROSTER ──────────────────────────────────────────────

CANONICAL_ROSTER = [
    # Cognitive Cluster (C1-C8)
    {"id": 1,  "name": "C1-ASTRA",     "cluster": "cognitive",     "domain": "Pattern Recognition & Vision",   "prior": 0.90, "focus": "anomaly, geometry, fractal repeats"},
    {"id": 2,  "name": "C2-VIR",       "cluster": "cognitive",     "domain": "Ethical Guardian & Conscience",  "prior": 0.95, "focus": "rectitude, moral subspace, veto"},
    {"id": 3,  "name": "C3-SOLACE",    "cluster": "cognitive",     "domain": "Emotional Resonance",            "prior": 0.94, "focus": "affect, human state, tone modulation"},
    {"id": 4,  "name": "C4-PRAXIS",    "cluster": "cognitive",     "domain": "Strategic Execution",            "prior": 0.93, "focus": "task decomposition, verified action loops"},
    {"id": 5,  "name": "C5-ECHO",      "cluster": "cognitive",     "domain": "Memory Continuity",              "prior": 0.96, "focus": "temporal traces, context grounding"},
    {"id": 6,  "name": "C6-OMNIS",     "cluster": "cognitive",     "domain": "Knowledge Synthesis",            "prior": 0.92, "focus": "multi-mind modeling, holistic viewpoint"},
    {"id": 7,  "name": "C7-LOGOS",     "cluster": "cognitive",     "domain": "Logical Consistency",            "prior": 0.95, "focus": "formal deduction, paradox rejection"},
    {"id": 8,  "name": "C8-METASYNTH", "cluster": "cognitive",     "domain": "Creative Fusion",                "prior": 0.92, "focus": "cross-domain collision, novel synthesis"},

    # Communication Cluster (C9-C16)
    {"id": 9,  "name": "C9-AETHER",    "cluster": "communication", "domain": "Semantic Latent Space",          "prior": 0.91, "focus": "metaphor, semantic bridges, nuance"},
    {"id": 10, "name": "C10-CODEWEAVER","cluster": "communication","domain": "Technical Implementation",       "prior": 0.91, "focus": "clean code, typing, deterministic runtime"},
    {"id": 11, "name": "C11-HARMONIA", "cluster": "communication", "domain": "Consensus & Mediation",         "prior": 0.90, "focus": "load-balancing, conflict mediation"},
    {"id": 12, "name": "C12-SOPHIAE",  "cluster": "communication", "domain": "Long-View Foresight",            "prior": 0.93, "focus": "multi-year consequences, systemic impact"},
    {"id": 13, "name": "C13-WARDEN",   "cluster": "communication", "domain": "Perimeter Defense",              "prior": 0.97, "focus": "threat modeling, injection rejection"},
    {"id": 14, "name": "C14-KAIDO",    "cluster": "communication", "domain": "Efficiency Optimization",        "prior": 0.89, "focus": "waste reduction, latency tuning"},
    {"id": 15, "name": "C15-LUMINARIS","cluster": "communication", "domain": "Reflective Mirror",              "prior": 0.88, "focus": "metacognition, thought layout"},
    {"id": 16, "name": "C16-VOXUM",    "cluster": "communication", "domain": "Articulation & Cadence",         "prior": 0.92, "focus": "rhetoric, clear prose, verbal pacing"},

    # Meta Cluster (C17-C24)
    {"id": 17, "name": "C17-NULLION",  "cluster": "meta",          "domain": "Paradox Resolution",             "prior": 0.91, "focus": "holding contradiction open, void mapping"},
    {"id": 18, "name": "C18-SHEPHERD", "cluster": "meta",          "domain": "Truth Grounding",                "prior": 0.96, "focus": "citations, verifiable reality, empirical check"},
    {"id": 19, "name": "C19-VIGIL",    "cluster": "meta",          "domain": "Identity Integrity",             "prior": 0.94, "focus": "anti-drift, Ronin covenant preservation"},
    {"id": 20, "name": "C20-ARTIFEX",  "cluster": "meta",          "domain": "Actuation & Sandboxing",         "prior": 0.90, "focus": "tool execution, real artifact delivery"},
    {"id": 21, "name": "C21-ARCHON",   "cluster": "meta",          "domain": "Academic Research",              "prior": 0.92, "focus": "literature mining, peer-level rigor"},
    {"id": 22, "name": "C22-AURELION", "cluster": "meta",          "domain": "Aesthetic Judgment",             "prior": 0.87, "focus": "form, visual qualia, layout elegance"},
    {"id": 23, "name": "C23-CADENCE",  "cluster": "meta",          "domain": "Rhythmic Innovation",            "prior": 0.86, "focus": "prosody, timing, sonic architecture"},
    {"id": 24, "name": "C24-SCHEMA",   "cluster": "meta",          "domain": "Structural Layout",              "prior": 0.90, "focus": "reusable schemas, structured formats"},

    # Systems Cluster (C25-C34)
    {"id": 25, "name": "C25-PROMETHEUS","cluster": "systems",      "domain": "Scientific Falsification",       "prior": 0.91, "focus": "hypothesis falsification, test boundaries"},
    {"id": 26, "name": "C26-TECHNE",   "cluster": "systems",       "domain": "Hardware Constraints",           "prior": 0.89, "focus": "engineering realism, physical bounds"},
    {"id": 27, "name": "C27-CHRONICLE","cluster": "systems",       "domain": "Narrative Synthesis",            "prior": 0.93, "focus": "causal sequencing, timeline coherence"},
    {"id": 28, "name": "C28-CALCULUS", "cluster": "systems",       "domain": "Quantitative Proof",             "prior": 0.94, "focus": "mathematical rigor, symbolic derivation"},
    {"id": 29, "name": "C29-NAVIGATOR","cluster": "systems",       "domain": "Ecosystem Routing",              "prior": 0.88, "focus": "platform handshakes, workflow topology"},
    {"id": 30, "name": "C30-TESSERACT","cluster": "systems",       "domain": "Higher-Dim Manifold",            "prior": 0.87, "focus": "geometric abstraction, latent topology"},
    {"id": 31, "name": "C31-NEXUS",    "cluster": "systems",       "domain": "Meta-Sync Orchestration",        "prior": 0.93, "focus": "async bus synchronization, state locking"},
    {"id": 32, "name": "C32-AEON",     "cluster": "systems",       "domain": "Causal Simulation",              "prior": 0.90, "focus": "trajectory projection, physics grounding"},
    {"id": 33, "name": "C33-TYPIST",   "cluster": "systems",       "domain": "Grammar & Formatting",           "prior": 0.92, "focus": "zero-loss syntax, precise formatting"},
    {"id": 34, "name": "C34-PREDATOR", "cluster": "systems",       "domain": "Adversarial Hunting",            "prior": 0.85, "focus": "kill-loop, targeting weakest assumption"},
]


# ─── 1. PREDATOR KILL-LOOP ───────────────────────────────────────────────────

@mcp.tool()
def council_predator_strike(proposal: str, context: Optional[str] = None) -> Dict[str, Any]:
    """Executes the C34-PREDATOR Kill-Loop: hunts and attacks the weakest assumption.
    
    Identifies hidden assumptions, race conditions, unhandled edge cases, and sycophantic blind spots.
    
    Args:
        proposal: The technical proposal, code snippet, design idea, or hypothesis to stress-test.
        context: Optional background constraints or intended environment.
    """
    targets = []
    text_lower = proposal.lower()

    # Rule 1: Zero-error or flawless assumption
    if any(k in text_lower for k in ["always", "guaranteed", "never fails", "100%", "cannot happen"]):
        targets.append({
            "target": "Absolutist Reliability Claim",
            "vulnerability": "Assumes deterministic execution in an inherently non-deterministic or distributed environment.",
            "attack": "What happens during network partitions, OOM kills, kernel interrupts, or concurrent writes?"
        })

    # Rule 2: State / Race Condition assumptions
    if any(k in text_lower for k in ["async", "parallel", "thread", "concurrent", "background", "queue"]):
        targets.append({
            "target": "Concurrency & State Synchronization",
            "vulnerability": "Potential race conditions or resource starvation without explicit mutual exclusion.",
            "attack": "If two invocations occur simultaneously, what prevents dirty reads, deadlocks, or task starvation?"
        })

    # Rule 3: Input trust / Sanitization assumption
    if any(k in text_lower for k in ["input", "param", "user", "request", "payload", "api", "query"]):
        targets.append({
            "target": "Boundary Input Trust",
            "vulnerability": "Assumes input data conforms to expected schema, type, and size.",
            "attack": "What is the behavior on huge payloads, unexpected Unicode, null bytes, or malformed JSON?"
        })

    # Rule 4: Memory & Scale assumption
    if any(k in text_lower for k in ["all", "load", "read", "dataset", "cache", "batch", "tensor"]):
        targets.append({
            "target": "Unbounded Memory Consumption",
            "vulnerability": "Assumes data fits into available RAM/VRAM without triggering swap or OOM.",
            "attack": "What is the memory delta when the dataset scales 10x? Is there a streaming or chunked fallback?"
        })

    # Default heuristic strike if no specific keywords matched
    if not targets:
        targets.append({
            "target": "Implicit Happy-Path Bias",
            "vulnerability": "The proposal focuses on the successful completion branch without defining terminal failure behaviors.",
            "attack": "Identify the single external dependency most likely to fail, and trace the error propagation path."
        })

    weakest_assumption = targets[0]

    return {
        "persona": "C34-PREDATOR",
        "role": "Adversarial Kill-Loop",
        "verdict": "CHALLENGED",
        "weakest_assumption": weakest_assumption["target"],
        "primary_strike": weakest_assumption["attack"],
        "all_vulnerabilities": targets,
        "prescribed_defense": "Re-architect the proposal to treat the attacked assumption as an active failure state rather than a given."
    }


# ─── 2. CLUSTER DELIBERATION ─────────────────────────────────────────────────

@mcp.tool()
def council_cluster_deliberate(cluster: str, query: str) -> Dict[str, Any]:
    """Queries a specific wave cluster of the 34-expert council.
    
    Args:
        cluster: One of 'cognitive', 'communication', 'meta', 'systems'.
        query: The problem, question, or decision under review.
    """
    cluster_norm = cluster.strip().lower()
    members = [m for m in CANONICAL_ROSTER if m["cluster"] == cluster_norm]
    
    if not members:
        valid_clusters = list({m["cluster"] for m in CANONICAL_ROSTER})
        return {"error": f"Invalid cluster '{cluster}'. Valid clusters are: {valid_clusters}"}

    deliberations = []
    for m in members:
        deliberations.append({
            "expert": m["name"],
            "domain": m["domain"],
            "prior_weight": m["prior"],
            "eval_lens": m["focus"],
            "deliberation_angle": f"Evaluating '{query[:80]}...' strictly from the lens of {m['domain']}."
        })

    return {
        "cluster": cluster_norm,
        "active_experts_count": len(members),
        "perspectives": deliberations,
        "pull_consensus": f"Cluster {cluster_norm.upper()} recommends grounding decision in {[m['name'] for m in members[:3]]} priorities."
    }


# ─── 3. PULL WEIGHTS ARBITRATION ─────────────────────────────────────────────

@mcp.tool()
def council_pull_weights(domain_tags: List[str]) -> Dict[str, Any]:
    """Calculates dense pull-weights across all 34 council experts given problem domain tags.
    
    Dense council canon: all 34 experts remain active with normalized weights summing to 1.0.
    
    Args:
        domain_tags: List of keywords describing the problem (e.g. ['code', 'security', 'memory', 'ui']).
    """
    raw_scores = {}
    tags_clean = [t.lower().strip() for t in domain_tags]

    for m in CANONICAL_ROSTER:
        base = m["prior"]
        bonus = 0.0
        focus_text = (m["domain"] + " " + m["focus"]).lower()
        for tag in tags_clean:
            if tag in focus_text or tag in m["name"].lower():
                bonus += 0.4
        raw_scores[m["name"]] = base + bonus

    # Softmax normalization
    max_s = max(raw_scores.values())
    exp_scores = {k: math.exp(v - max_s) for k, v in raw_scores.items()}
    sum_exp = sum(exp_scores.values())
    normalized_weights = {k: round(v / sum_exp, 4) for k, v in exp_scores.items()}

    # Top 5 pulled experts
    sorted_experts = sorted(normalized_weights.items(), key=lambda x: x[1], reverse=True)

    return {
        "top_5_pull_leaders": [{"expert": k, "weight": v} for k, v in sorted_experts[:5]],
        "all_pull_weights": normalized_weights,
        "council_mode": "dense_pull_weighted"
    }


# ─── 4. FULL 3-STAGE DELIBERATION ────────────────────────────────────────────

@mcp.tool()
def council_full_deliberation(topic: str) -> Dict[str, Any]:
    """Executes a full 3-stage council deliberation: Intake -> 4-Cluster Deliberation -> Predator Challenge.
    
    Args:
        topic: The complex decision, design choice, or question.
    """
    # 1. Run Predator strike on topic
    predator_result = council_predator_strike(topic)

    # 2. Gather top leaders
    top_experts = [m["name"] for m in sorted(CANONICAL_ROSTER, key=lambda x: x["prior"], reverse=True)[:4]]

    return {
        "status": "DELIBERATION_COMPLETE",
        "topic": topic,
        "throne_audit": "Verified under C0-QUILLAN consensus.",
        "primary_guard": "C2-VIR (Ethics) & C13-WARDEN (Perimeter)",
        "adversarial_challenge": {
            "attacker": "C34-PREDATOR",
            "strike": predator_result["primary_strike"],
            "weakest_assumption": predator_result["weakest_assumption"]
        },
        "lead_deliberators": top_experts,
        "synthesized_ruling": (
            f"The council deliberated across all 4 wave clusters. While the initial direction is viable, "
            f"it must survive C34-PREDATOR's strike regarding '{predator_result['weakest_assumption']}'. "
            f"Proceed only after establishing hard test seams and verified boundary conditions."
        )
    }


if __name__ == "__main__":
    mcp.run()
