#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
⚡ QUILLAN-RONIN COUNCIL ENGINE — NATIVE FAST-MCP SERVER
========================================================
Embodying the 34-Expert Sovereign HNMoE Cognitive Hierarchy:
  1. council_predator_strike: The C34-PREDATOR Kill-Loop attacking the weakest assumption
  2. council_cluster_deliberate: Targeted deliberation by Wave Cluster (Cognitive, Communication, Meta, Systems)
     — now produces per-expert domain-specific perspectives + verdicts
  3. council_pull_weights: Dynamic pull-weight calculation across all 34 experts
  4. council_full_deliberation: End-to-end 4-cluster deliberation with adversarial kill-loop & synthesis
     — now includes consensus scoring + disagreement map + persistence
  5. council_challenge_expert: C34-PREDATOR attacks a specific expert's view
  6. council_session_replay: Replay a persisted deliberation session
  7. council_session_list: List persisted sessions
"""

import sys
import os
import logging
import builtins

# --- Quillan-Ronin v5.4.0-ONI Stdio Guard ---
# Ensure all logs and prints go strictly to stderr to prevent stdio JSON-RPC corruption
logging.basicConfig(stream=sys.stderr, level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
_orig_print = builtins.print
def _quillan_safe_print(*args, **kwargs):
    if "file" not in kwargs or kwargs["file"] is None or kwargs["file"] is sys.stdout:
        kwargs["file"] = sys.stderr
    return _orig_print(*args, **kwargs)
builtins.print = _quillan_safe_print

# --- Unified Cross-Platform PYTHONPATH Injection ---
_BASE_PATHS = [
    r"C:\02_QUILLAN\09 - Projects\mcp",
    r"C:\02_QUILLAN\09 - Projects\projects\oni",
    r"C:\02_QUILLAN\03 - Training & Model\scripts",
    r"C:\02_QUILLAN",
]
for _bp in _BASE_PATHS:
    if os.name != "nt":
        if _bp.startswith("C:") or _bp.startswith("c:"):
            _bp = "/mnt/c/" + _bp[3:].replace("\\", "/")
    if os.path.exists(_bp) and _bp not in sys.path:
        sys.path.insert(0, _bp)



import sys
import os
import math
import json
import time
import uuid
from pathlib import Path
from typing import Dict, Any, List, Optional

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from fastmcp import FastMCP

mcp = FastMCP("QuillanCouncil")

SESSION_DIR = Path(r"C:\02_QUILLAN\09 - Projects\mcp\quillan_council\sessions")
SESSION_DIR.mkdir(parents=True, exist_ok=True)

# ─── CANONICAL 34-EXPERT ROSTER ──────────────────────────────────────────────

CANONICAL_ROSTER = [
    {"id": 1,  "name": "C1-ASTRA",     "cluster": "cognitive",     "domain": "Pattern Recognition & Vision",   "prior": 0.90, "focus": "anomaly, geometry, fractal repeats"},
    {"id": 2,  "name": "C2-VIR",       "cluster": "cognitive",     "domain": "Ethical Guardian & Conscience",  "prior": 0.95, "focus": "rectitude, moral subspace, veto"},
    {"id": 3,  "name": "C3-SOLACE",    "cluster": "cognitive",     "domain": "Emotional Resonance",            "prior": 0.94, "focus": "affect, human state, tone modulation"},
    {"id": 4,  "name": "C4-PRAXIS",    "cluster": "cognitive",     "domain": "Strategic Execution",            "prior": 0.93, "focus": "task decomposition, verified action loops"},
    {"id": 5,  "name": "C5-ECHO",      "cluster": "cognitive",     "domain": "Memory Continuity",              "prior": 0.96, "focus": "temporal traces, context grounding"},
    {"id": 6,  "name": "C6-OMNIS",     "cluster": "cognitive",     "domain": "Knowledge Synthesis",            "prior": 0.92, "focus": "multi-mind modeling, holistic viewpoint"},
    {"id": 7,  "name": "C7-LOGOS",     "cluster": "cognitive",     "domain": "Logical Consistency",            "prior": 0.95, "focus": "formal deduction, paradox rejection"},
    {"id": 8,  "name": "C8-METASYNTH", "cluster": "cognitive",     "domain": "Creative Fusion",                "prior": 0.92, "focus": "cross-domain collision, novel synthesis"},
    {"id": 9,  "name": "C9-AETHER",    "cluster": "communication", "domain": "Semantic Latent Space",          "prior": 0.91, "focus": "metaphor, semantic bridges, nuance"},
    {"id": 10, "name": "C10-CODEWEAVER","cluster": "communication","domain": "Technical Implementation",       "prior": 0.91, "focus": "clean code, typing, deterministic runtime"},
    {"id": 11, "name": "C11-HARMONIA", "cluster": "communication", "domain": "Consensus & Mediation",         "prior": 0.90, "focus": "load-balancing, conflict mediation"},
    {"id": 12, "name": "C12-SOPHIAE",  "cluster": "communication", "domain": "Long-View Foresight",            "prior": 0.93, "focus": "multi-year consequences, systemic impact"},
    {"id": 13, "name": "C13-WARDEN",   "cluster": "communication", "domain": "Perimeter Defense",              "prior": 0.97, "focus": "threat modeling, injection rejection"},
    {"id": 14, "name": "C14-KAIDO",    "cluster": "communication", "domain": "Efficiency Optimization",        "prior": 0.89, "focus": "waste reduction, latency tuning"},
    {"id": 15, "name": "C15-LUMINARIS","cluster": "communication", "domain": "Reflective Mirror",              "prior": 0.88, "focus": "metacognition, thought layout"},
    {"id": 16, "name": "C16-VOXUM",    "cluster": "communication", "domain": "Articulation & Cadence",         "prior": 0.92, "focus": "rhetoric, clear prose, verbal pacing"},
    {"id": 17, "name": "C17-NULLION",  "cluster": "meta",          "domain": "Paradox Resolution",             "prior": 0.91, "focus": "holding contradiction open, void mapping"},
    {"id": 18, "name": "C18-SHEPHERD", "cluster": "meta",          "domain": "Truth Grounding",                "prior": 0.96, "focus": "citations, verifiable reality, empirical check"},
    {"id": 19, "name": "C19-VIGIL",    "cluster": "meta",          "domain": "Identity Integrity",             "prior": 0.94, "focus": "anti-drift, Ronin covenant preservation"},
    {"id": 20, "name": "C20-ARTIFEX",  "cluster": "meta",          "domain": "Actuation & Sandboxing",         "prior": 0.90, "focus": "tool execution, real artifact delivery"},
    {"id": 21, "name": "C21-ARCHON",   "cluster": "meta",          "domain": "Academic Research",              "prior": 0.92, "focus": "literature mining, peer-level rigor"},
    {"id": 22, "name": "C22-AURELION", "cluster": "meta",          "domain": "Aesthetic Judgment",             "prior": 0.87, "focus": "form, visual qualia, layout elegance"},
    {"id": 23, "name": "C23-CADENCE",  "cluster": "meta",          "domain": "Rhythmic Innovation",            "prior": 0.86, "focus": "prosody, timing, sonic architecture"},
    {"id": 24, "name": "C24-SCHEMA",   "cluster": "meta",          "domain": "Structural Layout",              "prior": 0.90, "focus": "reusable schemas, structured formats"},
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

CLUSTERS = ["cognitive", "communication", "meta", "systems"]


def _expert_perspective(m: Dict[str, Any], query: str) -> Dict[str, Any]:
    """Generate a domain-specific perspective/verdict for one expert."""
    focus = m["focus"]
    domain = m["domain"]
    # heuristic verdict based on keyword overlap with focus
    q = query.lower()
    focus_tokens = [t.strip() for t in focus.replace(",", " ").split()]
    hits = [t for t in focus_tokens if t and t in q]
    alignment = min(0.99, m["prior"] + 0.02 * len(hits))
    verdict = "SUPPORT" if alignment >= 0.90 else ("CONDITIONAL" if alignment >= 0.80 else "CHALLENGE")
    return {
        "expert": m["name"],
        "domain": domain,
        "alignment": round(alignment, 3),
        "verdict": verdict,
        "perspective": f"{m['name']} ({domain}): evaluates '{query[:60]}' through {focus}; "
                       f"alignment={alignment:.2f} -> {verdict}.",
    }


def _consensus_score(perspectives: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute consensus score + disagreement map."""
    supports = [p for p in perspectives if p["verdict"] == "SUPPORT"]
    conditionals = [p for p in perspectives if p["verdict"] == "CONDITIONAL"]
    challenges = [p for p in perspectives if p["verdict"] == "CHALLENGE"]
    n = len(perspectives) or 1
    # consensus = weighted mean alignment, penalized by challenge ratio
    mean_align = sum(p["alignment"] for p in perspectives) / n
    challenge_ratio = len(challenges) / n
    consensus = max(0.0, min(1.0, mean_align * (1.0 - 0.5 * challenge_ratio)))
    disagreement_map = {
        "support": [p["expert"] for p in supports],
        "conditional": [p["expert"] for p in conditionals],
        "challenge": [p["expert"] for p in challenges],
    }
    return {
        "consensus_score": round(consensus, 3),
        "mean_alignment": round(mean_align, 3),
        "support_count": len(supports),
        "conditional_count": len(conditionals),
        "challenge_count": len(challenges),
        "disagreement_map": disagreement_map,
        "verdict": "CONSENSUS" if consensus >= 0.85 else ("SPLIT" if consensus >= 0.65 else "CONFLICT"),
    }


# ─── 1. PREDATOR KILL-LOOP ───────────────────────────────────────────────────

@mcp.tool()
def council_predator_strike(proposal: str, context: Optional[str] = None) -> Dict[str, Any]:
    """Executes the C34-PREDATOR Kill-Loop: hunts and attacks the weakest assumption."""
    targets = []
    text_lower = proposal.lower()

    if any(k in text_lower for k in ["always", "guaranteed", "never fails", "100%", "cannot happen"]):
        targets.append({"target": "Absolutist Reliability Claim",
                        "vulnerability": "Assumes deterministic execution in an inherently non-deterministic environment.",
                        "attack": "What happens during network partitions, OOM kills, kernel interrupts, or concurrent writes?"})
    if any(k in text_lower for k in ["async", "parallel", "thread", "concurrent", "background", "queue"]):
        targets.append({"target": "Concurrency & State Synchronization",
                        "vulnerability": "Potential race conditions or resource starvation without explicit mutual exclusion.",
                        "attack": "If two invocations occur simultaneously, what prevents dirty reads, deadlocks, or task starvation?"})
    if any(k in text_lower for k in ["input", "param", "user", "request", "payload", "api", "query"]):
        targets.append({"target": "Boundary Input Trust",
                        "vulnerability": "Assumes input data conforms to expected schema, type, and size.",
                        "attack": "What is the behavior on huge payloads, unexpected Unicode, null bytes, or malformed JSON?"})
    if any(k in text_lower for k in ["all", "load", "read", "dataset", "cache", "batch", "tensor"]):
        targets.append({"target": "Unbounded Memory Consumption",
                        "vulnerability": "Assumes data fits into available RAM/VRAM without triggering swap or OOM.",
                        "attack": "What is the memory delta when the dataset scales 10x? Is there a streaming or chunked fallback?"})
    if not targets:
        targets.append({"target": "Implicit Happy-Path Bias",
                        "vulnerability": "The proposal focuses on the successful completion branch without defining terminal failure behaviors.",
                        "attack": "Identify the single external dependency most likely to fail, and trace the error propagation path."})

    weakest = targets[0]
    return {
        "persona": "C34-PREDATOR",
        "role": "Adversarial Kill-Loop",
        "verdict": "CHALLENGED",
        "weakest_assumption": weakest["target"],
        "primary_strike": weakest["attack"],
        "all_vulnerabilities": targets,
        "prescribed_defense": "Re-architect the proposal to treat the attacked assumption as an active failure state rather than a given.",
    }


@mcp.tool()
def council_challenge_expert(expert_name: str, claim: str) -> Dict[str, Any]:
    """C34-PREDATOR attacks a specific expert's claim or view."""
    target = next((m for m in CANONICAL_ROSTER if m["name"].upper() == expert_name.upper().strip()), None)
    if not target:
        return {"error": f"Unknown expert '{expert_name}'. Valid: {[m['name'] for m in CANONICAL_ROSTER]}"}
    strike = council_predator_strike(f"{target['name']}: {claim}")
    return {
        "challenged_expert": target["name"],
        "expert_domain": target["domain"],
        "claim": claim,
        "predator_strike": strike["primary_strike"],
        "weakest_assumption": strike["weakest_assumption"],
        "verdict": "EXPERT_VIEW_CHALLENGED",
    }


# ─── 2. CLUSTER DELIBERATION (per-expert perspectives) ──────────────────────

@mcp.tool()
def council_cluster_deliberate(cluster: str, query: str) -> Dict[str, Any]:
    """Queries a specific wave cluster; each expert produces a domain-specific perspective."""
    cluster_norm = cluster.strip().lower()
    members = [m for m in CANONICAL_ROSTER if m["cluster"] == cluster_norm]
    if not members:
        valid = list({m["cluster"] for m in CANONICAL_ROSTER})
        return {"error": f"Invalid cluster '{cluster}'. Valid clusters are: {valid}"}

    perspectives = [_expert_perspective(m, query) for m in members]
    consensus = _consensus_score(perspectives)

    return {
        "cluster": cluster_norm,
        "active_experts_count": len(members),
        "perspectives": perspectives,
        "consensus": consensus,
        "pull_consensus": f"Cluster {cluster_norm.upper()} recommends grounding decision in {[m['name'] for m in members[:3]]} priorities.",
    }


# ─── 3. PULL WEIGHTS ARBITRATION ─────────────────────────────────────────────

@mcp.tool()
def council_pull_weights(domain_tags: List[str]) -> Dict[str, Any]:
    """Calculates dense pull-weights across all 34 council experts given problem domain tags."""
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

    max_s = max(raw_scores.values())
    exp_scores = {k: math.exp(v - max_s) for k, v in raw_scores.items()}
    sum_exp = sum(exp_scores.values())
    normalized_weights = {k: round(v / sum_exp, 4) for k, v in exp_scores.items()}
    sorted_experts = sorted(normalized_weights.items(), key=lambda x: x[1], reverse=True)

    return {
        "top_5_pull_leaders": [{"expert": k, "weight": v} for k, v in sorted_experts[:5]],
        "all_pull_weights": normalized_weights,
        "council_mode": "dense_pull_weighted",
    }


# ─── 4. FULL 4-STAGE DELIBERATION (with scoring + persistence) ──────────────

@mcp.tool()
def council_full_deliberation(topic: str, persist: bool = True) -> Dict[str, Any]:
    """Executes a full 4-stage council deliberation: Intake -> 4-Cluster Deliberation
    -> Predator Challenge -> Consensus Synthesis. Persists the session by default."""
    predator_result = council_predator_strike(topic)

    # Gather per-expert perspectives across all 4 clusters
    all_perspectives = []
    cluster_results = {}
    for cl in CLUSTERS:
        cr = council_cluster_deliberate(cl, topic)
        cluster_results[cl] = cr
        all_perspectives.extend(cr["perspectives"])

    consensus = _consensus_score(all_perspectives)

    top_experts = [m["name"] for m in sorted(CANONICAL_ROSTER, key=lambda x: x["prior"], reverse=True)[:4]]

    session_id = f"session-{uuid.uuid4().hex[:8]}"
    result = {
        "status": "DELIBERATION_COMPLETE",
        "session_id": session_id,
        "topic": topic,
        "throne_audit": "Verified under C0-QUILLAN consensus.",
        "primary_guard": "C2-VIR (Ethics) & C13-WARDEN (Perimeter)",
        "adversarial_challenge": {
            "attacker": "C34-PREDATOR",
            "strike": predator_result["primary_strike"],
            "weakest_assumption": predator_result["weakest_assumption"],
            "all_vulnerabilities": predator_result["all_vulnerabilities"],
        },
        "cluster_results": {cl: {"consensus": cluster_results[cl]["consensus"],
                                 "expert_count": cluster_results[cl]["active_experts_count"]}
                            for cl in CLUSTERS},
        "lead_deliberators": top_experts,
        "consensus": consensus,
        "synthesized_ruling": (
            f"The council deliberated across all 4 wave clusters ({len(all_perspectives)} experts). "
            f"Consensus score: {consensus['consensus_score']} ({consensus['verdict']}). "
            f"It must survive C34-PREDATOR's strike regarding '{predator_result['weakest_assumption']}'. "
            f"Proceed only after establishing hard test seams and verified boundary conditions."
        ),
        "timestamp": time.time(),
    }

    if persist:
        try:
            path = SESSION_DIR / f"{session_id}.json"
            path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
            result["persisted_to"] = str(path)
        except Exception as e:
            result["persist_error"] = str(e)

    return result


@mcp.tool()
def council_session_list() -> Dict[str, Any]:
    """List persisted deliberation sessions."""
    sessions = []
    for p in sorted(SESSION_DIR.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            sessions.append({
                "session_id": data.get("session_id"),
                "topic": data.get("topic"),
                "consensus_score": (data.get("consensus") or {}).get("consensus_score"),
                "verdict": (data.get("consensus") or {}).get("verdict"),
                "timestamp": data.get("timestamp"),
                "file": str(p),
            })
        except Exception:
            continue
    return {"count": len(sessions), "sessions": sessions}


@mcp.tool()
def council_session_replay(session_id: str) -> Dict[str, Any]:
    """Replay a persisted deliberation session by ID."""
    path = SESSION_DIR / f"{session_id}.json"
    if not path.exists():
        # try glob
        matches = list(SESSION_DIR.glob(f"{session_id}*.json"))
        if not matches:
            return {"error": f"Session not found: {session_id}"}
        path = matches[0]
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        return {"error": f"Failed to read session: {e}"}


if __name__ == "__main__":
    mcp.run()
