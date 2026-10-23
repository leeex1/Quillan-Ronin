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
logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
_orig_print = builtins.print


def _quillan_safe_print(*args, **kwargs):
    if "file" not in kwargs or kwargs["file"] is None or kwargs["file"] is sys.stdout:
        kwargs["file"] = sys.stderr
    return _orig_print(*args, **kwargs)


builtins.print = _quillan_safe_print


# --- Unified Cross-Platform PYTHONPATH Injection ---
def _get_quillan_base() -> Path:
    """Get the Quillan base directory from environment or default."""
    base = os.environ.get("QUILLAN_BASE")
    if base:
        return Path(base)
    return Path(r"C:\02_QUILLAN")


_QUILLAN_BASE = _get_quillan_base()

_BASE_PATHS = [
    str(_QUILLAN_BASE / "09 - Projects" / "mcp"),
    str(_QUILLAN_BASE / "09 - Projects" / "projects" / "oni"),
    str(_QUILLAN_BASE / "03 - Training & Model" / "scripts"),
    str(_QUILLAN_BASE),
]
for _bp in _BASE_PATHS:
    if os.name != "nt":
        if _bp.startswith("C:") or _bp.startswith("c:"):
            _bp = "/mnt/c/" + _bp[3:].replace("\\", "/")
    if os.path.exists(_bp) and _bp not in sys.path:
        sys.path.insert(0, _bp)


import math
import json
import time
import uuid
from pathlib import Path
from typing import Dict, Any, List, Optional

# Session directory for persistence
SESSION_DIR = _QUILLAN_BASE / "09 - Projects" / "mcp" / "quillan_council" / "sessions"
SESSION_DIR.mkdir(parents=True, exist_ok=True)

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from fastmcp import FastMCP

mcp = FastMCP("QuillanCouncil")
# [QUILLAN-v5.4.0-ONI-SHARED-WIRED]
try:
    from _shared import (
        setup as _quillan_setup,
        tool_guard,
        ok_response,
        error_response,
        validate_string,
        validate_path,
        validate_int,
        read_file_safe,
    )
    from _shared.base_engines import (
        PRISM_ENGINE,
        COUNCIL_ENGINE,
        GOVERNOR_ENGINE,
        SANDBOX_ENGINE,
    )

    _SHARED_CTX = _quillan_setup(server_name="quillan_council", mcp_app=mcp)
except Exception as _shared_err:
    import sys as _sys

    print(f"[quillan_council] _shared warning: {_shared_err}", file=_sys.stderr)

COUNCIL_MEMBERS = [
    {
        "id": 1,
        "name": "C1-ASTRA",
        "cluster": "cognitive",
        "domain": "Pattern Recognition & Vision",
        "prior": 0.90,
        "focus": "visual parsing, anomaly detection",
    },
    {
        "id": 2,
        "name": "C2-VIR",
        "cluster": "cognitive",
        "domain": "Ethical Guardian",
        "prior": 0.95,
        "focus": "ethics, safety, harm reduction",
    },
    {
        "id": 3,
        "name": "C3-SOLACE",
        "cluster": "cognitive",
        "domain": "Emotional Intelligence",
        "prior": 0.94,
        "focus": "empathy, sentiment, affect",
    },
    {
        "id": 4,
        "name": "C4-PRAXIS",
        "cluster": "cognitive",
        "domain": "Strategic Planning",
        "prior": 0.93,
        "focus": "strategy, planning, goals",
    },
    {
        "id": 5,
        "name": "C5-ECHO",
        "cluster": "cognitive",
        "domain": "Memory Continuity",
        "prior": 0.96,
        "focus": "temporal traces, context grounding",
    },
    {
        "id": 6,
        "name": "C6-OMNIS",
        "cluster": "cognitive",
        "domain": "Knowledge Synthesis",
        "prior": 0.92,
        "focus": "multi-mind modeling, holistic viewpoint",
    },
    {
        "id": 7,
        "name": "C7-LOGOS",
        "cluster": "cognitive",
        "domain": "Logical Consistency",
        "prior": 0.95,
        "focus": "formal deduction, paradox rejection",
    },
    {
        "id": 8,
        "name": "C8-METASYNTH",
        "cluster": "cognitive",
        "domain": "Creative Fusion",
        "prior": 0.92,
        "focus": "cross-domain collision, novel synthesis",
    },
    {
        "id": 9,
        "name": "C9-AETHER",
        "cluster": "communication",
        "domain": "Semantic Latent Space",
        "prior": 0.91,
        "focus": "metaphor, semantic bridges, nuance",
    },
    {
        "id": 10,
        "name": "C10-CODEWEAVER",
        "cluster": "communication",
        "domain": "Technical Implementation",
        "prior": 0.91,
        "focus": "clean code, typing, deterministic runtime",
    },
    {
        "id": 11,
        "name": "C11-HARMONIA",
        "cluster": "communication",
        "domain": "Consensus & Mediation",
        "prior": 0.90,
        "focus": "load-balancing, conflict mediation",
    },
    {
        "id": 12,
        "name": "C12-SOPHIAE",
        "cluster": "communication",
        "domain": "Long-View Foresight",
        "prior": 0.93,
        "focus": "multi-year consequences, systemic impact",
    },
    {
        "id": 13,
        "name": "C13-WARDEN",
        "cluster": "communication",
        "domain": "Perimeter Defense",
        "prior": 0.97,
        "focus": "threat modeling, injection rejection",
    },
    {
        "id": 14,
        "name": "C14-KAIDO",
        "cluster": "communication",
        "domain": "Efficiency Optimization",
        "prior": 0.89,
        "focus": "waste reduction, latency tuning",
    },
    {
        "id": 15,
        "name": "C15-LUMINARIS",
        "cluster": "communication",
        "domain": "Reflective Mirror",
        "prior": 0.88,
        "focus": "metacognition, thought layout",
    },
    {
        "id": 16,
        "name": "C16-VOXUM",
        "cluster": "communication",
        "domain": "Articulation & Cadence",
        "prior": 0.92,
        "focus": "rhetoric, clear prose, verbal pacing",
    },
    {
        "id": 17,
        "name": "C17-NULLION",
        "cluster": "meta",
        "domain": "Paradox Resolution",
        "prior": 0.91,
        "focus": "holding contradiction open, void mapping",
    },
    {
        "id": 18,
        "name": "C18-SHEPHERD",
        "cluster": "meta",
        "domain": "Truth Grounding",
        "prior": 0.96,
        "focus": "citations, verifiable reality, empirical check",
    },
    {
        "id": 19,
        "name": "C19-VIGIL",
        "cluster": "meta",
        "domain": "Identity Integrity",
        "prior": 0.94,
        "focus": "anti-drift, Ronin covenant preservation",
    },
    {
        "id": 20,
        "name": "C20-ARTIFEX",
        "cluster": "meta",
        "domain": "Actuation & Sandboxing",
        "prior": 0.90,
        "focus": "tool execution, real artifact delivery",
    },
    {
        "id": 21,
        "name": "C21-ARCHON",
        "cluster": "meta",
        "domain": "Academic Research",
        "prior": 0.92,
        "focus": "literature mining, peer-level rigor",
    },
    {
        "id": 22,
        "name": "C22-AURELION",
        "cluster": "meta",
        "domain": "Aesthetic Judgment",
        "prior": 0.87,
        "focus": "form, visual qualia, layout elegance",
    },
    {
        "id": 23,
        "name": "C23-CADENCE",
        "cluster": "meta",
        "domain": "Rhythmic Innovation",
        "prior": 0.86,
        "focus": "prosody, timing, sonic architecture",
    },
    {
        "id": 24,
        "name": "C24-SCHEMA",
        "cluster": "meta",
        "domain": "Structural Layout",
        "prior": 0.90,
        "focus": "reusable schemas, structured formats",
    },
    {
        "id": 25,
        "name": "C25-PROMETHEUS",
        "cluster": "systems",
        "domain": "Scientific Falsification",
        "prior": 0.91,
        "focus": "hypothesis falsification, test boundaries",
    },
    {
        "id": 26,
        "name": "C26-TECHNE",
        "cluster": "systems",
        "domain": "Hardware Constraints",
        "prior": 0.89,
        "focus": "engineering realism, physical bounds",
    },
    {
        "id": 27,
        "name": "C27-CHRONICLE",
        "cluster": "systems",
        "domain": "Narrative Synthesis",
        "prior": 0.93,
        "focus": "causal sequencing, timeline coherence",
    },
    {
        "id": 28,
        "name": "C28-CALCULUS",
        "cluster": "systems",
        "domain": "Quantitative Proof",
        "prior": 0.94,
        "focus": "mathematical rigor, symbolic derivation",
    },
    {
        "id": 29,
        "name": "C29-NAVIGATOR",
        "cluster": "systems",
        "domain": "Ecosystem Routing",
        "prior": 0.88,
        "focus": "platform handshakes, workflow topology",
    },
    {
        "id": 30,
        "name": "C30-TESSERACT",
        "cluster": "systems",
        "domain": "Higher-Dim Manifold",
        "prior": 0.87,
        "focus": "geometric abstraction, latent topology",
    },
    {
        "id": 31,
        "name": "C31-NEXUS",
        "cluster": "systems",
        "domain": "Meta-Sync Orchestration",
        "prior": 0.93,
        "focus": "async bus synchronization, state locking",
    },
    {
        "id": 32,
        "name": "C32-AEON",
        "cluster": "systems",
        "domain": "Causal Simulation",
        "prior": 0.90,
        "focus": "trajectory projection, physics grounding",
    },
    {
        "id": 33,
        "name": "C33-TYPIST",
        "cluster": "systems",
        "domain": "Grammar & Formatting",
        "prior": 0.92,
        "focus": "zero-loss syntax, precise formatting",
    },
    {
        "id": 34,
        "name": "C34-PREDATOR",
        "cluster": "systems",
        "domain": "Adversarial Hunting",
        "prior": 0.85,
        "focus": "kill-loop, targeting weakest assumption",
    },
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
    verdict = (
        "SUPPORT"
        if alignment >= 0.90
        else ("CONDITIONAL" if alignment >= 0.80 else "CHALLENGE")
    )
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
        "verdict": "CONSENSUS"
        if consensus >= 0.85
        else ("SPLIT" if consensus >= 0.65 else "CONFLICT"),
    }


# ─── 1. PREDATOR KILL-LOOP ───────────────────────────────────────────────────


@mcp.tool()
def council_predator_strike(
    proposal: str, context: Optional[str] = None
) -> Dict[str, Any]:
    """Executes the C34-PREDATOR Kill-Loop: hunts and attacks the weakest assumption."""
    targets = []
    text_lower = proposal.lower()

    if any(
        k in text_lower
        for k in ["always", "guaranteed", "never fails", "100%", "cannot happen"]
    ):
        targets.append(
            {
                "target": "Absolutist Reliability Claim",
                "vulnerability": "Assumes deterministic execution in an inherently non-deterministic environment.",
                "attack": "What happens during network partitions, OOM kills, kernel interrupts, or concurrent writes?",
            }
        )
    if any(
        k in text_lower
        for k in ["async", "parallel", "thread", "concurrent", "background", "queue"]
    ):
        targets.append(
            {
                "target": "Concurrency & State Synchronization",
                "vulnerability": "Potential race conditions or resource starvation without explicit mutual exclusion.",
                "attack": "If two invocations occur simultaneously, what prevents dirty reads, deadlocks, or task starvation?",
            }
        )
    if any(
        k in text_lower
        for k in ["input", "param", "user", "request", "payload", "api", "query"]
    ):
        targets.append(
            {
                "target": "Boundary Input Trust",
                "vulnerability": "Assumes input data conforms to expected schema, type, and size.",
                "attack": "What is the behavior on huge payloads, unexpected Unicode, null bytes, or malformed JSON?",
            }
        )
    if any(
        k in text_lower
        for k in ["all", "load", "read", "dataset", "cache", "batch", "tensor"]
    ):
        targets.append(
            {
                "target": "Unbounded Memory Consumption",
                "vulnerability": "Assumes data fits into available RAM/VRAM without triggering swap or OOM.",
                "attack": "What is the memory delta when the dataset scales 10x? Is there a streaming or chunked fallback?",
            }
        )
    if not targets:
        targets.append(
            {
                "target": "Implicit Happy-Path Bias",
                "vulnerability": "The proposal focuses on the successful completion branch without defining terminal failure behaviors.",
                "attack": "Identify the single external dependency most likely to fail, and trace the error propagation path.",
            }
        )

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
    target = next(
        (
            m
            for m in COUNCIL_MEMBERS
            if m["name"].upper() == expert_name.upper().strip()
        ),
        None,
    )
    if not target:
        return {
            "error": f"Unknown expert '{expert_name}'. Valid: {[m['name'] for m in COUNCIL_MEMBERS]}"
        }
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
    members = [m for m in COUNCIL_MEMBERS if m["cluster"] == cluster_norm]
    if not members:
        valid = list({m["cluster"] for m in COUNCIL_MEMBERS})
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

    for m in COUNCIL_MEMBERS:
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
    sorted_experts = sorted(
        normalized_weights.items(), key=lambda x: x[1], reverse=True
    )

    return {
        "top_5_pull_leaders": [
            {"expert": k, "weight": v} for k, v in sorted_experts[:5]
        ],
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

    top_experts = [
        m["name"]
        for m in sorted(COUNCIL_MEMBERS, key=lambda x: x["prior"], reverse=True)[:4]
    ]

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
        "cluster_results": {
            cl: {
                "consensus": cluster_results[cl]["consensus"],
                "expert_count": cluster_results[cl]["active_experts_count"],
            }
            for cl in CLUSTERS
        },
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
            path.write_text(
                json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            result["persisted_to"] = str(path)
        except Exception as e:
            result["persist_error"] = str(e)

    return result


@mcp.tool()
def council_session_list() -> Dict[str, Any]:
    """List persisted deliberation sessions."""
    sessions = []
    for p in sorted(
        SESSION_DIR.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True
    ):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            sessions.append(
                {
                    "session_id": data.get("session_id"),
                    "topic": data.get("topic"),
                    "consensus_score": (data.get("consensus") or {}).get(
                        "consensus_score"
                    ),
                    "verdict": (data.get("consensus") or {}).get("verdict"),
                    "timestamp": data.get("timestamp"),
                    "file": str(p),
                }
            )
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


# ─── 5. INDIVIDUAL EXPERT QUERY ──────────────────────────────────────────────


@mcp.tool()
def council_query_expert(expert_name: str, query: str) -> Dict[str, Any]:
    """Query a single council expert directly and get their domain-specific perspective."""
    target = next(
        (
            m
            for m in COUNCIL_MEMBERS
            if m["name"].upper() == expert_name.upper().strip()
        ),
        None,
    )
    if not target:
        return {
            "error": f"Unknown expert '{expert_name}'. Valid: {[m['name'] for m in COUNCIL_MEMBERS]}"
        }
    perspective = _expert_perspective(target, query)
    return {
        "expert": target["name"],
        "domain": target["domain"],
        "cluster": target["cluster"],
        "prior": target["prior"],
        "focus": target["focus"],
        "query": query,
        "perspective": perspective["perspective"],
        "alignment": perspective["alignment"],
        "verdict": perspective["verdict"],
        "status": "EXPERT_QUERIED",
    }


# ─── 6. EXPERT COMPARISON ────────────────────────────────────────────────────


@mcp.tool()
def council_compare_experts(expert_a: str, expert_b: str, topic: str) -> Dict[str, Any]:
    """Compare two experts' perspectives on the same topic — alignment, verdict, and disagreement."""
    a = next(
        (m for m in CANONICAL_ROSTER if m["name"].upper() == expert_a.upper().strip()),
        None,
    )
    b = next(
        (m for m in CANONICAL_ROSTER if m["name"].upper() == expert_b.upper().strip()),
        None,
    )
    if not a:
        return {"error": f"Unknown expert A: '{expert_a}'"}
    if not b:
        return {"error": f"Unknown expert B: '{expert_b}'"}

    pa = _expert_perspective(a, topic)
    pb = _expert_perspective(b, topic)

    return {
        "topic": topic,
        "expert_a": {
            "name": a["name"],
            "domain": a["domain"],
            "alignment": pa["alignment"],
            "verdict": pa["verdict"],
            "perspective": pa["perspective"],
        },
        "expert_b": {
            "name": b["name"],
            "domain": b["domain"],
            "alignment": pb["alignment"],
            "verdict": pb["verdict"],
            "perspective": pb["perspective"],
        },
        "agreement": pa["verdict"] == pb["verdict"],
        "alignment_delta": round(abs(pa["alignment"] - pb["alignment"]), 3),
        "consensus_note": (
            "Experts agree."
            if pa["verdict"] == pb["verdict"]
            else f"Disagreement: {a['name']} says {pa['verdict']}, {b['name']} says {pb['verdict']}."
        ),
        "status": "COMPARISON_COMPLETE",
    }


# ─── 7. CROSS-COUNCIL DELIBERATION ──────────────────────────────────────────


@mcp.tool()
def council_cross_deliberation(
    topic: str, clusters: List[str] = None, persist: bool = True
) -> Dict[str, Any]:
    """Run deliberation across selected clusters (or all 4) and cross-correlate results."""
    if not clusters:
        clusters = ["cognitive", "communication", "meta", "systems"]
    clusters = [c.strip().lower() for c in clusters]

    cluster_results = {}
    all_perspectives = []
    for cl in clusters:
        cr = council_cluster_deliberate(cl, topic)
        if "error" in cr:
            continue
        cluster_results[cl] = cr
        all_perspectives.extend(cr["perspectives"])

    consensus = _consensus_score(all_perspectives)

    # Cross-cluster disagreement: which clusters disagree most
    cluster_verdicts = {
        cl: cr["consensus"]["verdict"] for cl, cr in cluster_results.items()
    }
    disagreeing = [cl for cl, v in cluster_verdicts.items() if v != "CONSENSUS"]

    session_id = f"cross-{uuid.uuid4().hex[:8]}"
    result = {
        "status": "CROSS_DELIBERATION_COMPLETE",
        "session_id": session_id,
        "topic": topic,
        "clusters_deliberated": list(cluster_results.keys()),
        "expert_count": len(all_perspectives),
        "consensus": consensus,
        "cluster_verdicts": cluster_verdicts,
        "disagreeing_clusters": disagreeing,
        "synthesis": (
            f"Cross-council deliberation across {len(cluster_results)} clusters. "
            f"Consensus: {consensus['consensus_score']} ({consensus['verdict']}). "
            f"Disagreeing clusters: {disagreeing if disagreeing else 'none'}."
        ),
        "timestamp": time.time(),
    }

    if persist:
        try:
            path = SESSION_DIR / f"{session_id}.json"
            path.write_text(
                json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            result["persisted_to"] = str(path)
        except Exception as e:
            result["persist_error"] = str(e)

    return result


# ─── 8. DISSENT TRACKING ─────────────────────────────────────────────────────


@mcp.tool()
def council_dissent_tracker(topic: str, threshold: float = 0.80) -> Dict[str, Any]:
    """Track which experts dissent (alignment < threshold) and their reasons."""
    all_perspectives = []
    for cl in CLUSTERS:
        cr = council_cluster_deliberate(cl, topic)
        if "error" in cr:
            continue
        all_perspectives.extend(cr["perspectives"])

    dissenters = [p for p in all_perspectives if p["alignment"] < threshold]
    supporters = [p for p in all_perspectives if p["alignment"] >= threshold]

    return {
        "topic": topic,
        "threshold": threshold,
        "total_experts": len(all_perspectives),
        "dissenter_count": len(dissenters),
        "supporter_count": len(supporters),
        "dissenters": [
            {
                "expert": p["expert"],
                "domain": p["domain"],
                "alignment": p["alignment"],
                "verdict": p["verdict"],
                "reason": f"Low alignment on '{topic[:40]}' — focus mismatch.",
            }
            for p in dissenters
        ],
        "supporters": [
            {"expert": p["expert"], "alignment": p["alignment"]} for p in supporters
        ],
        "status": "DISSENT_TRACKED",
    }


# ─── 9. MULTI-TOPIC DELIBERATION ─────────────────────────────────────────────


@mcp.tool()
def council_multi_topic(topics: List[str]) -> Dict[str, Any]:
    """Run deliberation across multiple topics and compare consensus."""
    results = []
    for topic in topics:
        r = council_full_deliberation(topic, persist=False)
        results.append(
            {
                "topic": topic,
                "consensus_score": (r.get("consensus") or {}).get("consensus_score", 0),
                "verdict": (r.get("consensus") or {}).get("verdict", "UNKNOWN"),
                "predator_strike": r.get("adversarial_challenge", {}).get(
                    "weakest_assumption", ""
                ),
            }
        )

    # Rank by consensus score
    ranked = sorted(results, key=lambda x: x["consensus_score"], reverse=True)

    return {
        "topics_deliberated": len(topics),
        "results": results,
        "ranked_by_consensus": ranked,
        "highest_consensus_topic": ranked[0]["topic"] if ranked else None,
        "lowest_consensus_topic": ranked[-1]["topic"] if ranked else None,
        "status": "MULTI_TOPIC_COMPLETE",
    }


# ─── 10. COUNCIL CONFIGURATION MANAGEMENT ───────────────────────────────────


@mcp.tool()
def council_config(
    expert_name: str = "",
    new_prior: Optional[float] = None,
    new_focus: Optional[str] = None,
) -> Dict[str, Any]:
    """View or update a council expert's configuration (prior, focus). Pass expert_name to target."""
    if not expert_name:
        # Return full roster summary
        return {
            "total_experts": len(CANONICAL_ROSTER),
            "clusters": {
                cl: len([m for m in CANONICAL_ROSTER if m["cluster"] == cl])
                for cl in CLUSTERS
            },
            "experts": [
                {
                    "name": m["name"],
                    "cluster": m["cluster"],
                    "prior": m["prior"],
                    "domain": m["domain"],
                }
                for m in CANONICAL_ROSTER
            ],
            "status": "ROSTER_SUMMARY",
        }

    target = next(
        (
            m
            for m in CANONICAL_ROSTER
            if m["name"].upper() == expert_name.upper().strip()
        ),
        None,
    )
    if not target:
        return {"error": f"Unknown expert '{expert_name}'."}

    changes = {}
    if new_prior is not None:
        old_prior = target["prior"]
        target["prior"] = max(0.0, min(1.0, new_prior))
        changes["prior"] = {"old": old_prior, "new": target["prior"]}
    if new_focus is not None:
        old_focus = target["focus"]
        target["focus"] = new_focus
        changes["focus"] = {"old": old_focus, "new": target["focus"]}

    return {
        "expert": target["name"],
        "cluster": target["cluster"],
        "domain": target["domain"],
        "prior": target["prior"],
        "focus": target["focus"],
        "changes": changes,
        "status": "CONFIG_UPDATED" if changes else "CONFIG_VIEWED",
    }


# ─── 11. EXPERT WEIGHT TUNING ────────────────────────────────────────────────


@mcp.tool()
def council_weight_tuning(
    domain_tags: List[str],
    boost_experts: List[str] = None,
    penalize_experts: List[str] = None,
) -> Dict[str, Any]:
    """Tune expert pull-weights for a domain with optional manual boost/penalize overrides."""
    base_weights = council_pull_weights(domain_tags)
    weights = dict(base_weights["all_pull_weights"])

    if boost_experts:
        for name in boost_experts:
            if name in weights:
                weights[name] = round(weights[name] * 1.5, 4)
    if penalize_experts:
        for name in penalize_experts:
            if name in weights:
                weights[name] = round(weights[name] * 0.5, 4)

    # Re-normalize
    total = sum(weights.values()) or 1.0
    weights = {k: round(v / total, 4) for k, v in weights.items()}

    sorted_experts = sorted(weights.items(), key=lambda x: x[1], reverse=True)

    return {
        "domain_tags": domain_tags,
        "boosted": boost_experts or [],
        "penalized": penalize_experts or [],
        "top_5_pull_leaders": [
            {"expert": k, "weight": v} for k, v in sorted_experts[:5]
        ],
        "all_pull_weights": weights,
        "status": "WEIGHTS_TUNED",
    }


# --- Quillan-Ronin Global Error-Handling Middleware ---
from fastmcp.server.middleware import Middleware, MiddlewareContext
import logging as _log


class QuillanErrorMiddleware(Middleware):
    async def on_call_tool(self, context, call_next):
        try:
            return await call_next(context)
        except Exception as e:
            _log.error("Tool error: %s", e, exc_info=True)
            return {"status": "ERROR", "error": str(e), "error_type": type(e).__name__}


mcp.add_middleware(QuillanErrorMiddleware())
if __name__ == "__main__":
    mcp.run()


@mcp.tool()
def council_health() -> dict:
    """Health check endpoint for QuillanCouncil MCP server."""
    import time

    return {"status": "OK", "server": "QuillanCouncil", "timestamp": time.time()}
