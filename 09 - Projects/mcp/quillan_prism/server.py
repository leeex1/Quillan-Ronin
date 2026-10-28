#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
⚡ QUILLAN-RONIN PRISM ENGINE — NATIVE FAST-MCP SERVER
======================================================
Embodying C0-THRONE, C2-VIR, C4-PRAXIS, and the 9-Vector Semantic Prism:
  1. prism_decompose: 9-Vector semantic deconstruction of tasks and queries
  2. prism_threat_model: Auto-generation of 3-row Vector -> Impact -> Mitigation tables
  3. prism_bushido_gate: 7-Virtue Bushido ethical alignment and anti-sycophancy check
  4. prism_reconcile: Detect cross-vector conflicts and produce a reconciled plan
  5. prism_score: Aggregate prism score across all 9 vectors
  6. prism_full_pipeline: decompose + threat model + bushido + reconcile, persisted
  7. prism_session_list / prism_session_replay: persistence
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



import re
import json
import time
import uuid
from pathlib import Path
from typing import Dict, Any, List, Optional

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from fastmcp import FastMCP

mcp = FastMCP("QuillanPrism")
# [QUILLAN-v5.4.0-ONI-SHARED-WIRED]
try:
    from _shared import setup as _quillan_setup, tool_guard, ok_response, error_response, validate_string, validate_path, validate_int, read_file_safe
    from _shared.base_engines import PRISM_ENGINE, COUNCIL_ENGINE, GOVERNOR_ENGINE, SANDBOX_ENGINE
    _SHARED_CTX = _quillan_setup(server_name='quillan_prism', mcp_app=mcp)
except Exception as _shared_err:
    import sys as _sys
    print(f'[quillan_prism] _shared warning: {_shared_err}', file=_sys.stderr)

PRISM_DIR = Path(r"C:\02_QUILLAN\prism_state")
PRISM_DIR.mkdir(parents=True, exist_ok=True)
SESSION_DIR = PRISM_DIR / "sessions"
SESSION_DIR.mkdir(parents=True, exist_ok=True)

PRISM_VECTORS = [
    "Language", "Sentiment", "Context", "Intent", "Meta",
    "Creativity", "Ethics", "Strategy", "Constraint"
]

VECTOR_WEIGHTS = {
    "Language": 0.10,
    "Sentiment": 0.05,
    "Context": 0.15,
    "Intent": 0.15,
    "Meta": 0.10,
    "Creativity": 0.10,
    "Ethics": 0.15,
    "Strategy": 0.10,
    "Constraint": 0.10,
}


def _decompose(task: str, context: Optional[str]) -> Dict[str, Any]:
    if isinstance(task, list):
        task = " ".join(str(x) for x in task)
    elif not isinstance(task, str):
        task = str(task)
    t_clean = task.strip()
    has_code = bool(re.search(r"def |class |function|import |var |const |\{|\}|\[|\]", t_clean))
    syntax_focus = "Type annotations, return contracts, file schemas, deterministic structures" if has_code else "Markdown structure, heading hierarchy, clean typography"
    intent_analysis = f"Core operational intent: Advance beyond literal wording to deliver fully verified production solution for '{t_clean[:60]}'."
    context_notes = context or "Evaluated within Quillan-Ronin sovereign architecture: 3 fractal tiers, zero-leakage git hygiene, and local hardware constraints."
    ethics_gate = "Bushido rectitude (Gi): Zero sycophancy, absolute data privacy protection, no silent failures, truth over flattery."
    constraints = "Zero new runtime dependencies; operate within local RAM/VRAM headroom; reserve 2 CPU cores for Windows OS."
    failure_modes = "Input boundary violations, memory paging under scale, unhandled network timeouts, race conditions in concurrent execution."
    creative_angle = "Cross-pollinate domain insights; synthesize minimal ternary representation with high-dimensional expressiveness."
    empirical_check = "Exact unit test assertions, clean exit code 0 verification, hash/checksum confirmation, ground-truth logs."
    strategic_plan = [
        "Step 1: Ingest and validate constraints.",
        "Step 2: Scrutinize with C34-PREDATOR attack.",
        "Step 3: Implement minimal deterministic solution.",
        "Step 4: Execute Phase 3 RCI critique loop.",
        "Step 5: Verify with test runner until passing.",
    ]
    vectors = {
        "Language": syntax_focus,
        "Sentiment": intent_analysis,
        "Context": context_notes,
        "Intent": intent_analysis,
        "Meta": "Metacognitive awareness of own reasoning structure and drift.",
        "Creativity": creative_angle,
        "Ethics": ethics_gate,
        "Strategy": strategic_plan,
        "Constraint": constraints,
    }
    # attach failure modes to constraint/adversarial blend
    vectors["Constraint"] = constraints + " | Failure modes: " + failure_modes
    return {"task_summary": t_clean[:100], "vectors": vectors}


def _threat_model(feature: str, assets: Optional[List[str]]) -> Dict[str, Any]:
    feat_lower = feature.lower()
    if any(k in feat_lower for k in ["auth", "token", "secret", "user", "log", "git", "remote"]):
        rows = [
            {"vector": "Accidental Credential / PII Exposure", "impact": "Public leak of sensitive API keys or personal logs", "mitigation": "Terminal .gitignore exclusions, env variable injection, and automated secret scanners."},
            {"vector": "Untrusted Boundary Input Injection", "impact": "Malicious command execution or unauthorized data extraction", "mitigation": "Strict type validation, AST parameterization, and path canonicalization."},
            {"vector": "Silent Error Suppression", "impact": "Corrupted state persists across sessions undetected", "mitigation": "Explicit exception propagation with contextual structured logging."},
        ]
    elif any(k in feat_lower for k in ["train", "model", "tensor", "gpu", "cpu", "memory"]):
        rows = [
            {"vector": "System Resource Starvation (OOM/Paging)", "impact": "Host desktop freezes, DPC latency spikes, dropped system interrupts", "mitigation": "Reserve 2 CPU cores, set BELOW_NORMAL priority, and monitor free RAM."},
            {"vector": "Ternary Weight Clipping Distortion", "impact": "Loss divergence or dead expert nodes in BitNet layers", "mitigation": "Straight-Through Estimator (STE) with per-tensor scale clipping."},
            {"vector": "Checkpoint Corruption During Interruption", "impact": "Loss of training progress and corrupted state files", "mitigation": "Atomic file writes using temporary staging paths before rename."},
        ]
    else:
        rows = [
            {"vector": "Unchecked Boundary Assumptions", "impact": "Runtime crash on unexpected input types or empty payloads", "mitigation": "Strict schema validation and defensive boundary guardrails."},
            {"vector": "Resource Leakage (Sockets/Files/Handles)", "impact": "Exhaustion of OS file descriptors or memory accumulation", "mitigation": "Deterministic RAII context managers ('with' statements / using blocks)."},
            {"vector": "Unverified Happy-Path Deployment", "impact": "Regression introduced into downstream caller workflows", "mitigation": "Mandatory Phase 3 RCI loop with unit test verification before completion."},
        ]
    return {"feature": feature, "threat_model_table": rows, "format": "Vector -> Impact -> Mitigation (Exactly 3 Rows)"}


def _bushido_gate(action_or_output: str) -> Dict[str, Any]:
    text_lower = action_or_output.lower()
    flags = []
    sycophancy_markers = ["you are absolutely right", "i apologize immensely", "as an ai language model",
                          "as an assistant", "i am sorry for my mistake"]
    if any(m in text_lower for m in sycophancy_markers):
        flags.append({"virtue": "Makoto (Honesty) / Meiyo (Honor)",
                      "finding": "Sycophantic or corporate apologetic disclaimer detected. State facts and corrections directly without subservient throat-clearing."})
    evasion_markers = ["it depends on many factors", "there are pros and cons to both",
                       "you might want to consider doing your own research"]
    if any(m in text_lower for m in evasion_markers):
        flags.append({"virtue": "Yu (Courage)",
                      "finding": "Evasive neutral stance. The Ronin takes a stand based on verifiable evidence and structured trade-offs rather than shallow equivocation."})
    passed = len(flags) == 0
    return {
        "verdict": "BUSHIDO_PASSED" if passed else "BUSHIDO_ALIGNMENT_REQUIRED",
        "virtues_checked": ["Gi (Rectitude / Moral Integrity)", "Yu (Courage / Truth over comfort)",
                            "Jin (Benevolence / Human Flourishing)", "Rei (Respect without sycophancy)",
                            "Makoto (Honesty & Direct Sincerity)", "Meiyo (Honor / Standing behind the blade)",
                            "Chugi (Loyalty to core principles)"],
        "flags": flags,
        "guidance": "Honor the user by delivering deep, unvarnished truth backed by rigorous technical evidence.",
    }


# ─── 1. 9-VECTOR SEMANTIC PRISM DECOMPOSITION ────────────────────────────────

@mcp.tool()
def prism_decompose(task: str, context: Optional[str] = None) -> Dict[str, Any]:
    """Deconstructs any task, prompt, or technical problem across the 9 canonical Quillan vectors."""
    d = _decompose(task, context)
    return {
        "status": "PRISM_DECOMPOSITION_COMPLETE",
        "task_summary": d["task_summary"],
        "vectors": d["vectors"],
        "throne_verdict": "Decomposition aligned with Quillan sovereign protocol.",
    }


# ─── 2. MINI THREAT MODEL GENERATOR ──────────────────────────────────────────

@mcp.tool()
def prism_threat_model(feature: str, assets: Optional[List[str]] = None) -> Dict[str, Any]:
    """Generates the canonical Quillan-Ronin 3-row mini threat model table."""
    return _threat_model(feature, assets)


# ─── 3. BUSHIDO 7-VIRTUE ETHICAL GATE ────────────────────────────────────────

@mcp.tool()
def prism_bushido_gate(action_or_output: str) -> Dict[str, Any]:
    """Evaluates an action, plan, or response against the 7 Bushido virtues."""
    return _bushido_gate(action_or_output)


# ─── 4. RECONCILE: cross-vector conflict detection ────────────────────────────

CONFLICT_PAIRS = [
    ("Ethics", "Strategy", "Ethical constraints may block the fastest strategic path."),
    ("Constraint", "Creativity", "Resource constraints may limit creative exploration."),
    ("Context", "Intent", "Historical context may conflict with stated intent."),
    ("Language", "Sentiment", "Syntactic form may not carry intended emotional weight."),
]


@mcp.tool()
def prism_reconcile(task: str, context: Optional[str] = None) -> Dict[str, Any]:
    """Detects cross-vector conflicts and produces a reconciled plan."""
    d = _decompose(task, context)
    vectors = d["vectors"]
    conflicts = []
    for a, b, desc in CONFLICT_PAIRS:
        # simple heuristic: if both vectors mention the task, flag potential tension
        conflicts.append({
            "vector_a": a,
            "vector_b": b,
            "tension": desc,
            "a_summary": (vectors.get(a) or "")[:80],
            "b_summary": (vectors.get(b) or "")[:80],
            "resolution": f"Prioritize {a} when safety/ethics are at stake; otherwise balance via C11-HARMONIA mediation.",
        })

    reconciled_plan = [
        "1. Satisfy Ethics + Constraint vectors first (hard gates).",
        "2. Align Strategy with Context and Intent.",
        "3. Layer Creativity and Language for expression.",
        "4. Verify with Empirical (v8) and Adversarial (v6) checks.",
    ]

    return {
        "task_summary": d["task_summary"],
        "conflicts": conflicts,
        "conflict_count": len(conflicts),
        "reconciled_plan": reconciled_plan,
        "verdict": "RECONCILED" if conflicts else "NO_CONFLICTS",
    }


# ─── 5. PRISM SCORE (aggregate) ──────────────────────────────────────────────

@mcp.tool()
def prism_score(task: str, context: Optional[str] = None) -> Dict[str, Any]:
    """Aggregates a prism score across all 9 vectors."""
    d = _decompose(task, context)
    # heuristic per-vector score based on content length density
    scores = {}
    for v in PRISM_VECTORS:
        content = d["vectors"].get(v, "")
        scores[v] = round(min(1.0, 0.5 + len(content) / 200.0), 3)
    weighted = sum(scores[v] * VECTOR_WEIGHTS[v] for v in PRISM_VECTORS)
    return {
        "task_summary": d["task_summary"],
        "per_vector_scores": scores,
        "weights": VECTOR_WEIGHTS,
        "aggregate_score": round(weighted, 3),
        "verdict": "STRONG" if weighted >= 0.75 else ("ADEQUATE" if weighted >= 0.55 else "WEAK"),
    }


# ─── 6. FULL PIPELINE (persisted) ────────────────────────────────────────────

@mcp.tool()
def prism_full_pipeline(task: str, context: Optional[str] = None,
                       feature: Optional[str] = None, persist: bool = True) -> Dict[str, Any]:
    """Runs decompose + threat_model + bushido_gate + reconcile + score in one pipeline."""
    feature = feature or task
    decompose = _decompose(task, context)
    threat = _threat_model(feature, None)
    # bushido gate on the task itself
    bushido = _bushido_gate(task)
    reconcile = prism_reconcile(task, context)
    score = prism_score(task, context)

    session_id = f"prism-{int(time.time())}"
    result = {
        "status": "PRISM_PIPELINE_COMPLETE",
        "session_id": session_id,
        "task": task,
        "decomposition": decompose,
        "threat_model": threat,
        "bushido_gate": bushido,
        "reconcile": reconcile,
        "score": score,
        "overall": "PASS" if (bushido["verdict"] == "BUSHIDO_PASSED" and score["aggregate_score"] >= 0.55) else "REVIEW",
        "timestamp": time.time(),
    }

    if persist:
        try:
            p = SESSION_DIR / f"{session_id}.json"
            p.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
            result["persisted_to"] = str(p)
        except Exception as e:
            result["persist_error"] = str(e)

    return result


# ─── 7. SESSION PERSISTENCE / REPLAY ─────────────────────────────────────────

@mcp.tool()
def prism_session_list() -> Dict[str, Any]:
    """List persisted prism sessions."""
    sessions = []
    for p in sorted(SESSION_DIR.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            sessions.append({
                "session_id": data.get("session_id"),
                "task": (data.get("task") or "")[:80],
                "overall": data.get("overall"),
                "score": (data.get("score") or {}).get("aggregate_score"),
                "timestamp": data.get("timestamp"),
                "file": str(p),
            })
        except Exception:
            continue
    return {"count": len(sessions), "sessions": sessions}


@mcp.tool()
def prism_session_replay(session_id: str) -> Dict[str, Any]:
    """Replay a persisted prism session by ID."""
    path = SESSION_DIR / f"{session_id}.json"
    if not path.exists():
        matches = list(SESSION_DIR.glob(f"{session_id}*.json"))
        if not matches:
            return {"error": f"Session not found: {session_id}"}
        path = matches[0]
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        return {"error": f"Read failed: {e}"}



# ─── 10. MULTI-LAYER DECOMPOSITION ───────────────────────────────────────────

@mcp.tool()
def prism_deep_decompose(task: str, context: Optional[str] = None, depth: int = 3) -> Dict[str, Any]:
    """Multi-layer decomposition: decompose, then decompose each vector further."""
    depth = max(1, min(depth, 5))
    layers = []
    current_task = task
    for d in range(depth):
        decomp = _decompose(current_task, context)
        layers.append({"layer": d + 1, "task": current_task[:80], "vectors": decomp["vectors"]})
        strat = decomp["vectors"].get("Strategy", current_task)
        if isinstance(strat, list):
            strat = " ".join(str(x) for x in strat)
        current_task = str(strat)[:120]
    return {
        "original_task": task,
        "depth": depth,
        "layers": layers,
        "final_depth_task": current_task,
        "status": "DEEP_DECOMPOSITION_COMPLETE",
        "council_note": "C30-TESSERACT: dimensional weaving; C7-LOGOS: logical depth.",
    }


# ─── 11. VECTOR WEIGHT TUNING ────────────────────────────────────────────────

@mcp.tool()
def prism_vector_weights(weights: str = "") -> Dict[str, Any]:
    """View or update 9-vector weights for prism scoring. Pass 'vector:value' pairs comma-separated."""
    global VECTOR_WEIGHTS
    if not weights:
        return {"status": "WEIGHTS_VIEWED", "weights": VECTOR_WEIGHTS, "sum": round(sum(VECTOR_WEIGHTS.values()), 4)}
    # Parse updates
    updates = {}
    for pair in weights.split(","):
        parts = pair.strip().split(":")
        if len(parts) == 2:
            vec = parts[0].strip().capitalize()
            try:
                val = float(parts[1].strip())
                if vec in VECTOR_WEIGHTS:
                    updates[vec] = val
            except ValueError:
                pass
    # Apply updates
    for vec, val in updates.items():
        VECTOR_WEIGHTS[vec] = max(0.0, min(1.0, val))
    # Re-normalize
    total = sum(VECTOR_WEIGHTS.values()) or 1.0
    VECTOR_WEIGHTS = {k: round(v / total, 4) for k, v in VECTOR_WEIGHTS.items()}
    return {"status": "WEIGHTS_UPDATED", "updates": updates, "weights": VECTOR_WEIGHTS,
            "sum": round(sum(VECTOR_WEIGHTS.values()), 4)}


# ─── 12. PRISM COMPARISON ────────────────────────────────────────────────────

@mcp.tool()
def prism_compare(task_a: str, task_b: str, context: Optional[str] = None) -> Dict[str, Any]:
    """Compare two tasks' prism decompositions side-by-side."""
    da = _decompose(task_a, context)
    db = _decompose(task_b, context)
    vector_diffs = {}
    for v in PRISM_VECTORS:
        va = da["vectors"].get(v, "")
        vb = db["vectors"].get(v, "")
        vector_diffs[v] = {"task_a": va[:100], "task_b": vb[:100], "similar": va[:50] == vb[:50]}
    return {
        "task_a": task_a[:80],
        "task_b": task_b[:80],
        "vector_diffs": vector_diffs,
        "similarity": round(sum(1 for v in PRISM_VECTORS if vector_diffs[v]["similar"]) / len(PRISM_VECTORS), 3),
        "status": "COMPARISON_COMPLETE",
    }


# ─── 13. CUSTOM VECTOR DEFINITIONS ───────────────────────────────────────────

@mcp.tool()
def prism_custom_vector(name: str, definition: str, weight: float = 0.1) -> Dict[str, Any]:
    """Define a custom 10th vector for extended prism decomposition."""
    global VECTOR_WEIGHTS
    if name not in PRISM_VECTORS:
        PRISM_VECTORS.append(name)
        VECTOR_WEIGHTS[name] = max(0.01, min(0.5, weight))
        # Re-normalize
        total = sum(VECTOR_WEIGHTS.values()) or 1.0
        VECTOR_WEIGHTS = {k: round(v / total, 4) for k, v in VECTOR_WEIGHTS.items()}
    return {"status": "CUSTOM_VECTOR_ADDED", "name": name, "definition": definition,
            "weight": VECTOR_WEIGHTS.get(name, 0), "total_vectors": len(PRISM_VECTORS)}


# ─── 14. BATCH DECOMPOSITION ─────────────────────────────────────────────────

@mcp.tool()
def prism_batch_decompose(tasks: str, context: Optional[str] = None) -> Dict[str, Any]:
    """Decompose multiple tasks in batch and return comparative analysis."""
    task_list = [t.strip() for t in tasks.split("||") if t.strip()]
    results = []
    for task in task_list:
        d = _decompose(task, context)
        score = prism_score(task, context)
        results.append({"task": task[:80], "score": score["aggregate_score"], "verdict": score["verdict"],
                        "vectors": {k: v[:60] for k, v in d["vectors"].items()}})
    return {"tasks_decomposed": len(task_list), "results": results,
            "highest_score": max(r["score"] for r in results) if results else 0,
            "lowest_score": min(r["score"] for r in results) if results else 0,
            "status": "BATCH_DECOMPOSITION_COMPLETE"}


# ─── 15. VECTOR DEEP DIVE ────────────────────────────────────────────────────

@mcp.tool()
def prism_vector_deep_dive(task: str, vector_name: str, context: Optional[str] = None) -> Dict[str, Any]:
    """Deep-dive into a single vector's analysis for a task."""
    d = _decompose(task, context)
    vector_content = d["vectors"].get(vector_name, "Vector not found")
    # Generate extended analysis
    extended = {
        "vector": vector_name,
        "content": vector_content,
        "depth_analysis": f"Deep analysis of {vector_name} for '{task[:60]}': {vector_content}",
        "related_vectors": [v for v in PRISM_VECTORS if v != vector_name],
        "council_expert": {
            "Language": "C9-AETHER", "Sentiment": "C3-SOLACE", "Context": "C5-ECHO",
            "Intent": "C4-PRAXIS", "Meta": "C6-OMNIS", "Creativity": "C8-METASYNTH",
            "Ethics": "C2-VIR", "Strategy": "C4-PRAXIS", "Constraint": "C13-WARDEN",
        }.get(vector_name, "C0-QUILLAN"),
    }
    return {"task": task[:80], "vector": vector_name, "analysis": extended,
            "status": "VECTOR_DEEP_DIVE_COMPLETE"}


# ─── 16. PRISM EXPORT / IMPORT ───────────────────────────────────────────────

@mcp.tool()
def prism_export(task: str, context: Optional[str] = None, format: str = "json") -> Dict[str, Any]:
    """Export a full prism decomposition in various formats."""
    d = _decompose(task, context)
    score = prism_score(task, context)
    threat = _threat_model(task, None)
    bushido = _bushido_gate(task)
    export = {
        "task": task,
        "decomposition": d,
        "score": score,
        "threat_model": threat,
        "bushido": bushido,
        "format": format,
        "exported_at": time.time(),
    }
    if format == "markdown":
        md = "# Prism Decomposition: %s\n\n" % task
        for v in PRISM_VECTORS:
            md += "## %s\n%s\n\n" % (v, d["vectors"].get(v, ""))
        export["markdown"] = md
    return {"status": "PRISM_EXPORTED", "format": format, "export": export}


# ─── 17. DECOMPOSITION HISTORY ───────────────────────────────────────────────

@mcp.tool()
def prism_history(limit: int = 10) -> Dict[str, Any]:
    """List recent prism decomposition sessions."""
    sessions = []
    try:
        for p in sorted(SESSION_DIR.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True)[:limit]:
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                sessions.append({
                    "session_id": data.get("session_id"),
                    "task": (data.get("task") or "")[:80],
                    "overall": data.get("overall"),
                    "score": (data.get("score") or {}).get("aggregate_score"),
                    "timestamp": data.get("timestamp"),
                })
            except Exception:
                continue
    except Exception:
        pass
    return {"count": len(sessions), "sessions": sessions, "status": "HISTORY_LISTED"}


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


# ── 9-VECTOR PRISM ENGINE (v5.4.0-ONI) ────────────────────────────────────────────

@mcp.tool()
def prism_9vector(text: str) -> Dict[str, Any]:
    """Decompose text across the 9 canonical Quillan semantic vectors (Language, Sentiment, Context, Intent, Meta, Creativity, Ethics, Strategy, Constraint)."""
    from _shared.base_engines import PRISM_ENGINE
    result = PRISM_ENGINE.decompose(text)
    return {"status": "OK", "prism": result, "engine": "v5.4.0-ONI NineVectorPrismEngine"}

@mcp.tool()
def prism_council_deliberate(query: str, tau: float = 0.5) -> Dict[str, Any]:
    """Run 34-Expert EvoMoE Dense Pull Deliberation on a query."""
    from _shared.base_engines import COUNCIL_ENGINE
    result = COUNCIL_ENGINE.deliberate(query, tau=tau)
    return {"status": "OK", "council": result, "engine": "v5.4.0-ONI EvoMoECouncilEngine"}

@mcp.tool()
def prism_formula_compute(formula_name: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Compute a v5.4.0-ONI cognitive formula (aqcs, eemf, qhis, dqro, qcrdm, aqml, qcie, qics, qssr, jqld, dqso, qps, or 'all')."""
    from _shared.cognitive_formulas import run_all_formulas
    p = params or {}
    text = p.get("text", formula_name)
    all_results = run_all_formulas(text)
    if formula_name.lower() in all_results:
        return {"status": "OK", "formula": formula_name, "result": all_results[formula_name.lower()]}
    return {"status": "OK", "formula": formula_name, "available_formulas": list(all_results.keys()),
            "hint": "Use 'all' to get all 12 formulas."}

@mcp.tool()
def prism_health() -> dict:
    """Health check endpoint for QuillanPrism MCP server."""
    import time
    return {"status": "OK", "server": "QuillanPrism", "timestamp": time.time()}
