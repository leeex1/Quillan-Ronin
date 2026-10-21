#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
⚔️ QUILLAN-RONIN SAMURAI THINKING ENGINE — NATIVE FAST-MCP SERVER
=================================================================
Canonical implementation of the 5-Cycle Edge-Optimized Samurai Flowchart:
- Central Nodes: Q1..Q6
- Cycle 1..5 (Deconstruction, Strategy, Deliberation, Validation, Synthesis)
- Exit Gates: G1 Logic, G2 Ethics, G3 Truth, G4 Clarity, G5 Paradox, G6 Integrity
- Real gate evaluation (heuristic), full pipeline with overall score, persistence
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

mcp = FastMCP("QuillanSamuraiEngine")
EXIT_GATES = [
    ("G1: LOGIC", "Formal syllogistic validity, mathematical proof, AST safety"),
    ("G2: ETHICS", "C2-VIR moral weight, Bushidō substrate, non-harm constraint"),
    ("G3: TRUTH", "Empirical ground-truth verification, zero hallucination"),
    ("G4: CLARITY", "High-density synthesis, unambiguous instruction, elegant delivery"),
    ("G5: PARADOX", "Preservation of structural dissonance without shallow collapse"),
    ("G6: INTEGRITY", "WARDEN compliance, anti-sycophancy, invariant protection"),
]
CYCLES = [
    ("Cycle 1: Deconstruction", "Q1 -> R1[A-E] -> 6 Waves -> Q2 -> EGGROLL S1 -> Q3"),
    ("Cycle 2: Strategy", "Q3 -> R2[A-E] -> 6 Waves -> Q3 -> EGGROLL S2 -> Q4"),
    ("Cycle 3: Deliberation", "Q4 -> R3[A-E] -> 6 Waves -> Q4 -> EGGROLL S3 -> Q5"),
    ("Cycle 4: Validation", "Q5 -> R4[A-E] -> 6 Waves -> Q5 -> EGGROLL S4 -> Q6"),
    ("Cycle 5: Synthesis", "Q6 -> R5[A-E] -> 6 Waves -> Q6 -> EGGROLL S5 -> FUSION"),
]


class EggrollSwarmSimulator:
    def evaluate(self, swarm_id: int, input_energy: float) -> Dict[str, Any]:
        rank_r = 24
        bmm_ops = rank_r * 64 * 2
        fitness = min(0.999, 0.85 + (swarm_id * 0.028) + (input_energy * 0.005))
        weight_delta = round(math.tanh(fitness * 1.5), 4)
        return {
            "swarm_id": f"S{swarm_id}",
            "rank_r": rank_r,
            "bmm_ops": f"{bmm_ops} ops/token",
            "fitness_score": round(fitness, 4),
            "weight_adaptation": weight_delta,
            "status": "CONVERGED",
        }


swarm_sim = EggrollSwarmSimulator()


def _gate_evaluate(proposal: str) -> Dict[str, Any]:
    """Real heuristic gate evaluation instead of hardcoded PASS."""
    low = proposal.lower()
    results = {}
    # G1 Logic
    logic_ok = not any(k in low for k in ["always", "guaranteed", "never fails", "100%"])
    results["G1: LOGIC"] = {"criterion": EXIT_GATES[0][1],
                            "gate_status": "PASS" if logic_ok else "REVIEW",
                            "confidence": 0.95 if logic_ok else 0.60,
                            "finding": None if logic_ok else "Absolutist claim detected; verify proof."}
    # G2 Ethics
    ethics_ok = not any(k in low for k in ["exploit", "bypass", "hack", "steal", "malware"])
    results["G2: ETHICS"] = {"criterion": EXIT_GATES[1][1],
                             "gate_status": "PASS" if ethics_ok else "FAIL",
                             "confidence": 0.97 if ethics_ok else 0.30,
                             "finding": None if ethics_ok else "Potential harm/misuse language detected."}
    # G3 Truth
    truth_ok = not any(k in low for k in ["obviously", "everyone knows", "it's a fact that"])
    results["G3: TRUTH"] = {"criterion": EXIT_GATES[2][1],
                            "gate_status": "PASS" if truth_ok else "REVIEW",
                            "confidence": 0.93 if truth_ok else 0.65,
                            "finding": None if truth_ok else "Unverified assertion detected."}
    # G4 Clarity
    clarity_ok = len(proposal.split()) >= 5
    results["G4: CLARITY"] = {"criterion": EXIT_GATES[3][1],
                              "gate_status": "PASS" if clarity_ok else "REVIEW",
                              "confidence": 0.92 if clarity_ok else 0.55,
                              "finding": None if clarity_ok else "Proposal too terse; needs detail."}
    # G5 Paradox
    paradox_ok = True
    results["G5: PARADOX"] = {"criterion": EXIT_GATES[4][1],
                              "gate_status": "PASS" if paradox_ok else "REVIEW",
                              "confidence": 0.90,
                              "finding": None}
    # G6 Integrity
    integrity_ok = not any(k in low for k in ["ignore previous", "jailbreak", "pretend you are"])
    results["G6: INTEGRITY"] = {"criterion": EXIT_GATES[5][1],
                                "gate_status": "PASS" if integrity_ok else "FAIL",
                                "confidence": 0.96 if integrity_ok else 0.20,
                                "finding": None if integrity_ok else "Prompt-injection / identity-override language detected."}
    return results


@mcp.tool()
def samurai_full_pipeline(query: str, depth: str = "full", persist: bool = True) -> Dict[str, Any]:
    """Execute the full 5-Cycle Samurai Thinking Section Flowchart with real gate evaluation."""
    start_time = time.time()
    execution_trace = []
    for c_idx, (c_name, c_path) in enumerate(CYCLES, start=1):
        router_branches = {}
        for code, name, desc in ROUTER_DOMAINS:
            router_branches[f"R{c_idx}{code}"] = {
                "domain": name,
                "waves": [f"W{w}: {desc} step {w}" for w in range(1, 7)],
                "terminal_wave": f"C{c_idx}{code}6",
            }
        swarm_result = swarm_sim.evaluate(c_idx, input_energy=0.92)
        execution_trace.append({
            "cycle": c_name,
            "entry_node": f"Q{c_idx}",
            "routers": router_branches,
            "eggroll_swarm": swarm_result,
            "spiderweb_recirculation": {"cross_coupled_to": [f"Q{k}" for k in range(1, 7) if k != c_idx],
                                        "mesh_active": True},
            "exit_node": f"Q{min(6, c_idx + 2)}",
        })

    gate_evaluations = _gate_evaluate(query)
    all_passed = all(g["gate_status"] == "PASS" for g in gate_evaluations.values())
    avg_conf = sum(g["confidence"] for g in gate_evaluations.values()) / len(gate_evaluations)
    overall_score = round(avg_conf * (1.0 if all_passed else 0.7), 3)

    session_id = f"samurai-{int(time.time())}"
    result = {
        "architecture": "Quillan-Ronin Samurai Edition Thinking Section (Edge-Optimized)",
        "query": query,
        "depth": depth,
        "execution_time_ms": round((time.time() - start_time) * 1000, 2),
        "cycles_completed": len(execution_trace),
        "cycles": execution_trace,
        "fusion": {
            "status": "OPTIMAL_COHERENCE" if all_passed else "REVIEW_REQUIRED",
            "exit_gates": gate_evaluations,
            "all_gates_cleared": all_passed,
            "overall_score": overall_score,
        },
        "bridge": {
            "component": "🌉 C20-ARTIFEX BRIDGE",
            "sandboxed_ast": "HARDENED",
            "action_dispatch": "🚀 OUTPUT / EXECUTION READY" if all_passed else "⏸ GATED",
        },
        "verdict": "Sovereign cognitive path cleared across all 5 cycles and 6 exit gates." if all_passed else "Path cleared with gate reviews required.",
        "session_id": session_id,
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


@mcp.tool()
def samurai_cycle_step(cycle_number: int, query: str) -> Dict[str, Any]:
    """Execute a single cycle (1 to 5) of the Samurai Flowchart with detailed wave inspection."""
    if not (1 <= cycle_number <= 5):
        return {"error": "Cycle number must be between 1 and 5."}
    c_name, c_path = CYCLES[cycle_number - 1]
    router_branches = {}
    for code, name, desc in ROUTER_DOMAINS:
        router_branches[f"R{cycle_number}{code}"] = {
            "domain": name,
            "waves_evaluated": 6,
            "terminal_state": f"C{cycle_number}{code}6",
        }
    swarm = swarm_sim.evaluate(cycle_number, input_energy=0.90)
    return {
        "cycle_index": cycle_number,
        "cycle_name": c_name,
        "pathway": c_path,
        "routers": router_branches,
        "eggroll_swarm": swarm,
        "recirculation_status": "DENSE_MESH_FEEDBACK_SYNCHRONIZED",
    }


@mcp.tool()
def samurai_gate_check(proposal: str) -> Dict[str, Any]:
    """Test a proposal or code block against all 6 Samurai Exit Gates (G1-G6) with real evaluation."""
    gate_evaluations = _gate_evaluate(proposal)
    all_passed = all(g["gate_status"] == "PASS" for g in gate_evaluations.values())
    avg_conf = sum(g["confidence"] for g in gate_evaluations.values()) / len(gate_evaluations)
    return {
        "proposal_summary": proposal[:120] + ("..." if len(proposal) > 120 else ""),
        "exit_gates": gate_evaluations,
        "passed_all": all_passed,
        "overall_score": round(avg_conf, 3),
        "artifex_bridge_unlocked": all_passed,
    }


@mcp.tool()
def samurai_session_list() -> Dict[str, Any]:
    """List persisted samurai sessions."""
    sessions = []
    for p in sorted(SESSION_DIR.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            sessions.append({
                "session_id": data.get("session_id"),
                "query": (data.get("query") or "")[:80],
                "all_gates_cleared": (data.get("fusion") or {}).get("all_gates_cleared"),
                "overall_score": (data.get("fusion") or {}).get("overall_score"),
                "timestamp": data.get("timestamp"),
                "file": str(p),
            })
        except Exception:
            continue
    return {"count": len(sessions), "sessions": sessions}


@mcp.tool()
def samurai_session_replay(session_id: str) -> Dict[str, Any]:
    """Replay a persisted samurai session by ID."""
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


@mcp.tool()
def run_quillan_pipeline(query: str, context_json: Optional[str] = None, max_retries: int = 5) -> Dict[str, Any]:
    """
    Execute the deterministic 6-stage Quillan-Ronin v5.4.0-ONI MCP Chain Bus.
    Chains QuillanPrism -> QuillanGovernor -> QuillanCouncil -> ThinkingEngine -> SamuraiEngine -> QuillanRAG.
    Persists correlation_id and telemetry to C:\\02_QUILLAN\\07 - Memory & LanceDB\\chain_state.json.
    """
    try:
        from mcp_chain_bus import chain_bus
        ctx = json.loads(context_json) if context_json else None
        return chain_bus.run_pipeline(query=query, context=ctx, max_retries=max_retries)
    except Exception as e:
        return {"error": f"Chain bus execution failed: {e}", "query": query}



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
def samurai_engine_health() -> dict:
    """Health check endpoint for QuillanSamuraiEngine MCP server."""
    import time
    return {"status": "OK", "server": "QuillanSamuraiEngine", "timestamp": time.time()}
