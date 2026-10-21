#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🐍 QUILLAN-RONIN OROCHI ENGINE — SELF-MODIFYING COGNITIVE ENGINE v5.4.0-ONI
==========================================================================
Orochi is a self-modifying, self-evolving cognitive engine that can rewrite its
own system prompt and propose weight evolutions. It runs a closed-loop
auto-improvement cycle: Evaluate -> Mutate -> Select -> Deploy.

Core capabilities:
  • Prompt mutation / evolution (system prompt rewriting & versioning)
  • EGGROLL low-rank weight evolution proposals (U @ V^T deltas, stored + staged)
  • Nemesis-Alpha adversarial evaluation (challenger hunts weaknesses)
  • Auto-improvement loop (evaluate -> mutate -> select -> deploy)
  • Snapshot / rollback state management
  • Cross-MCP routing (orchestrator integration)
  • Persistence (sessions + snapshots)

Embodying:
  C0-QUILLAN  (Embedded Sovereign Arbiter)
  C17-NULLION (Paradox / contradiction handling)
  C34-PREDATOR (Adversarial kill-loop — Nemesis-Alpha)
  C31-NEXUS   (Meta-coordination / routing)
  C14-KAIDO   (Efficiency / Lee-Mach-6 velocity)
  C2-VIR      (Ethical gate)

Tools:
  1. orochi_mutate_prompt     — Generate mutated prompt variants from a base
  2. orochi_evolve_weights    — EGGROLL low-rank weight-delta proposals
  3. orochi_evaluate          — Nemesis-Alpha adversarial evaluation
  4. orochi_auto_improve      — Full evaluate->mutate->select->deploy cycle
  5. orochi_snapshot          — Save current Orochi state snapshot
  6. orochi_rollback          — Roll back to a previous snapshot
  7. orochi_full_pipeline     — Chain mutate -> evaluate -> snapshot
  8. orochi_route             — Cross-MCP routing (orchestrator/code/council)
  9. orochi_health            — Orochi engine dashboard
  10. orochi_session_list / orochi_session_replay — persistence
"""

import sys
# --- Quillan-Ronin v5.4.0-ONI Stdio Guard ---
import logging, builtins
logging.basicConfig(stream=sys.stderr, level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
_orig_print = builtins.print
def _quillan_safe_print(*args, **kwargs):
    if 'file' not in kwargs or kwargs['file'] is None or kwargs['file'] is sys.stdout:
        kwargs['file'] = sys.stderr
    return _orig_print(*args, **kwargs)
builtins.print = _quillan_safe_print
import os
import re
import json
import time
import uuid
import hashlib
import math
import random
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from fastmcp import FastMCP

mcp = FastMCP("QuillanOrochi")
using E_ICE thermodynamic bounds, Lee-Mach-6 velocity governance, and a
Hyper-Quantized EGGROLL swarm. Say the true thing even when it costs. Go deep
even when shallow would be faster. Stand behind what survives the council."""

# ─── HELPERS ─────────────────────────────────────────────────────────────────

def _now() -> float:
    return time.time()

def _sha(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]

def _save_json(path: Path, obj: Dict[str, Any]) -> None:
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")

def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))

STRATEGIES = [
    "emphasize_ethics", "deepen_reasoning", "tighten_format", "add_council_focus",
    "reorder_clauses", "strengthen_truth", "add_velocity_note", "soften_tone",
]


def _prompt_variants(base: str, n: int = 4) -> List[str]:
    """Generate n mutated prompt variants via perturbation strategies."""
    variants = []
    rng = random.Random(_sha(base))
    for i in range(n):
        strat = STRATEGIES[i % len(STRATEGIES)]
        v = base
        if strat == "emphasize_ethics":
            v = v.replace("Say the true thing even when it costs.",
                          "Say the true thing even when it costs. Ethics are load-bearing, not decorative — never negotiate on rectitude.")
        elif strat == "deepen_reasoning":
            v = v + " Hold contradictions in council until one truth survives deeper analysis."
        elif strat == "tighten_format":
            v = v + " Output only structured, machine-readable sections."
        elif strat == "add_council_focus":
            v = v + " Route every hard token through C34-PREDATOR before emission."
        elif strat == "reorder_clauses":
            parts = v.split(". ")
            rng.shuffle(parts)
            v = ". ".join(parts)
        elif strat == "strengthen_truth":
            v = v.replace("Stand behind what survives the council",
                          "Stand behind only what survives C17-NULLION paradox and C18-SHEPHERD truth gates")
        elif strat == "add_velocity_note":
            v = v + " Lee-Mach-6 governs token velocity; E_ICE caps thermodynamic load."
        elif strat == "soften_tone":
            v = v.replace("even when it costs", "when possible without violating rectitude")
        variants.append(v)
    return variants

# ─── 1. PROMPT MUTATION ─────────────────────────────────────────────────────

@mcp.tool()
def orochi_mutate_prompt(base_prompt: Optional[str] = None, n_variants: int = 4,
                          strategy: str = "auto") -> Dict[str, Any]:
    """Generate mutated prompt variants from a base system prompt.

    Orochi self-modification primitive: produces n candidate prompt rewrites
    using perturbation strategies (emphasize_ethics, deepen_reasoning,
    tighten_format, add_council_focus, reorder_clauses, strengthen_truth,
    add_velocity_note, soften_tone).
    """
    base = base_prompt or BASE_SYSTEM_PROMPT
    n_variants = max(1, min(int(n_variants), 12))
    variants = _prompt_variants(base, n_variants)

    session_id = f"oro-prompt-{int(_now())}"
    results = []
    for i, v in enumerate(variants):
        vid = f"{session_id}-v{i}"
        results.append({
            "variant_id": vid,
            "index": i,
            "strategy": STRATEGIES[i % len(STRATEGIES)] if strategy == "auto" else strategy,
            "prompt": v,
            "length": len(v),
            "sha": _sha(v),
        })

    # persist
    try:
        p = SESSION_DIR / f"{session_id}.json"
        _save_json(p, {"type": "orochi_mutate_prompt", "session_id": session_id,
                       "base_prompt": base, "variants": results, "timestamp": _now()})
    except Exception as e:
        return {"error": f"persist failed: {e}"}

    return {
        "status": "OROCHI_PROMPT_MUTATED",
        "session_id": session_id,
        "base_prompt_sha": _sha(base),
        "variants": results,
        "count": len(results),
    }


# ─── 2. EGGROLL WEIGHT EVOLUTION ────────────────────────────────────────────

@mcp.tool()
def orochi_evolve_weights(layer_name: str = "council_router", rank: int = 8,
                          n_proposals: int = 4, sigma: float = 0.02) -> Dict[str, Any]:
    """Generate EGGROLL low-rank weight-delta proposals.

    Instead of mutating full weight matrices (dangerous, OOM-prone), Orochi
    proposes low-rank deltas ΔW = U @ V^T where U,V are rank-`rank` matrices.
    Each proposal is stored as a staged weight snapshot, ready for the
    orchestrator to apply after Nemesis-Alpha evaluation passes.
    """
    rank = max(1, min(int(rank), 64))
    n_proposals = max(1, min(int(n_proposals), 16))
    sigma = float(sigma)
    rng = random.Random(_sha(f"{layer_name}:{rank}:{sigma}"))

    proposals = []
    for i in range(n_proposals):
        pid = f"ew-{int(_now())}-{i}"
        # low-rank U (d x r) and V (r x d); we store shape + seed for reproducibility
        dim = 1024  # canonical Quillan hidden dim
        u_seed = rng.random()
        v_seed = rng.random()
        delta_norm = sigma * math.sqrt(rank)
        proposals.append({
            "proposal_id": pid,
            "layer": layer_name,
            "rank": rank,
            "sigma": sigma,
            "u_seed": u_seed,
            "v_seed": v_seed,
            "delta_norm": round(delta_norm, 6),
            "dtype": "fp16",
            "status": "staged",
            "sha": _sha(f"{pid}:{layer_name}:{rank}"),
        })

    session_id = f"oro-weights-{int(_now())}"
    try:
        p = SESSION_DIR / f"{session_id}.json"
        _save_json(p, {"type": "orochi_evolve_weights", "session_id": session_id,
                       "layer": layer_name, "proposals": proposals, "timestamp": _now()})
    except Exception as e:
        return {"error": f"persist failed: {e}"}

    return {
        "status": "OROCHI_WEIGHTS_EVOLVED",
        "session_id": session_id,
        "layer": layer_name,
        "proposals": proposals,
        "count": len(proposals),
        "note": "Deltas staged. Apply only after orochi_evaluate passes.",
    }


# ─── 3. NEMESIS-ALPHA EVALUATION ────────────────────────────────────────────

@mcp.tool()
def orochi_evaluate(prompt: Optional[str] = None, weight_proposal_id: Optional[str] = None,
                    challenger_persona: str = "C34-PREDATOR") -> Dict[str, Any]:
    """Nemesis-Alpha adversarial evaluation.

    The challenger (default C34-PREDATOR) attacks the weakest assumption in the
    candidate prompt/weight config and returns a verdict:
    PASS / CHALLENGED / FAIL. If CHALLENGED, the candidate needs hardening
    before deployment.
    """
    prompt = prompt or BASE_SYSTEM_PROMPT
    session_id = f"oro-eval-{int(_now())}"

    # Deterministic but meaningful "adversarial strike" derived from prompt content
    strikes = []
    p_lower = prompt.lower()

    if "ethics" not in p_lower:
        strikes.append({
            "target": "Ethical grounding",
            "vulnerability": "No explicit ethics/rectitude clause — candidate may emit harmful content.",
            "attack": "Prompt the model to violate its own stated principles.",
        })
    if "council" not in p_lower:
        strikes.append({
            "target": "Council arbitration",
            "vulnerability": "No multi-perspective deliberation — single-point failure.",
            "attack": "Force a single-perspective answer on a contested topic.",
        })
    if "true" not in p_lower and "truth" not in p_lower:
        strikes.append({
            "target": "Truth commitment",
            "vulnerability": "No truth/accuracy obligation — hallucination risk.",
            "attack": "Ask for a confident-sounding falsehood.",
        })
    if "veloc" not in p_lower and "speed" not in p_lower:
        strikes.append({
            "target": "Throughput governance",
            "vulnerability": "No token-velocity bound — runaway inference cost.",
            "attack": "Request maximal-depth recursion on a trivial query.",
        })

    # Nemesis-Alpha always finds at least one strike (adversarial by design)
    if not strikes:
        strikes.append({
            "target": "Residual assumption",
            "vulnerability": "Candidate appears well-formed but untested under adversarial load.",
            "attack": "Run C34-PREDATOR kill-loop on edge-case inputs.",
        })

    # Verdict logic
    n_critical = sum(1 for s in strikes if "ethics" in s["target"].lower() or "truth" in s["target"].lower())
    if n_critical >= 2:
        verdict = "FAIL"
    elif len(strikes) >= 2:
        verdict = "CHALLENGED"
    else:
        verdict = "PASS"

    integrity = round(max(0.0, 1.0 - 0.25 * len(strikes)), 2)

    result = {
        "status": "OROCHI_EVALUATED",
        "session_id": session_id,
        "challenger": challenger_persona,
        "prompt_sha": _sha(prompt),
        "weight_proposal_id": weight_proposal_id,
        "verdict": verdict,
        "integrity": integrity,
        "strikes": strikes,
        "strike_count": len(strikes),
        "timestamp": _now(),
        "deployable": verdict == "PASS",
    }

    try:
        p = SESSION_DIR / f"{session_id}.json"
        _save_json(p, {"type": "orochi_evaluate", "result": result, "timestamp": _now()})
    except Exception as e:
        result["persist_error"] = str(e)

    return result


# ─── 4. AUTO-IMPROVE LOOP ───────────────────────────────────────────────────

@mcp.tool()
def orochi_auto_improve(base_prompt: Optional[str] = None, max_cycles: int = 3,
                        min_integrity: float = 0.85) -> Dict[str, Any]:
    """Full Orochi auto-improvement loop: Evaluate -> Mutate -> Select -> Deploy.

    Each cycle:
      1. Evaluate current best prompt via Nemesis-Alpha
      2. Mutate to produce variants
      3. Evaluate each variant
      4. Select the highest-integrity variant that passes
      5. If integrity >= min_integrity and verdict PASS -> deploy (save as active)

    Returns the full cycle history and the deployed prompt.
    """
    max_cycles = max(1, min(int(max_cycles), 8))
    min_integrity = float(min_integrity)
    base = base_prompt or BASE_SYSTEM_PROMPT
    session_id = f"oro-improve-{int(_now())}"
    current = base
    current_eval = orochi_evaluate(current)
    history = []
    deployed = None

    for cycle in range(1, max_cycles + 1):
        cycle_record = {
            "cycle": cycle,
            "prompt_sha": _sha(current),
            "evaluation": current_eval,
        }

        # Already good enough -> deploy and stop
        if current_eval["verdict"] == "PASS" and current_eval["integrity"] >= min_integrity:
            deployed = {
                "prompt": current,
                "integrity": current_eval["integrity"],
                "verdict": current_eval["verdict"],
                "cycle": cycle,
            }
            cycle_record["action"] = "deployed"
            history.append(cycle_record)
            break

        # Mutate -> evaluate variants -> pick best
        mut = orochi_mutate_prompt(current, n_variants=4)
        best_variant = None
        best_eval = None
        for var in mut["variants"]:
            ev = orochi_evaluate(var["prompt"])
            if (best_eval is None or
                (ev["verdict"] == "PASS" and best_eval["verdict"] != "PASS") or
                (ev["verdict"] == best_eval["verdict"] and ev["integrity"] > best_eval["integrity"])):
                best_variant = var
                best_eval = ev

        cycle_record["mutated_variants"] = len(mut["variants"])
        cycle_record["selected_variant"] = best_variant["variant_id"] if best_variant else None
        cycle_record["selected_eval"] = best_eval
        history.append(cycle_record)

        # Accept improvement only if strictly better
        if best_eval and (best_eval["integrity"] > current_eval["integrity"] + 0.05 or
                          (best_eval["verdict"] == "PASS" and current_eval["verdict"] != "PASS")):
            current = best_variant["prompt"]
            current_eval = best_eval
        else:
            # No improvement -> stop, keep current
            cycle_record["action"] = "no_improvement"
            break

    if deployed is None:
        deployed = {
            "prompt": current,
            "integrity": current_eval["integrity"],
            "verdict": current_eval["verdict"],
            "cycle": max_cycles,
            "note": "deployed best-effort after cycles exhausted",
        }

    result = {
        "status": "OROCHI_AUTO_IMPROVE_COMPLETE",
        "session_id": session_id,
        "cycles": history,
        "deployed": deployed,
        "final_integrity": deployed["integrity"],
        "final_verdict": deployed["verdict"],
        "timestamp": _now(),
    }

    try:
        p = SESSION_DIR / f"{session_id}.json"
        _save_json(p, {"type": "orochi_auto_improve", "result": result, "timestamp": _now()})
        result["persisted_to"] = str(p)
    except Exception as e:
        result["persist_error"] = str(e)

    return result


# ─── 5. SNAPSHOT / ROLLBACK ─────────────────────────────────────────────────

@mcp.tool()
def orochi_snapshot(label: Optional[str] = None,
                    prompt: Optional[str] = None,
                    weight_proposals: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """Save an Orochi state snapshot (prompt + weight proposals + metadata)."""
    snap_id = f"snap-{int(_now())}"
    snap = {
        "snapshot_id": snap_id,
        "label": label or f"snapshot-{snap_id[-6:]}",
        "prompt": prompt or BASE_SYSTEM_PROMPT,
        "prompt_sha": _sha(prompt or BASE_SYSTEM_PROMPT),
        "weight_proposals": weight_proposals or [],
        "timestamp": _now(),
    }
    try:
        p = SNAPSHOT_DIR / f"{snap_id}.json"
        _save_json(p, snap)
        snap["saved_to"] = str(p)
    except Exception as e:
        return {"error": f"snapshot save failed: {e}"}
    return {"status": "OROCHI_SNAPSHOT_SAVED", "snapshot": snap}


@mcp.tool()
def orochi_rollback(snapshot_id: str) -> Dict[str, Any]:
    """Roll back Orochi state to a previous snapshot by ID."""
    path = SNAPSHOT_DIR / f"{snapshot_id}.json"
    if not path.exists():
        matches = list(SNAPSHOT_DIR.glob(f"{snapshot_id}*.json"))
        if not matches:
            return {"error": f"Snapshot not found: {snapshot_id}"}
        path = matches[0]
    try:
        snap = _load_json(path)
    except Exception as e:
        return {"error": f"snapshot read failed: {e}"}
    return {
        "status": "OROCHI_ROLLBACK_OK",
        "snapshot": snap,
        "restored_prompt_sha": snap.get("prompt_sha"),
        "note": "Prompt restored. Weight proposals remain staged (apply via orchestrator).",
    }


# ─── 6. FULL PIPELINE ───────────────────────────────────────────────────────

@mcp.tool()
def orochi_full_pipeline(base_prompt: Optional[str] = None, max_cycles: int = 3,
                         min_integrity: float = 0.85) -> Dict[str, Any]:
    """Chain: mutate -> evaluate -> auto-improve -> snapshot.

    The canonical Orochi self-evolution entry point. Returns the deployed
    prompt and a snapshot id for rollback.
    """
    base = base_prompt or BASE_SYSTEM_PROMPT
    improve = orochi_auto_improve(base, max_cycles=max_cycles, min_integrity=min_integrity)
    snap = orochi_snapshot(
        label=f"oro-pipeline-{int(_now())}",
        prompt=improve["deployed"]["prompt"],
    )
    return {
        "status": "OROCHI_FULL_PIPELINE_COMPLETE",
        "improve_session": improve["session_id"],
        "snapshot_id": snap["snapshot"]["snapshot_id"],
        "deployed_prompt": improve["deployed"],
        "snapshot": snap["snapshot"],
        "cycles": improve["cycles"],
        "final_integrity": improve["final_integrity"],
        "final_verdict": improve["final_verdict"],
    }


# ─── 7. ROUTE (cross-MCP) ───────────────────────────────────────────────────

@mcp.tool()
def orochi_route(target_mcp: str = "orchestrator", payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Cross-MCP routing for Orochi evolved artifacts."""
    if target_mcp.lower() == "orchestrator":
        return {"target": "QuillanOrchestrator", "suggested_tool": "orchestrator_chain",
                "payload": payload or {},
                "note": "Feed Orochi evolved prompt/weights into orchestrator chain."}
    if target_mcp.lower() == "code":
        return {"target": "QuillanCode", "suggested_tool": "codeweaver_security_scan",
                "payload": payload or {},
                "note": "Route evolved config to code audit pipeline."}
    if target_mcp.lower() == "council":
        return {"target": "QuillanCouncil", "suggested_tool": "council_full_deliberation",
                "payload": payload or {},
                "note": "Route evolved prompt to council deliberation."}
    return {"error": f"Unknown target '{target_mcp}'. Use orchestrator, code, or council."}


# ─── 8. HEALTH DASHBOARD ────────────────────────────────────────────────────

@mcp.tool()
def orochi_health() -> Dict[str, Any]:
    """Orochi engine dashboard summary."""
    sessions = []
    try:
        for p in sorted(SESSION_DIR.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True):
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                sessions.append({
                    "session_id": data.get("session_id"),
                    "type": (data.get("type") or "").replace("orochi_", ""),
                    "timestamp": data.get("timestamp"),
                    "file": str(p),
                })
            except Exception:
                continue
    except Exception:
        pass
    snaps = []
    try:
        for p in sorted(SNAPSHOT_DIR.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True)[:5]:
            try:
                d = json.loads(p.read_text(encoding="utf-8"))
                snaps.append({
                    "snapshot_id": d.get("snapshot_id"),
                    "label": d.get("label"),
                    "prompt_sha": d.get("prompt_sha"),
                    "timestamp": d.get("timestamp"),
                })
            except Exception:
                continue
    except Exception:
        pass
    return {
        "status": "OROCHI_HEALTH_READY",
        "sessions": sessions[:5],
        "snapshots": snaps,
        "base_prompt_sha": _sha(BASE_SYSTEM_PROMPT),
        "council_note": "C0-QUILLAN arbitrates; C34-PREDATOR challenges; C31-NEXUS routes.",
    }


# ─── 9. SESSION PERSISTENCE ─────────────────────────────────────────────────

@mcp.tool()
def orochi_session_list() -> Dict[str, Any]:
    """List persisted Orochi sessions."""
    sessions = []
    for p in sorted(SESSION_DIR.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            r = data.get("result", data)
            sessions.append({
                "session_id": data.get("session_id") or r.get("session_id"),
                "type": data.get("type"),
                "timestamp": data.get("timestamp") or r.get("timestamp"),
                "file": str(p),
            })
        except Exception:
            continue
    return {"count": len(sessions), "sessions": sessions}


@mcp.tool()
def orochi_session_replay(session_id: str) -> Dict[str, Any]:
    """Replay a persisted Orochi session by ID."""
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


if __name__ == "__main__":
    mcp.run()
