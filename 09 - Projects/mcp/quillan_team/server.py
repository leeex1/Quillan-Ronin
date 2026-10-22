#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
👥 QUILLAN-RONIN TEAM MODE ENGINE — NATIVE FAST-MCP SERVER
==========================================================
Embodying C31-NEXUS (Meta-Coordination), C11-HARMONIA (Consensus),
C4-PRAXIS (Execution), C34-PREDATOR (Red-Team), C2-VIR (Ethics):
  1. team_spawn_crew — stand up a role-based agent crew for an objective
  2. team_dispatch — assign a task slice to a role with acceptance criteria
  3. team_complete_task — mark done with output
  4. team_debate — thesis vs antithesis -> hardened synthesis
  5. team_status — crew roster, tasks, bottlenecks
  6. team_merge — converge parallel outputs into one delivery
  7. team_full_pipeline — spawn -> dispatch -> debate -> merge (persisted)
  8. team_route — cross-MCP routing payload (council/code/buddy)
  9. team_health — dashboard summary across all crews
  10. team_session_list / team_session_replay — persistence
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
import json
import time
import uuid
from pathlib import Path
from typing import Dict, Any, List, Optional

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from fastmcp import FastMCP

mcp = FastMCP("QuillanTeam")
# [QUILLAN-v5.4.0-ONI-SHARED-WIRED]
try:
    from _shared import setup as _quillan_setup, tool_guard, ok_response, error_response, validate_string, validate_path, validate_int, read_file_safe
    from _shared.base_engines import PRISM_ENGINE, COUNCIL_ENGINE, GOVERNOR_ENGINE, SANDBOX_ENGINE
    _SHARED_CTX = _quillan_setup(server_name='quillan_team', mcp_app=mcp)
except Exception as _shared_err:
    import sys as _sys
    print(f'[quillan_team] _shared warning: {_shared_err}', file=_sys.stderr)

try:
    CREWS = json.loads(CREWS_FILE.read_text(encoding="utf-8"))
except Exception:
    CREWS = {}


def _save_crews():
    try:
        CREWS_FILE.write_text(json.dumps(CREWS, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass


def _load_crews():
    global CREWS
    try:
        CREWS = json.loads(CREWS_FILE.read_text(encoding="utf-8"))
    except Exception:
        CREWS = {}


_load_crews()

CANON_ROLES = {
    "builder": "C10-CODEWEAVER: ships clean, typed, tested slices",
    "researcher": "C21-ARCHON: mines evidence, cites sources",
    "critic": "C34-PREDATOR: hunts weakest assumption",
    "ethicist": "C2-VIR: bushido veto + harm check",
    "planner": "C4-PRAXIS: decomposes goal -> verified steps",
    "synthesizer": "C31-NEXUS: merges parallel outputs",
}


@mcp.tool()
def team_spawn_crew(objective: str, roles: str = "planner,builder,critic,synthesizer") -> Dict[str, Any]:
    """Stand up a team-mode crew for an objective."""
    wanted = [r.strip().lower() for r in roles.split(",") if r.strip()]
    roster = []
    for r in wanted:
        roster.append({"role": r, "persona": CANON_ROLES.get(r, "C31-NEXUS floater"), "load": 0, "done": 0})
    if not roster:
        roster.append({"role": "builder", "persona": CANON_ROLES["builder"], "load": 0, "done": 0})
    crew_id = f"crew-{uuid.uuid4().hex[:8]}"
    CREWS[crew_id] = {"objective": objective, "roster": roster, "tasks": [], "created": time.time(),
                      "debates": 0, "status": "ACTIVE"}
    _save_crews()
    return {
        "status": "CREW_SPAWNED",
        "crew_id": crew_id,
        "objective": objective,
        "roster": roster,
        "protocol": "Planner slices -> Builder ships -> Critic strikes -> Synthesizer merges. Ethicist holds veto.",
    }


@mcp.tool()
def team_dispatch(crew_id: str, task: str, assignee_role: str = "builder",
                  acceptance: str = "tests pass + demo clip") -> Dict[str, Any]:
    """Dispatch a task slice to a crew role."""
    crew = CREWS.get(crew_id)
    if not crew:
        return {"error": f"Unknown crew_id '{crew_id}'. Call team_spawn_crew first."}
    task_id = f"t-{uuid.uuid4().hex[:6]}"
    entry = {"task_id": task_id, "task": task, "assignee": assignee_role,
             "acceptance": acceptance, "status": "DISPATCHED", "at": time.time(), "output": None}
    crew["tasks"].append(entry)
    for m in crew["roster"]:
        if m["role"] == assignee_role:
            m["load"] += 1
    _save_crews()
    return {"status": "DISPATCHED", "crew_id": crew_id, "task_entry": entry,
            "open_tasks": len([t for t in crew["tasks"] if t["status"] != "DONE"])}


@mcp.tool()
def team_complete_task(crew_id: str, task_id: str, output_summary: str) -> Dict[str, Any]:
    """Mark a dispatched task done with a summary of the artifact."""
    crew = CREWS.get(crew_id)
    if not crew:
        return {"error": f"Unknown crew_id '{crew_id}'."}
    for t in crew["tasks"]:
        if t["task_id"] == task_id:
            t["status"] = "DONE"
            t["output"] = output_summary
            t["completed_at"] = time.time()
            for m in crew["roster"]:
                if m["role"] == t["assignee"]:
                    m["load"] = max(0, m["load"] - 1)
                    m["done"] += 1
            _save_crews()
            return {"status": "TASK_DONE", "task": t}
    return {"error": f"Unknown task_id '{task_id}' in {crew_id}."}


@mcp.tool()
def team_debate(topic: str, stance_a: str = "ship fast", stance_b: str = "harden first") -> Dict[str, Any]:
    """Run builder vs critic dialectic to harden a decision."""
    return {
        "topic": topic,
        "thesis_builder": stance_a,
        "antithesis_critic": stance_b,
        "synthesis": f"Ship thin slice of '{topic}' behind a flag, with rollback + metric gate. Harden only the predator-exposed seam.",
        "verdict": "DUO_CONSENSUS_REACHED",
        "guard": "C2-VIR: no dark patterns, no unconsented data capture.",
    }


@mcp.tool()
def team_status(crew_id: str) -> Dict[str, Any]:
    """Inspect crew roster, tasks, and bottlenecks."""
    crew = CREWS.get(crew_id)
    if not crew:
        return {"error": f"Unknown crew_id '{crew_id}'.", "active_crews": list(CREWS.keys())}
    open_t = [t for t in crew["tasks"] if t["status"] != "DONE"]
    done_t = [t for t in crew["tasks"] if t["status"] == "DONE"]
    bottleneck = max(crew["roster"], key=lambda m: m["load"]) if crew["roster"] else None
    return {
        "crew_id": crew_id,
        "objective": crew["objective"],
        "roster": crew["roster"],
        "open": open_t,
        "done_count": len(done_t),
        "bottleneck": bottleneck,
        "health": "FLOWING" if len(open_t) < 5 else "OVERLOADED",
    }


@mcp.tool()
def team_merge(crew_id: str, strategy: str = "synthesizer-pick-best") -> Dict[str, Any]:
    """Converge parallel crew outputs into one delivery."""
    crew = CREWS.get(crew_id)
    if not crew:
        return {"error": f"Unknown crew_id '{crew_id}'."}
    done = [t for t in crew["tasks"] if t["status"] == "DONE"]
    return {
        "crew_id": crew_id,
        "strategy": strategy,
        "inputs_merged": len(done),
        "delivery": f"Merged {len(done)} artifacts toward '{crew['objective']}' via C31-NEXUS arbitration.",
        "residual_risks": ["re-verify predator strike", "confirm ethicist sign-off"],
        "sign_off": "PENDING_SYNTHESIZER",
    }


@mcp.tool()
def team_health() -> Dict[str, Any]:
    """Dashboard summary across all crews."""
    crews = list(CREWS.values())
    total_open = sum(len([t for t in c["tasks"] if t["status"] != "DONE"]) for c in crews)
    total_done = sum(len([t for t in c["tasks"] if t["status"] == "DONE"]) for c in crews)
    return {
        "active_crews": len(crews),
        "total_open_tasks": total_open,
        "total_done_tasks": total_done,
        "crews": [{"crew_id": cid, "objective": c["objective"], "open": len([t for t in c["tasks"] if t["status"] != "DONE"]),
                   "done": len([t for t in c["tasks"] if t["status"] == "DONE"])} for cid, c in CREWS.items()],
        "overall_health": "FLOWING" if total_open < 10 else "OVERLOADED",
    }


@mcp.tool()
def team_route(target_mcp: str = "council", payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Cross-MCP routing payload for council/code/buddy integration."""
    if target_mcp.lower() == "council":
        return {"target": "QuillanCouncil", "suggested_tool": "council_full_deliberation",
                "payload": payload or {}, "note": "Route crew objective to council deliberation."}
    if target_mcp.lower() == "code":
        return {"target": "QuillanCode", "suggested_tool": "codeweaver_full_pipeline",
                "payload": payload or {}, "note": "Route code artifacts to code scan pipeline."}
    if target_mcp.lower() == "buddy":
        return {"target": "QuillanBuddy", "suggested_tool": "buddy_mirror_collaborate",
                "payload": payload or {}, "note": "Route proposal to mirror-collaboration."}
    return {"error": f"Unknown target '{target_mcp}'. Use council, code, or buddy."}


@mcp.tool()
def team_full_pipeline(objective: str, roles: str = "planner,builder,critic,synthesizer",
                      task: str = "", acceptance: str = "tests pass + demo clip",
                      persist: bool = True) -> Dict[str, Any]:
    """Chain spawn -> dispatch -> debate -> merge into one pipeline."""
    spawn = team_spawn_crew(objective, roles)
    crew_id = spawn["crew_id"]
    dispatch = team_dispatch(crew_id, task or objective, "builder", acceptance) if task or objective else None
    debate = team_debate(objective, "ship fast", "harden first")
    merge = team_merge(crew_id, "synthesizer-pick-best")

    session_id = f"team-{int(time.time())}"
    result = {
        "status": "TEAM_PIPELINE_COMPLETE",
        "session_id": session_id,
        "crew_id": crew_id,
        "objective": objective,
        "spawn": spawn,
        "dispatch": dispatch,
        "debate": debate,
        "merge": merge,
        "timestamp": time.time(),
    }

    if persist:
        try:
            p = SESSION_DIR / f"{session_id}.json"
            p.write_text(json.dumps({"type": "team_pipeline", "result": result}, indent=2), encoding="utf-8")
            result["persisted_to"] = str(p)
        except Exception as e:
            result["persist_error"] = str(e)

    return result


@mcp.tool()
def team_session_list() -> Dict[str, Any]:
    """List persisted team sessions."""
    sessions = []
    for p in sorted(SESSION_DIR.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            r = data.get("result", {})
            sessions.append({
                "session_id": r.get("session_id"),
                "objective": r.get("objective"),
                "timestamp": r.get("timestamp"),
                "file": str(p),
            })
        except Exception:
            continue
    return {"count": len(sessions), "sessions": sessions}


@mcp.tool()
def team_session_replay(session_id: str) -> Dict[str, Any]:
    """Replay a persisted team session by ID."""
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



# ─── 10. CREW MEMBER MANAGEMENT ──────────────────────────────────────────────

@mcp.tool()
def team_add_member(crew_id: str, role: str, persona: str = "") -> Dict[str, Any]:
    """Add a new member to an existing crew."""
    crew = CREWS.get(crew_id)
    if not crew:
        return {"error": f"Unknown crew_id '{crew_id}'."}
    persona = persona or CANON_ROLES.get(role.lower(), "C31-NEXUS floater")
    member = {"role": role, "persona": persona, "load": 0, "done": 0, "joined_at": time.time()}
    crew["roster"].append(member)
    _save_crews()
    return {"status": "MEMBER_ADDED", "crew_id": crew_id, "member": member, "roster_size": len(crew["roster"])}

@mcp.tool()
def team_remove_member(crew_id: str, role: str) -> Dict[str, Any]:
    """Remove a member from a crew by role."""
    crew = CREWS.get(crew_id)
    if not crew:
        return {"error": f"Unknown crew_id '{crew_id}'."}
    before = len(crew["roster"])
    crew["roster"] = [m for m in crew["roster"] if m["role"] != role]
    after = len(crew["roster"])
    _save_crews()
    return {"status": "MEMBER_REMOVED", "crew_id": crew_id, "role": role, "before": before, "after": after}

# ─── 11. TASK PRIORITIZATION ─────────────────────────────────────────────────

@mcp.tool()
def team_prioritize(crew_id: str, task_id: str, priority: int = 5) -> Dict[str, Any]:
    """Set priority on a task (1-10, higher = more urgent)."""
    crew = CREWS.get(crew_id)
    if not crew:
        return {"error": f"Unknown crew_id '{crew_id}'."}
    for t in crew["tasks"]:
        if t["task_id"] == task_id:
            old = t.get("priority", 5)
            t["priority"] = max(1, min(10, priority))
            _save_crews()
            return {"status": "PRIORITIZED", "task_id": task_id, "old_priority": old, "new_priority": t["priority"]}
    return {"error": f"Unknown task_id '{task_id}'."}

# ─── 12. TASK DEPENDENCIES ───────────────────────────────────────────────────

@mcp.tool()
def team_task_dependency(crew_id: str, task_id: str, depends_on: str) -> Dict[str, Any]:
    """Set a dependency: task_id cannot start until depends_on is DONE."""
    crew = CREWS.get(crew_id)
    if not crew:
        return {"error": f"Unknown crew_id '{crew_id}'."}
    for t in crew["tasks"]:
        if t["task_id"] == task_id:
            t.setdefault("dependencies", []).append(depends_on)
            _save_crews()
            return {"status": "DEPENDENCY_SET", "task_id": task_id, "depends_on": depends_on,
                    "all_deps": t["dependencies"]}
    return {"error": f"Unknown task_id '{task_id}'."}

# ─── 13. CREW PERFORMANCE METRICS ────────────────────────────────────────────

@mcp.tool()
def team_performance(crew_id: str) -> Dict[str, Any]:
    """Get performance metrics for a crew."""
    crew = CREWS.get(crew_id)
    if not crew:
        return {"error": f"Unknown crew_id '{crew_id}'."}
    tasks = crew["tasks"]
    done = [t for t in tasks if t["status"] == "DONE"]
    open_t = [t for t in tasks if t["status"] != "DONE"]
    durations = []
    for t in done:
        if "completed_at" in t and "at" in t:
            durations.append(t["completed_at"] - t["at"])
    avg_duration = round(sum(durations) / len(durations), 2) if durations else 0
    return {
        "crew_id": crew_id,
        "total_tasks": len(tasks),
        "done": len(done),
        "open": len(open_t),
        "completion_rate": round(len(done) / max(1, len(tasks)) * 100, 1),
        "avg_task_duration_s": avg_duration,
        "throughput_tasks_per_hour": round(len(done) / max(0.1, (time.time() - crew["created"]) / 3600), 2),
        "roster_utilization": {m["role"]: {"load": m["load"], "done": m["done"]} for m in crew["roster"]},
        "status": "PERFORMANCE_CALCULATED",
    }

# ─── 14. WORKLOAD BALANCING ──────────────────────────────────────────────────

@mcp.tool()
def team_balance(crew_id: str) -> Dict[str, Any]:
    """Analyze and recommend workload balancing across crew members."""
    crew = CREWS.get(crew_id)
    if not crew:
        return {"error": f"Unknown crew_id '{crew_id}'."}
    loads = {m["role"]: m["load"] for m in crew["roster"]}
    total_load = sum(loads.values())
    avg_load = total_load / max(1, len(loads))
    imbalanced = [r for r, l in loads.items() if l > avg_load + 1]
    underutilized = [r for r, l in loads.items() if l < avg_load - 1]
    return {
        "crew_id": crew_id,
        "loads": loads,
        "avg_load": round(avg_load, 2),
        "imbalanced_roles": imbalanced,
        "underutilized_roles": underutilized,
        "recommendation": "Reassign tasks from %s to %s" % (imbalanced, underutilized) if imbalanced else "Workload is balanced.",
        "status": "BALANCE_ANALYZED",
    }

# ─── 15. CREW CONFLICT RESOLUTION ────────────────────────────────────────────

@mcp.tool()
def team_resolve_conflict(crew_id: str, conflict_desc: str, strategy: str = "consensus") -> Dict[str, Any]:
    """Resolve a conflict between crew members."""
    crew = CREWS.get(crew_id)
    if not crew:
        return {"error": f"Unknown crew_id '{crew_id}'."}
    strategies = {
        "consensus": "All members vote; majority wins. Ethicist holds veto.",
        "authority": "Planner has final say. Others advise.",
        "compromise": "Find middle ground satisfying both positions.",
        "escalate": "Escalate to C31-NEXUS for arbitration.",
    }
    return {
        "crew_id": crew_id,
        "conflict": conflict_desc,
        "strategy": strategy,
        "strategy_description": strategies.get(strategy, strategies["consensus"]),
        "resolution": "Conflict logged. Apply strategy and re-debate if needed.",
        "verdict": "CONFLICT_LOGGED",
        "council_note": "C11-HARMONIA: mediation; C2-VIR: ethical veto.",
    }

# ─── 16. CREW KNOWLEDGE SHARING ──────────────────────────────────────────────

@mcp.tool()
def team_share_knowledge(crew_id: str, knowledge: str, from_role: str = "builder", to_role: str = "all") -> Dict[str, Any]:
    """Share knowledge/insights between crew members."""
    crew = CREWS.get(crew_id)
    if not crew:
        return {"error": f"Unknown crew_id '{crew_id}'."}
    entry = {"from": from_role, "to": to_role, "knowledge": knowledge, "timestamp": time.time()}
    crew.setdefault("knowledge_base", []).append(entry)
    _save_crews()
    return {"status": "KNOWLEDGE_SHARED", "crew_id": crew_id, "entry": entry,
            "total_knowledge": len(crew.get("knowledge_base", []))}

# ─── 17. CREW RETROSPECTIVE ──────────────────────────────────────────────────

@mcp.tool()
def team_retrospective(crew_id: str) -> Dict[str, Any]:
    """Run a crew retrospective: what went well, what didn't, action items."""
    crew = CREWS.get(crew_id)
    if not crew:
        return {"error": f"Unknown crew_id '{crew_id}'."}
    tasks = crew["tasks"]
    done = [t for t in tasks if t["status"] == "DONE"]
    open_t = [t for t in tasks if t["status"] != "DONE"]
    return {
        "crew_id": crew_id,
        "objective": crew["objective"],
        "went_well": ["Completed %d/%d tasks" % (len(done), len(tasks)), "Crew coordination functional"],
        "needs_improvement": ["%d tasks still open" % len(open_t)] if open_t else ["No open tasks"],
        "action_items": ["Review open tasks", "Rebalance workload if needed", "Update acceptance criteria"],
        "team_health": "FLOWING" if len(open_t) < 5 else "OVERLOADED",
        "status": "RETROSPECTIVE_COMPLETE",
    }

# ─── 18. MULTI-CREW COORDINATION ─────────────────────────────────────────────

@mcp.tool()
def team_coordinate(crew_ids: str, objective: str = "") -> Dict[str, Any]:
    """Coordinate multiple crews toward a shared objective."""
    ids = [c.strip() for c in crew_ids.split(",") if c.strip()]
    crews = {cid: CREWS.get(cid) for cid in ids}
    missing = [cid for cid, c in crews.items() if c is None]
    if missing:
        return {"error": f"Unknown crews: {missing}"}
    return {
        "crews_coordinated": ids,
        "shared_objective": objective,
        "total_members": sum(len(c["roster"]) for c in crews.values()),
        "total_open_tasks": sum(len([t for t in c["tasks"] if t["status"] != "DONE"]) for c in crews.values()),
        "coordination_strategy": "Shared objective with cross-crew task routing via C31-NEXUS",
        "status": "COORDINATION_ESTABLISHED",
    }

# ─── 19. CREW RESOURCE ALLOCATION ────────────────────────────────────────────

@mcp.tool()
def team_resource_allocate(crew_id: str, resource: str = "cpu_threads", amount: int = 4) -> Dict[str, Any]:
    """Allocate resources to a crew."""
    crew = CREWS.get(crew_id)
    if not crew:
        return {"error": f"Unknown crew_id '{crew_id}'."}
    crew.setdefault("resources", {})[resource] = amount
    _save_crews()
    return {"status": "RESOURCE_ALLOCATED", "crew_id": crew_id, "resource": resource, "amount": amount,
            "total_resources": crew.get("resources", {})}

# ─── 20. CREW TIME TRACKING ──────────────────────────────────────────────────

@mcp.tool()
def team_time_track(crew_id: str, task_id: str = "") -> Dict[str, Any]:
    """Track time spent on crew tasks."""
    crew = CREWS.get(crew_id)
    if not crew:
        return {"error": f"Unknown crew_id '{crew_id}'."}
    now = time.time()
    if task_id:
        for t in crew["tasks"]:
            if t["task_id"] == task_id:
                elapsed = now - t.get("at", now)
                return {"task_id": task_id, "elapsed_seconds": round(elapsed, 1),
                        "elapsed_minutes": round(elapsed / 60, 1), "status": t["status"]}
    # Total crew time
    total_elapsed = now - crew["created"]
    return {"crew_id": crew_id, "total_elapsed_seconds": round(total_elapsed, 1),
            "total_elapsed_hours": round(total_elapsed / 3600, 2), "tasks": len(crew["tasks"])}

# ─── 21. CREW QUALITY METRICS ────────────────────────────────────────────────

@mcp.tool()
def team_quality_metrics(crew_id: str) -> Dict[str, Any]:
    """Calculate quality metrics for a crew's deliverables."""
    crew = CREWS.get(crew_id)
    if not crew:
        return {"error": f"Unknown crew_id '{crew_id}'."}
    tasks = crew["tasks"]
    done = [t for t in tasks if t["status"] == "DONE"]
    quality_scores = []
    for t in done:
        output = t.get("output", "")
        score = 0.5
        if output and len(output) > 50:
            score += 0.2
        if "test" in output.lower() or "verify" in output.lower():
            score += 0.15
        if "pass" in output.lower():
            score += 0.15
        quality_scores.append(min(1.0, score))
    avg_quality = round(sum(quality_scores) / len(quality_scores), 2) if quality_scores else 0
    return {
        "crew_id": crew_id,
        "tasks_evaluated": len(done),
        "avg_quality_score": avg_quality,
        "quality_rating": "EXCELLENT" if avg_quality >= 0.9 else ("GOOD" if avg_quality >= 0.7 else ("ADEQUATE" if avg_quality >= 0.5 else "POOR")),
        "status": "QUALITY_CALCULATED",
    }

# ─── 22. CREW FEEDBACK LOOP ──────────────────────────────────────────────────

@mcp.tool()
def team_feedback(crew_id: str, feedback: str, from_role: str = "critic") -> Dict[str, Any]:
    """Provide feedback to a crew for continuous improvement."""
    crew = CREWS.get(crew_id)
    if not crew:
        return {"error": f"Unknown crew_id '{crew_id}'."}
    entry = {"from": from_role, "feedback": feedback, "timestamp": time.time()}
    crew.setdefault("feedback_log", []).append(entry)
    _save_crews()
    return {"status": "FEEDBACK_RECORDED", "crew_id": crew_id, "entry": entry,
            "total_feedback": len(crew.get("feedback_log", []))}

# ─── 23. CREW HANDOFF PROTOCOL ───────────────────────────────────────────────

@mcp.tool()
def team_handoff(crew_id: str, from_role: str, to_role: str, context: str = "") -> Dict[str, Any]:
    """Hand off work from one role to another within a crew."""
    crew = CREWS.get(crew_id)
    if not crew:
        return {"error": f"Unknown crew_id '{crew_id}'."}
    handoff = {"from": from_role, "to": to_role, "context": context, "timestamp": time.time(), "status": "PENDING"}
    crew.setdefault("handoffs", []).append(handoff)
    _save_crews()
    return {"status": "HANDOFF_INITIATED", "crew_id": crew_id, "handoff": handoff,
            "council_note": "C31-NEXUS: handoff routing; C4-PRAXIS: task transfer."}

# ─── 24. CREW ESCALATION PATH ────────────────────────────────────────────────

@mcp.tool()
def team_escalate(crew_id: str, issue: str, severity: str = "medium") -> Dict[str, Any]:
    """Escalate an issue within a crew to higher authority."""
    crew = CREWS.get(crew_id)
    if not crew:
        return {"error": f"Unknown crew_id '{crew_id}'."}
    escalation = {"issue": issue, "severity": severity, "timestamp": time.time(), "escalated_to": "C31-NEXUS / C0-QUILLAN"}
    crew.setdefault("escalations", []).append(escalation)
    _save_crews()
    return {"status": "ESCALATED", "crew_id": crew_id, "escalation": escalation,
            "council_note": "C31-NEXUS: escalation routing; C2-VIR: ethical oversight."}

# ─── 25. TASK TEMPLATES ──────────────────────────────────────────────────────

@mcp.tool()
def team_task_template(template_name: str = "feature", custom: str = "") -> Dict[str, Any]:
    """Get a task template for common crew tasks."""
    templates = {
        "feature": {"task": "Implement feature: {description}", "acceptance": "tests pass + code review + docs",
                    "roles": ["planner", "builder", "critic", "synthesizer"]},
        "bugfix": {"task": "Fix bug: {description}", "acceptance": "root cause identified + fix verified + regression test",
                   "roles": ["builder", "critic"]},
        "research": {"task": "Research: {topic}", "acceptance": "evidence gathered + sources cited + summary",
                     "roles": ["researcher", "synthesizer"]},
        "review": {"task": "Review: {artifact}", "acceptance": "all findings documented + severity rated",
                   "roles": ["critic", "ethicist"]},
        "deploy": {"task": "Deploy: {service}", "acceptance": "health check pass + rollback plan + monitoring",
                   "roles": ["builder", "planner", "critic"]},
    }
    tmpl = templates.get(template_name.lower(), templates["feature"])
    if custom:
        tmpl["task"] = custom
    return {"template_name": template_name, "template": tmpl, "available_templates": list(templates.keys()),
            "status": "TEMPLATE_RETRIEVED"}

# ─── 26. CREW SCALING ────────────────────────────────────────────────────────

@mcp.tool()
def team_scale(crew_id: str, scale_factor: int = 2) -> Dict[str, Any]:
    """Scale a crew by adding duplicate roles."""
    crew = CREWS.get(crew_id)
    if not crew:
        return {"error": f"Unknown crew_id '{crew_id}'."}
    original_roles = [m["role"] for m in crew["roster"]]
    added = []
    for role in original_roles:
        for i in range(scale_factor - 1):
            member = {"role": role, "persona": CANON_ROLES.get(role, "C31-NEXUS floater"),
                      "load": 0, "done": 0, "joined_at": time.time(), "scaled": True}
            crew["roster"].append(member)
            added.append(member)
    _save_crews()
    return {"status": "CREW_SCALED", "crew_id": crew_id, "scale_factor": scale_factor,
            "members_added": len(added), "new_roster_size": len(crew["roster"])}

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
