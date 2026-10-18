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

CREWS_FILE = Path(r"C:\02_QUILLAN\09 - Projects\mcp\quillan_team\crews.json")
SESSION_DIR = Path(r"C:\02_QUILLAN\09 - Projects\mcp\quillan_team\sessions")
SESSION_DIR.mkdir(parents=True, exist_ok=True)

CREWS: Dict[str, Dict[str, Any]] = {}


def _load_crews():
    global CREWS
    try:
        CREWS = json.loads(CREWS_FILE.read_text(encoding="utf-8"))
    except Exception:
        CREWS = {}


def _save_crews():
    try:
        CREWS_FILE.write_text(json.dumps(CREWS, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass


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
        return {"target": "RoninBuddy", "suggested_tool": "buddy_mirror_collaborate",
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


if __name__ == "__main__":
    mcp.run()
