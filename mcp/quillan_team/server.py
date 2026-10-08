#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
👥 QUILLAN-RONIN TEAM MODE ENGINE — NATIVE FAST-MCP SERVER
===========================================================
Embodying C31-NEXUS (Meta-Coordination), C11-HARMONIA (Consensus),
C4-PRAXIS (Execution), C34-PREDATOR (Red-Team), C2-VIR (Ethics):
  1. team_spawn_crew — stand up a role-based agent crew for an objective
  2. team_dispatch — assign a task slice to a role with acceptance criteria
  3. team_debate — thesis vs antithesis -> hardened synthesis
  4. team_status — crew roster, load, and blockers
  5. team_merge — converge parallel outputs into one coherent delivery
"""

import sys
import time
import uuid
from typing import Dict, Any, List, Optional

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from fastmcp import FastMCP

mcp = FastMCP("QuillanTeam")

CREWS: Dict[str, Dict[str, Any]] = {}

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
    """Stand up a team-mode crew for an objective.

    Args:
        objective: Single-sentence mission (e.g. 'Ship Godot vertical slice combat').
        roles: Comma-separated roles from builder,researcher,critic,ethicist,planner,synthesizer.
    """
    wanted = [r.strip().lower() for r in roles.split(",") if r.strip()]
    roster = []
    for r in wanted:
        roster.append({"role": r, "persona": CANON_ROLES.get(r, "C31-NEXUS floater"), "load": 0, "done": 0})
    if not roster:
        roster.append({"role": "builder", "persona": CANON_ROLES["builder"], "load": 0, "done": 0})
    crew_id = f"crew-{uuid.uuid4().hex[:8]}"
    CREWS[crew_id] = {"objective": objective, "roster": roster, "tasks": [], "created": time.time(), "debates": 0}
    return {
        "status": "CREW_SPAWNED",
        "crew_id": crew_id,
        "objective": objective,
        "roster": roster,
        "protocol": "Planner slices -> Builder ships -> Critic strikes -> Synthesizer merges. Ethicist holds veto.",
    }


@mcp.tool()
def team_dispatch(crew_id: str, task: str, assignee_role: str = "builder", acceptance: str = "tests pass + demo clip") -> Dict[str, Any]:
    """Dispatch a task slice to a crew role.

    Args:
        crew_id: Crew ID from team_spawn_crew.
        task: Task slice (keep < 1 day of work).
        assignee_role: Role to own it.
        acceptance: Done criteria.
    """
    crew = CREWS.get(crew_id)
    if not crew:
        return {"error": f"Unknown crew_id '{crew_id}'. Call team_spawn_crew first."}
    task_id = f"t-{uuid.uuid4().hex[:6]}"
    entry = {"task_id": task_id, "task": task, "assignee": assignee_role, "acceptance": acceptance, "status": "DISPATCHED", "at": time.time()}
    crew["tasks"].append(entry)
    for m in crew["roster"]:
        if m["role"] == assignee_role:
            m["load"] += 1
    return {"status": "DISPATCHED", "crew_id": crew_id, "task_entry": entry, "open_tasks": len([t for t in crew["tasks"] if t["status"] != "DONE"])}


@mcp.tool()
def team_complete_task(crew_id: str, task_id: str, output_summary: str) -> Dict[str, Any]:
    """Mark a dispatched task done with a summary of the artifact.

    Args:
        crew_id: Crew ID.
        task_id: Task ID from team_dispatch.
        output_summary: What was produced + where it lives.
    """
    crew = CREWS.get(crew_id)
    if not crew:
        return {"error": f"Unknown crew_id '{crew_id}'."}
    for t in crew["tasks"]:
        if t["task_id"] == task_id:
            t["status"] = "DONE"
            t["output"] = output_summary
            for m in crew["roster"]:
                if m["role"] == t["assignee"]:
                    m["load"] = max(0, m["load"] - 1)
                    m["done"] += 1
            return {"status": "TASK_DONE", "task": t}
    return {"error": f"Unknown task_id '{task_id}' in {crew_id}."}


@mcp.tool()
def team_debate(topic: str, stance_a: str = "ship fast", stance_b: str = "harden first") -> Dict[str, Any]:
    """Run builder vs critic dialectic to harden a decision.

    Args:
        topic: Decision under debate.
        stance_a: Thesis (builder).
        stance_b: Antithesis (critic).
    """
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
    """Inspect crew roster, tasks, and bottlenecks.

    Args:
        crew_id: Crew ID.
    """
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
    """Converge parallel crew outputs into one delivery.

    Args:
        crew_id: Crew ID.
        strategy: Merge strategy.
    """
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


if __name__ == "__main__":
    mcp.run()
