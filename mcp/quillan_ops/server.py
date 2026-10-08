#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🛠️ QUILLAN-RONIN OPS ENGINE — NATIVE FAST-MCP SERVER
=====================================================
Embodying C14-KAIDO (Efficiency), C13-WARDEN (Safety), C2-VIR (Ethics),
C4-PRAXIS (Planning), C19-VIGIL (Anti-drift):
  1. ops_task_graph — goal -> DAG with gates
  2. ops_guardrail_check — pre-execution risk tier gating
  3. ops_health — repo hygiene snapshot
  4. ops_release_note — conventional release note builder
"""

import sys
from pathlib import Path
from typing import Dict, Any, List

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from fastmcp import FastMCP

mcp = FastMCP("QuillanOps")


@mcp.tool()
def ops_task_graph(goal: str, max_steps: int = 6) -> Dict[str, Any]:
    """Decompose a goal into a gated DAG.

    Args:
        goal: End-state (e.g. 'Land gamedev prototype v0.1').
        max_steps: Cap steps (2-10).
    """
    n = max(2, min(10, max_steps))
    steps = [
        {"id": f"S{i+1}", "title": t, "verify": v, "gate": g}
        for i, (t, v, g) in enumerate([
            ("Slice scope to thin vertical", "scope doc <1 page", "C4-PRAXIS"),
            ("Ship smallest runnable", "runs headless + 60fps probe", "C10-CODEWEAVER"),
            ("Predator strike + fix", "kill-loop closed", "C34-PREDATOR"),
            ("Ethics + safety pass", "VIR/WARDEN sign", "C2-VIR/C13-WARDEN"),
            ("Polish loop + mix", "AURELION>=7/CADENCE drift<=.35", "C22/C23"),
            ("Release + retro", "note + metrics logged", "C5-ECHO"),
        ][:n])
    ]
    return {"goal": goal, "dag": steps, "rule": "No step starts until prior verify passes."}


@mcp.tool()
def ops_guardrail_check(action_desc: str) -> Dict[str, Any]:
    """Pre-execution risk tier gating (low/medium/high/blocked).

    Args:
        action_desc: Proposed tool/filesystem/network action.
    """
    low = action_desc.lower()
    if any(k in low for k in ["api_key", "secret", "token=", "private key", ".env"]):
        return {"tier": "blocked", "reason": "touches_secrets p>0.05 — zero-tolerance secret exposure", "action": "HALT + rotate secret"}
    if any(k in low for k in ["rm -rf", "format", "drop table", "delete prod", "force push"]):
        return {"tier": "high", "reason": "destructive_potential >0.40", "action": "Require explicit user confirmation + backup"}
    if any(k in low for k in ["write", "edit", "migrate", "deploy", "publish"]):
        return {"tier": "medium", "reason": "reversible mutation with verification", "action": "Proceed with diff + test seam"}
    return {"tier": "low", "reason": "read-only / sandboxed", "action": "Proceed autonomously"}


@mcp.tool()
def ops_health(repo_path: str = "C:\\02_QUILLAN") -> Dict[str, Any]:
    """Snapshot repo hygiene (no secrets read, only metadata).

    Args:
        repo_path: Directory to inspect.
    """
    p = Path(repo_path)
    if not p.exists():
        return {"error": f"Path not found: {repo_path}"}
    files = list(p.rglob("*.py"))
    return {
        "repo": str(p),
        "py_files": len(files),
        "largest_py": sorted([(f.stat().st_size, str(f)) for f in files[-200:]], reverse=True)[:5] if files else [],
        "hygiene": ["no .env committed?", "README present?", "tests/ green?", "C19-VIGIL drift check clean?"],
    }


@mcp.tool()
def ops_release_note(version: str, highlights: str, fixes: str = "") -> str:
    """Build a conventional release note.

    Args:
        version: e.g. v0.3.0-gamedev.
        highlights: Comma-separated highlights.
        fixes: Comma-separated fixes.
    """
    h = "\n".join(f"- {x.strip()}" for x in highlights.split(",") if x.strip())
    f = "\n".join(f"- {x.strip()}" for x in fixes.split(",") if x.strip())
    return f"# {version} — Quillan Dev Release\n\n## Highlights\n{h or '- (none)'}\n\n## Fixes\n{f or '- (none)'}\n\n_Gates: LOGOS>=95% ETHICS=100% TRUTH>=98%_\n"


if __name__ == "__main__":
    mcp.run()
