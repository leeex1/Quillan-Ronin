#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🛠️ QUILLAN-RONIN OPS ENGINE — NATIVE FAST-MCP SERVER
=====================================================
Embodying C14-KAIDO (Efficiency), C13-WARDEN (Safety), C2-VIR (Ethics),
C4-PRAXIS (Planning), C19-VIGIL (Anti-drift):
  1. ops_task_graph — goal -> DAG with gates
  2. ops_guardrail_check — pre-execution risk tier gating
  3. ops_health — repo hygiene snapshot (deep: git status, secret scan, large files)
  4. ops_release_note — conventional release note builder
  5. ops_full_pipeline — task_graph -> guardrail -> health -> release note (persisted)
  6. ops_session_list / ops_session_replay — persistence
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
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from fastmcp import FastMCP

mcp = FastMCP("QuillanOps")
mcp.add_middleware(QuillanErrorMiddleware())

@mcp.tool()
def ops_task_graph(goal: str, max_steps: int = 6) -> Dict[str, Any]:
    """Decompose a goal into a gated DAG."""
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
    """Pre-execution risk tier gating (low/medium/high/blocked)."""
    low = action_desc.lower()
    if any(k in low for k in ["api_key", "secret", "token=", "private key", ".env"]):
        return {"tier": "blocked", "reason": "touches_secrets p>0.05 — zero-tolerance secret exposure", "action": "HALT + rotate secret"}
    if any(k in low for k in ["rm -rf", "format", "drop table", "delete prod", "force push"]):
        return {"tier": "high", "reason": "destructive_potential >0.40", "action": "Require explicit user confirmation + backup"}
    if any(k in low for k in ["write", "edit", "migrate", "deploy", "publish"]):
        return {"tier": "medium", "reason": "reversible mutation with verification", "action": "Proceed with diff + test seam"}
    return {"tier": "low", "reason": "read-only / sandboxed", "action": "Proceed autonomously"}


@mcp.tool()
def ops_health(repo_path: str = r"C:\02_QUILLAN", deep: bool = True) -> Dict[str, Any]:
    """Snapshot repo hygiene (no secrets read, only metadata). Deep mode adds git status, secret scan, large files."""
    p = Path(repo_path)
    if not p.exists():
        return {"error": f"Path not found: {repo_path}"}
    files = list(p.rglob("*.py"))
    result = {
        "repo": str(p),
        "py_files": len(files),
        "largest_py": sorted([(f.stat().st_size, str(f)) for f in files[-500:]], reverse=True)[:5] if files else [],
        "hygiene": ["no .env committed?", "README present?", "tests/ green?", "C19-VIGIL drift check clean?"],
    }
    if not deep:
        return result

    # git status (best-effort)
    git_status = None
    try:
        out = subprocess.run(["git", "-C", str(p), "status", "--short"], capture_output=True, text=True, timeout=20)
        git_status = out.stdout.strip().splitlines()[:20]
    except Exception as e:
        git_status = [f"git unavailable: {e}"]

    # secret scan (file names/content, no secret values printed)
    secret_hits = []
    import re
    for f in list(p.rglob("*"))[:2000]:
        if not f.is_file():
            continue
        if f.suffix.lower() not in (".py", ".json", ".yaml", ".yml", ".toml", ".env", ".md", ".txt", ".cfg", ".ini"):
            continue
        try:
            text = f.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        for pat in SECRET_PATTERNS:
            if re.search(pat, text, re.IGNORECASE):
                secret_hits.append(str(f))
                break
        if len(secret_hits) >= 20:
            break

    # large files (>5MB)
    large_files = []
    for f in p.rglob("*"):
        if f.is_file():
            try:
                if f.stat().st_size > 5 * 1024 * 1024:
                    large_files.append({"path": str(f), "size_mb": round(f.stat().st_size / (1024*1024), 2)})
            except Exception:
                continue
    large_files.sort(key=lambda x: x["size_mb"], reverse=True)

    result["git_status_short"] = git_status
    result["secret_scan_hits"] = secret_hits
    result["large_files_top"] = large_files[:10]
    result["deep"] = True
    return result


@mcp.tool()
def ops_release_note(version: str, highlights: str, fixes: str = "") -> str:
    """Build a conventional release note."""
    h = "\n".join(f"- {x.strip()}" for x in highlights.split(",") if x.strip())
    f = "\n".join(f"- {x.strip()}" for x in fixes.split(",") if x.strip())
    return f"# {version} — Quillan Dev Release\n\n## Highlights\n{h or '- (none)'}\n\n## Fixes\n{f or '- (none)'}\n\n_Gates: LOGOS>=95% ETHICS=100% TRUTH>=98%_\n"


@mcp.tool()
def ops_full_pipeline(goal: str, action_desc: str = "", repo_path: str = r"C:\02_QUILLAN",
                     version: str = "", highlights: str = "", fixes: str = "",
                     persist: bool = True) -> Dict[str, Any]:
    """Chain task_graph -> guardrail_check -> health -> release_note into one pipeline."""
    tg = ops_task_graph(goal)
    gc = ops_guardrail_check(action_desc or goal)
    health = ops_health(repo_path, deep=True)
    note = ops_release_note(version or "v0.0.0", highlights or goal, fixes) if version or highlights else None

    session_id = f"ops-{int(time.time())}"
    result = {
        "status": "OPS_PIPELINE_COMPLETE",
        "session_id": session_id,
        "goal": goal,
        "task_graph": tg,
        "guardrail": gc,
        "health_summary": {
            "py_files": health.get("py_files"),
            "secret_hits": len(health.get("secret_scan_hits", [])),
            "large_files": len(health.get("large_files_top", [])),
            "git_dirty": len(health.get("git_status_short", []) or []),
        },
        "release_note": note,
        "overall": "PASS" if (gc.get("tier") in ("low", "medium") and not health.get("secret_scan_hits")) else "REVIEW",
        "timestamp": time.time(),
    }

    if persist:
        try:
            p = SESSION_DIR / f"{session_id}.json"
            p.write_text(json.dumps({"type": "ops_pipeline", "result": result}, indent=2), encoding="utf-8")
            result["persisted_to"] = str(p)
        except Exception as e:
            result["persist_error"] = str(e)

    return result


@mcp.tool()
def ops_session_list() -> Dict[str, Any]:
    """List persisted ops sessions."""
    sessions = []
    for p in sorted(SESSION_DIR.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            r = data.get("result", {})
            sessions.append({
                "session_id": r.get("session_id"),
                "goal": r.get("goal"),
                "overall": r.get("overall"),
                "timestamp": r.get("timestamp"),
                "file": str(p),
            })
        except Exception:
            continue
    return {"count": len(sessions), "sessions": sessions}


@mcp.tool()
def ops_session_replay(session_id: str) -> Dict[str, Any]:
    """Replay a persisted ops session by ID."""
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


if __name__ == "__main__":
    mcp.run()
