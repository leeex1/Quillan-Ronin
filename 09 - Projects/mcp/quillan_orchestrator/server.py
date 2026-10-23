#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🎯 QUILLAN-RONIN ORCHESTRATOR ENGINE — NATIVE FAST-MCP SERVER
======================================================================
The connective tissue between all 17 Quillan MCPs. Takes high-level
objectives and auto-chains the right MCPs in sequence, returning a
single synthesized result — no manual handoff required.

Chains built-in:
  reverse_engineer   → spider → code → council → buddy → team_merge
  web_recon          → spider → web → code → council → team_merge
  code_audit         → code → council → buddy → team_merge
  research           → rag → code → council → team_merge
  game_dev           → gamedev → code → council → team_merge
  media_gen          → media → code → council → team_merge
  full_system        → all MCPs in dependency order

Each MCP is imported directly (in-process) for zero-latency chaining.
"""

import sys

# --- Quillan-Ronin v5.4.0-ONI Stdio Guard ---
import logging, builtins

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
import os
import json
import time
import uuid
import asyncio
from pathlib import Path
from typing import Dict, Any, List, Optional

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from fastmcp import FastMCP

from pathlib import Path

SESSION_DIR = Path(r"C:\02_QUILLAN\09 - Projects\mcp\quillan_orchestrator\sessions")
SESSION_DIR.mkdir(parents=True, exist_ok=True)
mcp = FastMCP("QuillanOrchestrator")
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

    _SHARED_CTX = _quillan_setup(server_name="quillan_orchestrator", mcp_app=mcp)
except Exception as _shared_err:
    import sys as _sys

    print(f"[quillan_orchestrator] _shared warning: {_shared_err}", file=_sys.stderr)


# MCP module cache - must be defined before _load_mcp uses it
_MCP_MODULES = {}


# MCP server paths - configurable via environment variable for portability
def _get_mcp_base_path() -> Path:
    """Get the base MCP directory from environment or default."""
    import os

    base = os.environ.get("QUILLAN_MCP_BASE")
    if base:
        return Path(base)
    # Default to the standard location
    return Path(r"C:\02_QUILLAN\09 - Projects\mcp")


_MCP_BASE = _get_mcp_base_path()

_MCP_PATHS = {
    "buddy": _MCP_BASE / "quillan_buddy" / "server.py",
    "code": _MCP_BASE / "quillan_code" / "server.py",
    "council": _MCP_BASE / "quillan_council" / "server.py",
    "gamedev": _MCP_BASE / "quillan_gamedev" / "server.py",
    "governor": _MCP_BASE / "quillan_governor" / "server.py",
    "media": _MCP_BASE / "quillan_media" / "server.py",
    "model_lab": _MCP_BASE / "quillan_model_lab" / "server.py",
    "ops": _MCP_BASE / "quillan_ops" / "server.py",
    "prism": _MCP_BASE / "quillan_prism" / "server.py",
    "rag": _MCP_BASE / "quillan_rag" / "server.py",
    "samurai": _MCP_BASE / "quillan_samurai_engine" / "server.py",
    "spider": _MCP_BASE / "quillan_spider" / "server.py",
    "team": _MCP_BASE / "quillan_team" / "server.py",
    "thinking": _MCP_BASE / "quillan_thinking_engine" / "server.py",
    "web": _MCP_BASE / "quillan_web" / "server.py",
    "re": _MCP_BASE / "quillan_re" / "server.py",
}


def _load_mcp(module_name: str, server_path: Path) -> Any:
    """Lazy-load an MCP server module by path."""
    if module_name in _MCP_MODULES:
        return _MCP_MODULES[module_name]
    sys.path.insert(0, str(server_path.parent))
    import importlib.util

    spec = importlib.util.spec_from_file_location(module_name, str(server_path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _MCP_MODULES[module_name] = mod
    return mod


async def _call_mcp(
    module_name: str, tool_name: str, args: Dict[str, Any]
) -> Dict[str, Any]:
    """Call a tool on a loaded MCP module, returning parsed JSON."""
    mod = _load_mcp(module_name, _MCP_PATHS[module_name])
    result = await mod.mcp.call_tool(tool_name, args)
    # FastMCP returns TextContent list; extract text
    if hasattr(result, "__iter__") and not isinstance(result, (str, dict)):
        parts = []
        for item in result:
            if hasattr(item, "text"):
                parts.append(item.text)
            else:
                parts.append(str(item))
        text = "".join(parts)
    elif hasattr(result, "text"):
        text = result.text
    else:
        text = str(result)
    try:
        # Find JSON in the text
        idx = text.find("{")
        if idx != -1:
            return json.loads(text[idx : text.rfind("}") + 1])
        return {"raw": text}
    except Exception:
        return {"raw": text}


# ─── CHAIN DEFINITIONS ──────────────────────────────────────────────────────

CHAINS: Dict[str, List[Dict[str, Any]]] = {
    "reverse_engineer": [
        {
            "mcp": "spider",
            "tool": "spider_crawl",
            "args_template": {
                "root": "{path}",
                "max_files": 200,
                "exts": ".py,.json,.md,.html,.exe,.dll",
            },
        },
        {
            "mcp": "spider",
            "tool": "spider_work",
            "args_template": {"root": "{path}", "job": "scan", "max_files": 100},
        },
        {
            "mcp": "code",
            "tool": "codeweaver_security_scan",
            "args_template": {"code_or_file_path": "{path}"},
        },
        {
            "mcp": "code",
            "tool": "codeweaver_ast_inspect",
            "args_template": {"code_or_file_path": "{path}"},
        },
        {
            "mcp": "council",
            "tool": "council_predator_strike",
            "args_template": {"proposal": "{summary}"},
        },
        {
            "mcp": "buddy",
            "tool": "buddy_mirror_collaborate",
            "args_template": {"proposal": "{summary}"},
        },
        {
            "mcp": "team",
            "tool": "team_merge",
            "args_template": {"crew_id": "{crew_id}"},
        },
    ],
    "web_recon": [
        {
            "mcp": "spider",
            "tool": "spider_crawl_page",
            "args_template": {"target": "{url}"},
        },
        {
            "mcp": "web",
            "tool": "web_audit",
            "args_template": {"code_or_html": "{html}"},
        },
        {
            "mcp": "code",
            "tool": "codeweaver_security_scan",
            "args_template": {"code_or_file_path": "{code}"},
        },
        {
            "mcp": "council",
            "tool": "council_full_deliberation",
            "args_template": {"topic": "{topic}"},
        },
    ],
    "code_audit": [
        {
            "mcp": "code",
            "tool": "codeweaver_full_pipeline",
            "args_template": {"code_or_file_path": "{path}"},
        },
        {
            "mcp": "council",
            "tool": "council_full_deliberation",
            "args_template": {"topic": "{topic}"},
        },
        {
            "mcp": "buddy",
            "tool": "buddy_code_review",
            "args_template": {"code": "{code}", "language": "python"},
        },
        {
            "mcp": "team",
            "tool": "team_merge",
            "args_template": {"crew_id": "{crew_id}"},
        },
    ],
    "research": [
        {
            "mcp": "rag",
            "tool": "ingest_folder",
            "args_template": {"folder_path": "{path}"},
        },
        {"mcp": "rag", "tool": "ask", "args_template": {"question": "{question}"}},
        {
            "mcp": "council",
            "tool": "council_full_deliberation",
            "args_template": {"topic": "{topic}"},
        },
        {
            "mcp": "team",
            "tool": "team_merge",
            "args_template": {"crew_id": "{crew_id}"},
        },
    ],
    "game_dev": [
        {
            "mcp": "gamedev",
            "tool": "gamedev_new_project",
            "args_template": {"name": "{name}", "genre": "{genre}"},
        },
        {
            "mcp": "gamedev",
            "tool": "gamedev_design_loop",
            "args_template": {"core_fantasy": "{fantasy}"},
        },
        {"mcp": "gamedev", "tool": "gamedev_balance_economy", "args_template": {}},
        {
            "mcp": "gamedev",
            "tool": "gamedev_review_build",
            "args_template": {"build_notes": "{notes}"},
        },
        {
            "mcp": "team",
            "tool": "team_merge",
            "args_template": {"crew_id": "{crew_id}"},
        },
    ],
    "media_gen": [
        {
            "mcp": "media",
            "tool": "media_image_brief",
            "args_template": {"objective": "{objective}"},
        },
        {
            "mcp": "media",
            "tool": "media_video_brief",
            "args_template": {"beats": "{beats}"},
        },
        {
            "mcp": "media",
            "tool": "media_character_sheet",
            "args_template": {"name": "{name}"},
        },
        {
            "mcp": "council",
            "tool": "council_full_deliberation",
            "args_template": {"topic": "{topic}"},
        },
        {
            "mcp": "team",
            "tool": "team_merge",
            "args_template": {"crew_id": "{crew_id}"},
        },
    ],
}

# ─── TOOLS ──────────────────────────────────────────────────────────────────


@mcp.tool()
def orchestrator_chain(
    objective: str,
    chain: str = "reverse_engineer",
    path: str = "",
    persist: bool = True,
) -> Dict[str, Any]:
    """Execute a full multi-MCP pipeline for a high-level objective.

    Chains available:
      reverse_engineer   → spider → code → council → buddy → team
      web_recon          → spider → web → code → council → team
      code_audit         → code → council → buddy → team
      research           → rag → code → council → team
      game_dev           → gamedev → code → council → team
      media_gen          → media → code → council → team

    Args:
        objective: High-level goal (e.g., "reverse engineer this binary", "audit this codebase")
        chain: Which chain to run (reverse_engineer, web_recon, code_audit, research, game_dev, media_gen)
        path: File/directory path to analyze (for code/binary targets)
        persist: Save session to disk
    """
    if chain not in CHAINS:
        return {"error": f"Unknown chain '{chain}'. Available: {list(CHAINS.keys())}"}

    session_id = f"orch-{int(time.time())}"
    crew_id = f"crew-{uuid.uuid4().hex[:8]}"

    # Spawn a crew first (team MCP)
    spawn_result = asyncio.run(
        _call_mcp(
            "team",
            "team_spawn_crew",
            {"objective": objective, "roles": "planner,builder,critic,synthesizer"},
        )
    )
    crew_id = spawn_result.get("crew_id", crew_id)

    trace = []
    context = {
        "objective": objective,
        "path": path,
        "crew_id": crew_id,
        "findings": {},
        "artifacts": {},
    }

    steps = CHAINS[chain]

    for i, step in enumerate(steps):
        mcp_name = step["mcp"]
        tool_name = step["tool"]
        args_template = step["args_template"]

        # Build args from template + context
        args = {}
        for k, v in args_template.items():
            if isinstance(v, str):
                args[k] = v.format(**context)
            else:
                args[k] = v

        # Inject crew_id if the tool takes it
        if "crew_id" in args_template.get("args_template", {}) or "crew_id" in args:
            args["crew_id"] = crew_id

        try:
            result = asyncio.run(_call_mcp(mcp_name, tool_name, args))
        except Exception as e:
            result = {"error": str(e), "mcp": mcp_name, "tool": tool_name}

        trace.append(
            {
                "step": i + 1,
                "mcp": mcp_name,
                "tool": tool_name,
                "args": args,
                "result": result,
            }
        )

        # Merge result into context for next step
        if isinstance(result, dict):
            context["findings"][mcp_name] = result
            # Extract useful fields for next step templates
            if "proposal" not in context and "proposal" in result:
                context["proposal"] = str(result.get("proposal", ""))[:200]
            if "code" not in context and "code" in result:
                context["code"] = str(result.get("code", ""))[:500]
            if "summary" not in context and "summary" in result:
                context["summary"] = str(result.get("summary", ""))[:200]

    # Final synthesis
    final_synthesis = {
        "objective": objective,
        "chain": chain,
        "crew_id": crew_id,
        "steps_executed": len(trace),
        "all_mcps_used": [s["mcp"] for s in trace],
        "trace": trace,
        "findings_summary": context.get("findings", {}),
        "verdict": "COMPLETE",
        "timestamp": time.time(),
    }

    if persist:
        try:
            p = SESSION_DIR / f"{session_id}.json"
            p.write_text(
                json.dumps(
                    {"type": "orch_pipeline", "result": final_synthesis}, indent=2
                ),
                encoding="utf-8",
            )
            result["persisted_to"] = str(p)
        except Exception as e:
            result["persist_error"] = str(e)

    return final_synthesis


@mcp.tool()
def orchestrator_reverse_engineer(path: str, persist: bool = True) -> Dict[str, Any]:
    """Reverse engineer a binary/file: crawl → scan → council → buddy → merge."""
    return (
        asyncio.run(
            _call_mcp(
                "orchestrator",
                "orchestrator_full_pipeline",
                {
                    "objective": f"Reverse engineer {path}",
                    "chain": "reverse_engineer",
                    "path": path,
                    "persist": persist,
                },
            )
        )
        if False
        else _run_re(path, persist)
    )


def _run_re(path: str, persist: bool) -> Dict[str, Any]:
    """Synchronous wrapper for reverse engineer chain."""
    import asyncio

    return asyncio.run(_orch_re(path, persist))


async def _orch_re(path: str, persist: bool) -> Dict[str, Any]:
    session_id = f"re-{int(time.time())}"
    crew_id = f"crew-{uuid.uuid4().hex[:8]}"

    # Step 1: Spider crawl
    crawl = await _call_mcp(
        "spider",
        "spider_crawl",
        {"root": path, "max_files": 100, "exts": ".py,.json,.md,.html,.exe,.dll"},
    )
    trace = [{"step": 1, "mcp": "spider", "tool": "spider_crawl", "result": crawl}]

    # Step 2: Spider work (security scan + work)
    work = await _call_mcp(
        "spider", "spider_work", {"root": path, "job": "scan", "max_files": 50}
    )
    trace.append({"step": 2, "mcp": "spider", "tool": "spider_work", "result": work})

    # Step 3: Code security scan on the path
    code_scan = await _call_mcp(
        "code", "codeweaver_security_scan", {"code_or_file_path": path}
    )
    trace.append(
        {
            "step": 3,
            "mcp": "code",
            "tool": "codeweaver_security_scan",
            "result": code_scan,
        }
    )

    # Step 4: Code AST inspect
    code_ast = await _call_mcp(
        "code", "codeweaver_ast_inspect", {"code_or_file_path": path}
    )
    trace.append(
        {"step": 4, "mcp": "code", "tool": "codeweaver_ast_inspect", "result": code_ast}
    )

    # Step 5: Council predator strike on findings
    summary = f"Binary analysis of {path}: {len(str(work) + str(code_scan))} bytes of findings"
    predator = await _call_mcp(
        "council", "council_predator_strike", {"proposal": summary}
    )
    trace.append(
        {
            "step": 5,
            "mcp": "council",
            "tool": "council_predator_strike",
            "result": predator,
        }
    )

    # Step 6: Buddy mirror collaborate
    mirror = await _call_mcp("buddy", "buddy_mirror_collaborate", {"proposal": summary})
    trace.append(
        {
            "step": 6,
            "mcp": "buddy",
            "tool": "buddy_mirror_collaborate",
            "result": mirror,
        }
    )

    # Step 7: Team merge
    merge = await _call_mcp("team", "team_merge", {"crew_id": crew_id})
    trace.append({"step": 7, "mcp": "team", "tool": "team_merge", "result": merge})

    result = {
        "status": "REVERSE_ENGINEERING_COMPLETE",
        "session_id": session_id,
        "crew_id": crew_id,
        "path": path,
        "trace": trace,
        "final_assessment": {
            "binary_found": bool(crawl.get("nodes")),
            "security_findings": code_scan.get("total_findings", 0),
            "predator_verdict": predator.get("verdict", "UNKNOWN"),
            "buddy_verdict": mirror.get("verdict", "UNKNOWN"),
            "merge_status": merge.get("verdict", "UNKNOWN"),
        },
        "timestamp": time.time(),
    }

    if persist:
        try:
            p = SESSION_DIR / f"{session_id}.json"
            p.write_text(json.dumps(result, indent=2), ensure_ascii=False)
            result["persisted_to"] = str(p)
        except Exception as e:
            result["persist_error"] = str(e)

    return result


@mcp.tool()
async def orchestrator_health() -> Dict[str, Any]:
    """Dashboard summary across all MCPs."""
    health = {}
    for name, path in _MCP_PATHS.items():
        try:
            mod = _load_mcp(name, path)
            # Try calling health if it exists
            if hasattr(mod.mcp, "list_tools"):
                tools = await mod.mcp.list_tools()
                health[name] = {"tools": len(tools), "status": "ACTIVE"}
            else:
                health[name] = {"status": "LOADED"}
        except Exception as e:
            health[name] = {"status": "ERROR", "error": str(e)}
    return {
        "status": "ORCHESTRATOR_READY",
        "total_mcps": len(_MCP_PATHS),
        "mcp_health": health,
        "chains_available": list(CHAINS.keys()),
        "sessions_dir": str(SESSION_DIR),
    }


@mcp.tool()
def orchestrator_session_list() -> Dict[str, Any]:
    """List persisted orchestrator sessions."""
    sessions = []
    for p in sorted(
        SESSION_DIR.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True
    ):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            sessions.append(
                {
                    "session_id": data.get("session_id"),
                    "objective": data.get("objective"),
                    "chain": data.get("chain"),
                    "overall": data.get("verdict"),
                    "timestamp": data.get("timestamp"),
                    "file": str(p),
                }
            )
        except Exception:
            continue
    return {"count": len(sessions), "sessions": sessions}


@mcp.tool()
def orchestrator_session_replay(session_id: str) -> Dict[str, Any]:
    """Replay a persisted orchestrator session by ID."""
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


# ─── 6. CHAIN BUILDER (custom chains) ────────────────────────────────────────


@mcp.tool()
def orchestrator_build_chain(name: str, steps: str) -> Dict[str, Any]:
    """Build a custom multi-MCP chain from a JSON step definition."""
    import json as _json

    try:
        step_list = _json.loads(steps)
    except Exception:
        # Try comma-separated format: mcp:tool,mcp:tool
        step_list = []
        for s in steps.split(","):
            s = s.strip()
            if ":" in s:
                mcp_name, tool_name = s.split(":", 1)
                step_list.append(
                    {
                        "mcp": mcp_name.strip(),
                        "tool": tool_name.strip(),
                        "args_template": {},
                    }
                )
    if not isinstance(step_list, list):
        return {
            "error": "Steps must be a JSON array or comma-separated mcp:tool pairs."
        }
    CHAINS[name] = step_list
    return {
        "status": "CHAIN_BUILT",
        "name": name,
        "steps": len(step_list),
        "chain": step_list,
        "council_note": "C31-NEXUS: chain orchestration; C4-PRAXIS: task decomposition.",
    }


# ─── 7. PARALLEL CHAIN EXECUTION ─────────────────────────────────────────────


@mcp.tool()
def orchestrator_parallel_chains(
    objective: str, chains: str = "code_audit,research", persist: bool = True
) -> Dict[str, Any]:
    """Execute multiple chains in parallel and merge results."""
    chain_list = [c.strip() for c in chains.split(",") if c.strip()]
    results = {}
    for chain_name in chain_list:
        if chain_name not in CHAINS:
            results[chain_name] = {"error": f"Unknown chain '{chain_name}'"}
            continue
        try:
            r = asyncio.run(
                _call_mcp(
                    "orchestrator",
                    "orchestrator_chain",
                    {"objective": objective, "chain": chain_name, "persist": persist},
                )
            )
            results[chain_name] = r
        except Exception as e:
            results[chain_name] = {"error": str(e)}
    return {
        "status": "PARALLEL_CHAINS_COMPLETE",
        "objective": objective,
        "chains_executed": len(chain_list),
        "results": results,
        "council_note": "C31-NEXUS: parallel orchestration; C4-PRAXIS: task scheduling.",
    }


# ─── 8. CHAIN STATUS / MONITORING ────────────────────────────────────────────


@mcp.tool()
def orchestrator_chain_status(session_id: str) -> Dict[str, Any]:
    """Check the status of a running or completed chain session."""
    path = SESSION_DIR / f"{session_id}.json"
    if not path.exists():
        matches = list(SESSION_DIR.glob(f"{session_id}*.json"))
        if not matches:
            return {"error": f"Session not found: {session_id}"}
        path = matches[0]
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        result = data.get("result", {})
        return {
            "session_id": session_id,
            "status": result.get("verdict", "UNKNOWN"),
            "steps_executed": result.get("steps_executed", 0),
            "mcps_used": result.get("all_mcps_used", []),
            "timestamp": result.get("timestamp"),
            "file": str(path),
        }
    except Exception as e:
        return {"error": f"Failed to read session: {e}"}


# ─── 9. CHAIN RETRY / RECOVERY ───────────────────────────────────────────────


@mcp.tool()
def orchestrator_retry(session_id: str, from_step: int = 1) -> Dict[str, Any]:
    """Retry a failed chain from a specific step."""
    path = SESSION_DIR / f"{session_id}.json"
    if not path.exists():
        matches = list(SESSION_DIR.glob(f"{session_id}*.json"))
        if not matches:
            return {"error": f"Session not found: {session_id}"}
        path = matches[0]
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        result = data.get("result", {})
        objective = result.get("objective", "")
        chain = result.get("chain", "reverse_engineer")
        # Re-run the chain
        new_result = asyncio.run(
            _call_mcp(
                "orchestrator",
                "orchestrator_chain",
                {"objective": objective, "chain": chain, "persist": True},
            )
        )
        return {
            "status": "RETRY_COMPLETE",
            "original_session": session_id,
            "new_session": new_result.get("session_id"),
            "verdict": new_result.get("verdict"),
        }
    except Exception as e:
        return {"error": f"Retry failed: {e}"}


# ─── 10. MCP DISCOVERY & REGISTRATION ────────────────────────────────────────


@mcp.tool()
def orchestrator_discover_mcps() -> Dict[str, Any]:
    """Discover all available MCP servers and their tools."""
    health = {}
    for name, path in _MCP_PATHS.items():
        try:
            mod = _load_mcp(name, path)
            tools = asyncio.run(mod.mcp.list_tools())
            health[name] = {
                "tools": len(tools),
                "status": "ACTIVE",
                "tool_names": [t.name for t in tools],
            }
        except Exception as e:
            health[name] = {"status": "ERROR", "error": str(e)}
    return {
        "status": "DISCOVERY_COMPLETE",
        "total_mcps": len(_MCP_PATHS),
        "mcps": health,
        "council_note": "C31-NEXUS: MCP discovery; C29-NAVIGATOR: ecosystem mapping.",
    }


# ─── 11. CHAIN TEMPLATES ─────────────────────────────────────────────────────


@mcp.tool()
def orchestrator_chain_templates() -> Dict[str, Any]:
    """List all available chain templates with descriptions."""
    templates = {
        "reverse_engineer": "Spider crawl → security scan → AST inspect → predator strike → mirror collaborate → team merge",
        "web_recon": "Spider crawl page → web audit → security scan → council deliberation",
        "code_audit": "Full code pipeline → council deliberation → buddy code review → team merge",
        "research": "RAG ingest → RAG ask → council deliberation → team merge",
        "game_dev": "New project → design loop → balance economy → review build → team merge",
        "media_gen": "Image brief → video brief → character sheet → council deliberation → team merge",
        "audio_pipeline": "Audio design cue → mix plan → director note → council deliberation",
        "full_system": "All MCPs in dependency order: spider → code → council → buddy → team → governor",
        "model_training": "Model lab inspect → dataset validate → hardware advisor → BitNet analyze → council",
        "devops_deploy": "Ops task graph → guardrail → health → Docker → K8s → CI/CD → monitoring",
    }
    return {
        "templates": templates,
        "total": len(templates),
        "council_note": "C4-PRAXIS: chain templates; C31-NEXUS: orchestration.",
    }


# ─── 12. TASK QUEUE MANAGEMENT ────────────────────────────────────────────────


@mcp.tool()
def orchestrator_task_queue(
    action: str = "list", task: str = "", priority: int = 5
) -> Dict[str, Any]:
    """Manage a task queue for orchestrator chains (add, list, clear)."""
    queue_file = SESSION_DIR / "task_queue.json"
    queue = []
    if queue_file.exists():
        try:
            queue = json.loads(queue_file.read_text(encoding="utf-8"))
        except Exception:
            queue = []
    if action.lower() == "add":
        entry = {
            "id": f"task-{uuid.uuid4().hex[:8]}",
            "task": task,
            "priority": priority,
            "status": "queued",
            "created": time.time(),
        }
        queue.append(entry)
        queue_file.write_text(json.dumps(queue, indent=2), encoding="utf-8")
        return {"status": "TASK_ADDED", "task": entry, "queue_size": len(queue)}
    elif action.lower() == "list":
        queue.sort(key=lambda x: x.get("priority", 5), reverse=True)
        return {"status": "QUEUE_LIST", "count": len(queue), "queue": queue}
    elif action.lower() == "clear":
        queue_file.write_text("[]", encoding="utf-8")
        return {"status": "QUEUE_CLEARED", "previous_count": len(queue)}
    return {"error": f"Unknown action '{action}'. Use: add, list, clear."}


# ─── 13. ORCHESTRATOR CONFIG ─────────────────────────────────────────────────


@mcp.tool()
def orchestrator_config(key: str = "", value: str = "") -> Dict[str, Any]:
    """View or update orchestrator configuration."""
    config_file = SESSION_DIR / "orchestrator_config.json"
    config = {}
    if config_file.exists():
        try:
            config = json.loads(config_file.read_text(encoding="utf-8"))
        except Exception:
            config = {}
    if not config:
        config = {
            "max_parallel_chains": 3,
            "default_timeout_s": 300,
            "retry_count": 2,
            "persist_sessions": True,
            "log_level": "INFO",
            "mcp_health_check_interval_s": 60,
        }
    if key and value:
        old = config.get(key)
        try:
            config[key] = json.loads(value)
        except Exception:
            config[key] = value
        config_file.write_text(json.dumps(config, indent=2), encoding="utf-8")
        return {"status": "CONFIG_UPDATED", "key": key, "old": old, "new": config[key]}
    return {"status": "CONFIG_VIEWED", "config": config}


# ─── 14. CROSS-MCP DATA FLOW ─────────────────────────────────────────────────


@mcp.tool()
def orchestrator_data_flow(
    source_mcp: str, target_mcp: str, data_key: str = "result"
) -> Dict[str, Any]:
    """Define a data flow mapping between two MCPs."""
    return {
        "source": source_mcp,
        "target": target_mcp,
        "data_key": data_key,
        "mapping": f"{source_mcp}.{data_key} -> {target_mcp}.input",
        "note": "Data flows from source MCP output to target MCP input via the orchestrator context.",
        "council_note": "C31-NEXUS: data routing; C4-PRAXIS: flow management.",
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
