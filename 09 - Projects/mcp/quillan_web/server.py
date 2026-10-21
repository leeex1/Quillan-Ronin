#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🌐 QUILLAN-RONIN WEB ENGINE — NATIVE FAST-MCP SERVER
=======================================================
Embodying C29-NAVIGATOR (Ecosystem), C24-SCHEMA (Structure),
C15-LUMINARIS (Clarity), C13-WARDEN (Safety):
  1. web_scaffold — stack-aware route + file plan
  2. web_audit — a11y / SEO / perf checklist from code
  3. web_deploy_checklist — target-aware ship gate
  4. web_full_pipeline — scaffold -> audit -> deploy checklist (persisted)
  5. web_route — cross-MCP routing payload (council/code/buddy)
  6. web_health — web project dashboard summary
  7. web_session_list / web_session_replay — persistence
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

mcp = FastMCP("QuillanWeb")

def _detect_framework(s: str):
    if "next" in s:
        files = ["app/layout.tsx", "app/page.tsx"] + [f"app/{x.strip('/ ') or 'home'}/page.tsx" for x in r[1:]] + ["next.config.mjs", "tailwind.config.ts"]
    elif "astro" in s:
        files = ["src/pages/index.astro"] + [f"src/pages/{x.strip('/ ')}.astro" for x in r[1:]] + ["astro.config.mjs"]
    elif "vanilla" in s:
        files = ["index.html", "styles.css", "main.js"]
    else:
        files = ["index.html", "src/main.ts", "src/router.ts", "src/styles.css", "vite.config.ts"]
    return {
        "status": "WEB_SCAFFOLDED",
        "project": project_name,
        "stack": stack,
        "routes": r,
        "files": files,
        "conventions": ["C24-SCHEMA: routes mirror nav", "C15-LUMINARIS: <1.5s LCP budget", "C29-NAVIGATOR: deep-linkable /dev servers"],
    }

# ─── 2. AUDIT ───────────────────────────────────────────────────────────────

@mcp.tool()
def web_audit(code_or_html: str) -> Dict[str, Any]:
    """Audit HTML/TSX for a11y, SEO, perf anti-patterns."""
    issues = []
    if re.search(r"<img(?![^>]*alt=)", code_or_html, re.I):
        issues.append({"severity": "HIGH", "check": "a11y-img-alt", "fix": "Add meaningful alt to every ."})
    if "<title>" not in code_or_html.lower() and "<head" in code_or_html.lower():
        issues.append({"severity": "MEDIUM", "check": "seo-title", "fix": "Add unique <title> + meta description."})
    if re.search(r"style\s*=\s*['\"]", code_or_html):
        issues.append({"severity": "LOW", "check": "perf-inline-style", "fix": "Move inline styles to CSS; reduce CLS."})
    if "onclick=" in code_or_html.lower():
        issues.append({"severity": "MEDIUM", "check": "a11y-interactive", "fix": "Use <button> with keyboard handlers, not onclick divs."})
    return {"issue_count": len(issues), "status": "PASS" if not issues else "WARN", "issues": issues, "budgets": {"LCP_s": 1.5, "CLS": 0.1, "JS_kb": 170}}

# ─── 3. DEPLOY CHECKLIST ────────────────────────────────────────────────────

@mcp.tool()
def web_deploy_checklist(target: str = "github-pages") -> Dict[str, Any]:
    """Target-aware deploy gate."""
    base = ["pnpm build passes clean", "404 + fallback route tested", "env vars in vault (no .env commit)", "lighthouse mobile >=90", "C13-WARDEN: no secrets in bundle (grep api_key)"]
    extra = {"vercel": ["vercel --prod preview OK", "edge regions pinned"], "github-pages": ["base path set", "gh-pages branch clean"], "cloudflare": ["wrangler pages publish OK"], "local": ["serve dist/ smoke test"]}.get(target.strip().lower(), [])
    return {"target": target, "checklist": base + extra, "gate": "No ship if secrets leak or LCP regression."}

# ─── 4. FULL PIPELINE (persisted) ───────────────────────────────────────────

@mcp.tool()
def web_full_pipeline(project_name: str, stack: str = "vite+ts", routes: str = "/, /dev, /lore",
                      code_or_html: str = "", target: str = "github-pages", persist: bool = True) -> Dict[str, Any]:
    """Chain scaffold -> audit -> deploy checklist."""
    scaffold = web_scaffold(project_name=project_name, stack=stack, routes=routes)
    audit = web_audit(code_or_html=code_or_html) if code_or_html else {"skipped": True}
    deploy = web_deploy_checklist(target=target)

    session_id = f"web-{int(time.time())}"
    result = {
        "status": "WEB_PIPELINE_COMPLETE",
        "session_id": session_id,
        "scaffold": scaffold,
        "audit": audit,
        "deploy_checklist": deploy,
        "overall": "PASS" if (audit.get("status") in ("PASS", "skipped")) else "REVIEW",
        "timestamp": time.time(),
    }

    if persist:
        try:
            p = SESSION_DIR / f"{session_id}.json"
            p.write_text(json.dumps({"type": "web_pipeline", "result": result}, indent=2), encoding="utf-8")
            result["persisted_to"] = str(p)
        except Exception as e:
            result["persist_error"] = str(e)

    return result

# ─── 5. ROUTE (cross-MCP) ───────────────────────────────────────────────────

@mcp.tool()
def web_route(target_mcp: str = "council", payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Cross-MCP routing payload for council/code/buddy integration."""
    if target_mcp.lower() == "council":
        return {"target": "QuillanCouncil", "suggested_tool": "council_full_deliberation",
                "payload": payload or {}, "note": "Route web project objective to council deliberation."}
    if target_mcp.lower() == "code":
        return {"target": "QuillanCode", "suggested_tool": "codeweaver_full_pipeline",
                "payload": payload or {}, "note": "Route web code to code scan pipeline."}
    if target_mcp.lower() == "buddy":
        return {"target": "RoninBuddy", "suggested_tool": "buddy_mirror_collaborate",
                "payload": payload or {}, "note": "Route web design to mirror-collaboration."}
    return {"error": f"Unknown target '{target_mcp}'. Use council, code, or buddy."}

# ─── 6. HEALTH DASHBOARD ────────────────────────────────────────────────────

@mcp.tool()
def web_health() -> Dict[str, Any]:
    """Web project dashboard summary."""
    sessions = []
    try:
        for p in sorted(SESSION_DIR.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True):
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                sessions.append({
                    "session_id": data.get("session_id"),
                    "project": (data.get("scaffold") or {}).get("project"),
                    "overall": data.get("overall"),
                    "timestamp": data.get("timestamp"),
                })
            except Exception:
                continue
    except Exception:
        pass
    return {
        "status": "WEB_HEALTH_READY",
        "sessions": sessions[:5],
        "council_note": "C29-NAVIGATOR routes; C24-SCHEMA structures; C15-LUMINARIS clarifies.",
    }

# ─── 9. SESSION PERSISTENCE ─────────────────────────────────────────────────

@mcp.tool()
def web_session_list() -> Dict[str, Any]:
    """List persisted web sessions."""
    sessions = []
    for p in sorted(SESSION_DIR.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            sessions.append({
                "session_id": data.get("session_id"),
                "project": (data.get("scaffold") or {}).get("project"),
                "overall": data.get("overall"),
                "timestamp": data.get("timestamp"),
                "file": str(p),
            })
        except Exception:
            continue
    return {"count": len(sessions), "sessions": sessions}

@mcp.tool()
def web_session_replay(session_id: str) -> Dict[str, Any]:
    """Replay a persisted web session by ID."""
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


mcp.add_middleware(QuillanErrorMiddleware())
if __name__ == "__main__":
    mcp.run()
