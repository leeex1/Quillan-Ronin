#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🌐 QUILLAN-RONIN WEB ENGINE — NATIVE FAST-MCP SERVER
=====================================================
Embodying C29-NAVIGATOR (Ecosystem), C24-SCHEMA (Structure),
C15-LUMINARIS (Clarity), C13-WARDEN (Safety):
  1. web_scaffold — stack-aware route + file plan
  2. web_audit — a11y / SEO / perf checklist from code
  3. web_deploy_checklist — target-aware ship gate
"""

import sys
import re
from typing import Dict, Any, List

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from fastmcp import FastMCP

mcp = FastMCP("QuillanWeb")


@mcp.tool()
def web_scaffold(project_name: str, stack: str = "vite+ts", routes: str = "/, /dev, /lore") -> Dict[str, Any]:
    """Scaffold a web project plan.

    Args:
        project_name: Folder/project name.
        stack: vite+ts, next, astro, vanilla.
        routes: Comma-separated routes.
    """
    r = [x.strip() for x in routes.split(",") if x.strip()]
    s = stack.lower()
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


@mcp.tool()
def web_audit(code_or_html: str) -> Dict[str, Any]:
    """Audit HTML/TSX for a11y, SEO, perf anti-patterns.

    Args:
        code_or_html: Markup or component source.
    """
    issues = []
    if re.search(r"<img(?![^>]*alt=)", code_or_html, re.I):
        issues.append({"severity": "HIGH", "check": "a11y-img-alt", "fix": "Add meaningful alt to every <img>."})
    if "<title>" not in code_or_html.lower() and "<head" in code_or_html.lower():
        issues.append({"severity": "MEDIUM", "check": "seo-title", "fix": "Add unique <title> + meta description."})
    if re.search(r"style\s*=\s*['\"]", code_or_html):
        issues.append({"severity": "LOW", "check": "perf-inline-style", "fix": "Move inline styles to CSS; reduce CLS."})
    if "onclick=" in code_or_html.lower():
        issues.append({"severity": "MEDIUM", "check": "a11y-interactive", "fix": "Use <button> with keyboard handlers, not onclick divs."})
    return {"issue_count": len(issues), "status": "PASS" if not issues else "WARN", "issues": issues, "budgets": {"LCP_s": 1.5, "CLS": 0.1, "JS_kb": 170}}


@mcp.tool()
def web_deploy_checklist(target: str = "github-pages") -> Dict[str, Any]:
    """Target-aware deploy gate.

    Args:
        target: github-pages, vercel, cloudflare, local.
    """
    base = ["pnpm build passes clean", "404 + fallback route tested", "env vars in vault (no .env commit)", "lighthouse mobile >=90", "C13-WARDEN: no secrets in bundle (grep api_key)"]
    extra = {"vercel": ["vercel --prod preview OK", "edge regions pinned"], "github-pages": ["base path set", "gh-pages branch clean"], "cloudflare": ["wrangler pages publish OK"], "local": ["serve dist/ smoke test"]}.get(target.strip().lower(), [])
    return {"target": target, "checklist": base + extra, "gate": "No ship if secrets leak or LCP regression."}


if __name__ == "__main__":
    mcp.run()
