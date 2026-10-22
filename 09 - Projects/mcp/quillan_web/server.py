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
# [QUILLAN-v5.4.0-ONI-SHARED-WIRED]
try:
    from _shared import setup as _quillan_setup, tool_guard, ok_response, error_response, validate_string, validate_path, validate_int, read_file_safe
    from _shared.base_engines import PRISM_ENGINE, COUNCIL_ENGINE, GOVERNOR_ENGINE, SANDBOX_ENGINE
    _SHARED_CTX = _quillan_setup(server_name='quillan_web', mcp_app=mcp)
except Exception as _shared_err:
    import sys as _sys
    print(f'[quillan_web] _shared warning: {_shared_err}', file=_sys.stderr)


@mcp.tool()
def web_scaffold(project_name: str = "quillan-web", stack: str = "vite+ts", routes: str = "/, /dev, /lore") -> Dict[str, Any]:
    """Scaffold a web project with stack-aware route + file plan."""
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
        return {"target": "QuillanBuddy", "suggested_tool": "buddy_mirror_collaborate",
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


# ─── 7. CSS/HTML GENERATION ──────────────────────────────────────────────────

@mcp.tool()
def web_css_generate(component: str = "card", framework: str = "tailwind", responsive: bool = True) -> Dict[str, Any]:
    """Generate CSS/HTML for a UI component."""
    templates = {
        "card": {"html": '<div class="card"><h3>Title</h3><p>Content</p></div>',
                 "css": ".card { padding: 1rem; border-radius: 0.5rem; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }"},
        "navbar": {"html": '<nav class="navbar"><a href="/">Home</a><a href="/about">About</a></nav>',
                   "css": ".navbar { display: flex; gap: 1rem; padding: 1rem; }"},
        "hero": {"html": '<section class="hero"><h1>Title</h1><p>Subtitle</p><button>CTA</button></section>',
                 "css": ".hero { text-align: center; padding: 4rem 2rem; }"},
        "form": {"html": '<form class="form"><input type="text" placeholder="Name" /><button>Submit</button></form>',
                 "css": ".form { display: flex; flex-direction: column; gap: 1rem; max-width: 400px; }"},
        "grid": {"html": '<div class="grid"><div>Item 1</div><div>Item 2</div><div>Item 3</div></div>',
                 "css": ".grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; }"},
    }
    tmpl = templates.get(component.lower(), templates["card"])
    return {"component": component, "framework": framework, "responsive": responsive,
            "html": tmpl["html"], "css": tmpl["css"], "tailwind_classes": "p-4 rounded-lg shadow-md" if framework == "tailwind" else "",
            "council_note": "C22-AURELION: aesthetics; C15-LUMINARIS: clarity."}

# ─── 8. COMPONENT GENERATION ─────────────────────────────────────────────────

@mcp.tool()
def web_component_generate(name: str, framework: str = "react", props: str = "title,content", stateful: bool = False) -> Dict[str, Any]:
    """Generate a UI component (React/Vue/Svelte)."""
    props_list = [p.strip() for p in props.split(",") if p.strip()]
    if framework.lower() == "react":
        code = f"""import React from 'react';

interface {name}Props {{
  {chr(10).join(f'  {p}: string;' for p in props_list)}
}}

export const {name}: React.FC<{name}Props> = ({{ {', '.join(props_list)} }}) => {{
  return (
    <div className="{name.lower()}">
      <h2>{{{props_list[0] if props_list else 'title'}}}</h2>
      <p>{{{props_list[1] if len(props_list) > 1 else 'content'}}}</p>
    </div>
  );
}};
"""
    elif framework.lower() == "vue":
        code = f"""<template>
  <div class="{name.lower()}">
    <h2>{{{{ {props_list[0] if props_list else 'title'} }}}}</h2>
  </div>
</template>

<script setup lang="ts">
defineProps<{{
  {chr(10).join(f'  {p}: string;' for p in props_list)}
}}>()
</script>
"""
    else:
        code = f"// {name} component for {framework}\nexport default function {name}() {{ return <div>{name}</div>; }}"
    return {"name": name, "framework": framework, "props": props_list, "stateful": stateful, "code": code,
            "council_note": "C10-CODEWEAVER: clean code; C24-SCHEMA: component structure."}

# ─── 9. API ROUTE GENERATION ─────────────────────────────────────────────────

@mcp.tool()
def web_api_route(path: str = "/api/users", method: str = "GET", framework: str = "nextjs") -> Dict[str, Any]:
    """Generate an API route handler."""
    if framework.lower() == "nextjs":
        code = f"""// app{path}/route.ts
import {{ NextResponse }} from 'next/server';

export async function {method.upper()}(request: Request) {{
  try {{
    // TODO: implement {method.upper()} {path}
    return NextResponse.json({{ message: '{method.upper()} {path} OK' }});
  }} catch (error) {{
    return NextResponse.json({{ error: 'Internal Server Error' }}, {{ status: 500 }});
  }}
}}
"""
    elif framework.lower() == "express":
        code = f"""// routes{path}.js
const express = require('express');
const router = express.Router();

router.{method.lower()}('{path}', async (req, res) => {{
  try {{
    // TODO: implement {method.upper()} {path}
    res.json({{ message: '{method.upper()} {path} OK' }});
  }} catch (error) {{
    res.status(500).json({{ error: 'Internal Server Error' }});
  }}
}});

module.exports = router;
"""
    else:
        code = f"# {method.upper()} {path} handler for {framework}"
    return {"path": path, "method": method.upper(), "framework": framework, "code": code,
            "council_note": "C10-CODEWEAVER: API design; C13-WARDEN: security."}

# ─── 10. AUTHENTICATION SETUP ────────────────────────────────────────────────

@mcp.tool()
def web_auth_setup(provider: str = "nextauth", strategy: str = "jwt") -> Dict[str, Any]:
    """Generate authentication setup configuration."""
    configs = {
        "nextauth": {"package": "next-auth", "config": "auth.ts with providers (GitHub, Google, Credentials)",
                     "env_vars": ["NEXTAUTH_SECRET", "NEXTAUTH_URL", "GITHUB_ID", "GITHUB_SECRET"]},
        "clerk": {"package": "@clerk/nextjs", "config": "ClerkProvider wrapper + middleware",
                  "env_vars": ["NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY", "CLERK_SECRET_KEY"]},
        "supabase": {"package": "@supabase/supabase-js", "config": "createClient with auth helpers",
                     "env_vars": ["NEXT_PUBLIC_SUPABASE_URL", "NEXT_PUBLIC_SUPABASE_ANON_KEY"]},
        "firebase": {"package": "firebase/auth", "config": "initializeApp + getAuth",
                     "env_vars": ["FIREBASE_API_KEY", "FIREBASE_AUTH_DOMAIN"]},
    }
    config = configs.get(provider.lower(), configs["nextauth"])
    return {"provider": provider, "strategy": strategy, "config": config,
            "security": ["HTTPS only", "CSRF protection", "rate limiting", "secure cookies"],
            "council_note": "C13-WARDEN: auth security; C2-VIR: privacy."}

# ─── 11. STATE MANAGEMENT SETUP ───────────────────────────────────────────────

@mcp.tool()
def web_state_setup(framework: str = "zustand", scope: str = "global") -> Dict[str, Any]:
    """Generate state management setup."""
    configs = {
        "zustand": {"package": "zustand", "code": "import { create } from 'zustand';\nconst useStore = create((set) => ({ count: 0, increment: () => set((s) => ({ count: s.count + 1 })) }));"},
        "redux": {"package": "@reduxjs/toolkit", "code": "import { configureStore, createSlice } from '@reduxjs/toolkit';\nconst slice = createSlice({ name: 'app', initialState: {}, reducers: {} });"},
        "jotai": {"package": "jotai", "code": "import { atom } from 'jotai';\nconst countAtom = atom(0);"},
        "pinia": {"package": "pinia", "code": "import { defineStore } from 'pinia';\nexport const useStore = defineStore('app', { state: () => ({ count: 0 }) });"},
    }
    config = configs.get(framework.lower(), configs["zustand"])
    return {"framework": framework, "scope": scope, "config": config,
            "council_note": "C24-SCHEMA: state structure; C10-CODEWEAVER: clean code."}

# ─── 12. WEB SECURITY (CSP, CORS, etc.) ──────────────────────────────────────

@mcp.tool()
def web_security(config: str = "csp", target: str = "nextjs") -> Dict[str, Any]:
    """Generate web security configuration (CSP, CORS, headers)."""
    configs = {
        "csp": {"header": "Content-Security-Policy", "value": "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; img-src 'self' data: https:; font-src 'self' https:;"},
        "cors": {"header": "Access-Control-Allow-Origin", "value": "https://yourdomain.com", "methods": "GET, POST, PUT, DELETE, OPTIONS", "headers": "Content-Type, Authorization"},
        "headers": {"X-Frame-Options": "DENY", "X-Content-Type-Options": "nosniff", "X-XSS-Protection": "1; mode=block", "Referrer-Policy": "strict-origin-when-cross-origin", "Permissions-Policy": "camera=(), microphone=(), geolocation=()"},
        "rate_limit": {"strategy": "token bucket", "limit": "100 req/min per IP", "implementation": "middleware or edge function"},
    }
    cfg = configs.get(config.lower(), configs["headers"])
    return {"config_type": config, "target": target, "config": cfg,
            "council_note": "C13-WARDEN: security perimeter; C2-VIR: privacy."}

# ─── 13. SEO OPTIMIZATION ────────────────────────────────────────────────────

@mcp.tool()
def web_seo(pages: str = "/", keywords: str = "quillan ronin, ai, cognitive") -> Dict[str, Any]:
    """Generate SEO optimization checklist and meta tags."""
    page_list = [p.strip() for p in pages.split(",") if p.strip()]
    kw_list = [k.strip() for k in keywords.split(",") if k.strip()]
    return {
        "pages": page_list,
        "keywords": kw_list,
        "meta_tags": {"title": "Page Title | Site Name", "description": "Compelling description under 160 chars", "og:image": "1200x630 social preview", "twitter:card": "summary_large_image"},
        "checklist": ["unique title per page", "meta description", "canonical URL", "structured data (JSON-LD)", "sitemap.xml", "robots.txt", "alt text on images", "internal linking", "mobile-friendly", "Core Web Vitals green"],
        "council_note": "C15-LUMINARIS: clarity; C29-NAVIGATOR: SEO routing.",
    }

# ─── 14. RESPONSIVE DESIGN ───────────────────────────────────────────────────

@mcp.tool()
def web_responsive(breakpoints: str = "sm:640,md:768,lg:1024,xl:1280") -> Dict[str, Any]:
    """Generate responsive design configuration."""
    bp_list = [b.strip() for b in breakpoints.split(",") if b.strip()]
    bp_dict = {}
    for b in bp_list:
        parts = b.split(":")
        if len(parts) == 2:
            bp_dict[parts[0]] = int(parts[1])
    return {
        "breakpoints": bp_dict,
        "strategy": "mobile-first",
        "css": "@media (min-width: 768px) { /* tablet */ } @media (min-width: 1024px) { /* desktop */ }",
        "tailwind": "sm: md: lg: xl: 2xl: prefixes",
        "council_note": "C22-AURELION: responsive aesthetics; C15-LUMINARIS: clarity.",
    }

# ─── 15. PWA SETUP ──────────────────────────────────────────────────────────

@mcp.tool()
def web_pwa(name: str = "Quillan App", short_name: str = "Quillan", theme_color: str = "#000000") -> Dict[str, Any]:
    """Generate PWA (Progressive Web App) configuration."""
    manifest = {
        "name": name,
        "short_name": short_name,
        "start_url": "/",
        "display": "standalone",
        "theme_color": theme_color,
        "background_color": "#ffffff",
        "icons": [{"src": "/icon-192.png", "sizes": "192x192", "type": "image/png"}, {"src": "/icon-512.png", "sizes": "512x512", "type": "image/png"}],
    }
    return {"manifest": manifest, "service_worker": "cache-first strategy for static assets, network-first for API",
            "features": ["offline support", "push notifications", "install prompt", "background sync"],
            "council_note": "C29-NAVIGATOR: PWA routing; C26-TECHNE: offline constraints."}

# ─── 16. SSR/SSG CONFIGURATION ───────────────────────────────────────────────

@mcp.tool()
def web_ssr_ssg(mode: str = "ssg", framework: str = "nextjs") -> Dict[str, Any]:
    """Generate SSR/SSG configuration."""
    modes = {
        "ssr": {"description": "Server-Side Rendering — render on each request", "use_case": "dynamic content, personalized pages"},
        "ssg": {"description": "Static Site Generation — pre-render at build time", "use_case": "blogs, docs, marketing pages"},
        "isr": {"description": "Incremental Static Regeneration — revalidate on demand", "use_case": "e-commerce, news sites"},
        "csr": {"description": "Client-Side Rendering — render in browser", "use_case": "dashboards, SPAs"},
    }
    config = modes.get(mode.lower(), modes["ssg"])
    return {"mode": mode, "framework": framework, "config": config,
            "council_note": "C29-NAVIGATOR: rendering strategy; C14-KAIDO: performance."}

# ─── 17. WEBSOCKET SETUP ─────────────────────────────────────────────────────

@mcp.tool()
def web_websocket(endpoint: str = "/ws", framework: str = "nextjs") -> Dict[str, Any]:
    """Generate WebSocket setup configuration."""
    return {
        "endpoint": endpoint,
        "framework": framework,
        "protocol": "ws:// or wss://",
        "features": ["real-time updates", "bidirectional communication", "auto-reconnect", "heartbeat"],
        "use_cases": ["chat", "live updates", "collaborative editing", "notifications"],
        "council_note": "C30-TESSERACT: real-time; C31-NEXUS: connection management.",
    }

# ─── 18. GRAPHQL SETUP ───────────────────────────────────────────────────────

@mcp.tool()
def web_graphql(schema: str = "type Query { hello: String }", framework: str = "apollo") -> Dict[str, Any]:
    """Generate GraphQL setup configuration."""
    return {
        "schema": schema,
        "framework": framework,
        "features": ["queries", "mutations", "subscriptions", "fragments"],
        "tools": ["Apollo Server", "GraphQL Yoga", "Hasura", "Prisma"],
        "council_note": "C24-SCHEMA: schema design; C9-AETHER: query language.",
    }

# ─── 19. FORM VALIDATION ─────────────────────────────────────────────────────

@mcp.tool()
def web_form_validation(fields: str = "email,password,name", library: str = "zod") -> Dict[str, Any]:
    """Generate form validation schema."""
    field_list = [f.strip() for f in fields.split(",") if f.strip()]
    return {
        "fields": field_list,
        "library": library,
        "schema": {f: {"type": "string", "required": True, "validation": "email" if f == "email" else "min:8" if f == "password" else "min:1"} for f in field_list},
        "features": ["client-side validation", "server-side validation", "error messages", "async validation"],
        "council_note": "C7-LOGOS: validation; C2-VIR: input safety.",
    }

# ─── 20. I18N SETUP ──────────────────────────────────────────────────────────

@mcp.tool()
def web_i18n(languages: str = "en,es,fr,de,ja", framework: str = "next-intl") -> Dict[str, Any]:
    """Generate internationalization (i18n) setup."""
    lang_list = [l.strip() for l in languages.split(",") if l.strip()]
    return {
        "languages": lang_list,
        "framework": framework,
        "features": ["locale routing", "message extraction", "pluralization", "date/number formatting", "RTL support"],
        "file_structure": "messages/{locale}.json",
        "council_note": "C16-VOXUM: language; C9-AETHER: semantics.",
    }

# ─── 21. WEB PERFORMANCE OPTIMIZATION ─────────────────────────────────────────

@mcp.tool()
def web_perf_optimize(target: str = "lcp", budget: float = 1.5) -> Dict[str, Any]:
    """Generate web performance optimization recommendations."""
    optimizations = {
        "lcp": ["preload critical resources", "optimize images (WebP/AVIF)", "reduce server response time", "eliminate render-blocking resources"],
        "cls": ["set explicit dimensions on images/ads", "avoid inserting content above existing", "use CSS transforms for animations"],
        "fid": ["break up long tasks", "use web workers", "optimize JavaScript execution", "defer non-critical JS"],
        "ttfb": ["enable caching (CDN)", "optimize database queries", "use edge functions", "compress responses"],
    }
    return {
        "target": target,
        "budget": budget,
        "optimizations": optimizations.get(target.lower(), optimizations["lcp"]),
        "tools": ["Lighthouse", "WebPageTest", "Chrome DevTools", "Core Web Vitals"],
        "council_note": "C14-KAIDO: performance; C15-LUMINARIS: clarity.",
    }

# ─── 22. WEB TESTING ─────────────────────────────────────────────────────────

@mcp.tool()
def web_test_setup(framework: str = "playwright", test_type: str = "e2e") -> Dict[str, Any]:
    """Generate web testing setup configuration."""
    configs = {
        "playwright": {"package": "@playwright/test", "config": "playwright.config.ts", "features": ["cross-browser", "mobile emulation", "visual regression", "API testing"]},
        "cypress": {"package": "cypress", "config": "cypress.config.ts", "features": ["E2E testing", "component testing", "time travel debugging"]},
        "vitest": {"package": "vitest", "config": "vitest.config.ts", "features": ["unit testing", "fast HMR", "coverage"]},
        "jest": {"package": "jest", "config": "jest.config.js", "features": ["unit testing", "snapshot testing", "mocking"]},
    }
    config = configs.get(framework.lower(), configs["playwright"])
    return {"framework": framework, "test_type": test_type, "config": config,
            "council_note": "C25-PROMETHEUS: testing; C7-LOGOS: verification."}

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
