#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🎨 QUILLAN-RONIN MEDIA ENGINE — NATIVE FAST-MCP SERVER
=======================================================
Embodying C22-AURELION (Aesthetics), C15-LUMINARIS (Clarity),
C1-ASTRA (Vision), C8-METASYNTH (Fusion):
  1. media_image_brief — objective -> strict JSON-schema image brief
  2. media_video_brief — beats -> camera + duration brief
  3. media_character_sheet — consistent character bible
  4. media_prompt_enhance — raw prompt -> boosted + negative tags
  5. media_full_pipeline — image brief -> video brief -> character sheet -> prompt enhance (persisted)
  6. media_route — cross-MCP routing payload (council/code/buddy)
  7. media_health — media project dashboard summary
  8. media_session_list / media_session_replay — persistence
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

mcp = FastMCP("QuillanMedia")
mcp.add_middleware(QuillanErrorMiddleware())
@mcp.tool()
def media_image_brief(objective: str, style: str = "cinematic", composition: str = "rule of thirds", palette: str = "warm") -> Dict[str, Any]:


    """Build a strict image-generation brief (tool-ingestible JSON schema)."""
    return {
        "Objective": objective,
        "Brief": f"{style}, {composition}, palette {palette}. Single hero subject, readable silhouette at thumbnail.",
        "Content": [objective, f"style:{style}", f"composition:{composition}", f"palette:{palette}"],
        "Style": style,
        "Camera": {"lens": "35mm", "angle": "eye-level", "depth": "shallow bokeh bg, sharp hero"},
        "Quality": QUALITY_TAGS,
        "Negative": NEGATIVE_TAGS,
        "gate": "C22-AURELION >=7.0 aesthetic or refine (+10 steps, -0.5 CFG).",
    }

# ─── 2. VIDEO BRIEF ─────────────────────────────────────────────────────────

@mcp.tool()
def media_video_brief(beats: str = "establish, escalate, payoff", duration_s: int = 8, camera: str = "slow dolly-in") -> Dict[str, Any]:
    """Build a short video/shot brief."""
    beat_list = [b.strip() for b in beats.split(",") if b.strip()]
    per = round(duration_s / max(1, len(beat_list)), 1) if beat_list else duration_s
    return {
        "Objective": f"{duration_s}s shot: {' -> '.join(beat_list)}",
        "Brief": f"{camera}, {per}s per beat, motivated motion only.",
        "Content": beat_list,
        "Style": "cinematic, coherent lighting across cuts",
        "Camera": {"move": camera, "fps": 24, "shutter": "180deg"},
        "Negative": NEGATIVE_TAGS + ", jitter, morphing, extra limbs",
        "gate": "C23-CADENCE drift <=0.35 or re-sequence.",
    }

# ─── 3. CHARACTER SHEET ─────────────────────────────────────────────────────

@mcp.tool()
def media_character_sheet(name: str, role: str = "protagonist", traits: str = "brave, scarred, teal cloak") -> Dict[str, Any]:
    """Create a reusable character bible for consistency across generations."""
    trait_list = [t.strip() for t in traits.split(",") if t.strip()]
    return {
        "name": name,
        "role": role,
        "signature_traits": trait_list,
        "anchor_prompt": f"{name}, {role}, {', '.join(trait_list)}, consistent face, consistent costume, model sheet, front/side/three-quarter",
        "Negative": NEGATIVE_TAGS,
        "usage": "Paste anchor_prompt into every image brief Content[0] to lock identity.",
    }

# ─── 4. PROMPT ENHANCE ──────────────────────────────────────────────────────

@mcp.tool()
def media_prompt_enhance(raw_prompt: str) -> Dict[str, Any]:
    """Enhance a raw prompt with Quillan aesthetic boosters."""
    enhanced = f"{raw_prompt.strip()}, {QUALITY_TAGS}"
    return {"original": raw_prompt, "enhanced": enhanced, "negative": NEGATIVE_TAGS, "cfg": 6.5, "steps": 35, "sampler": "DPM++ 2M Karras"}

# ─── 5. FULL PIPELINE (persisted) ───────────────────────────────────────────

@mcp.tool()
def media_full_pipeline(objective: str, style: str = "stylized-painterly", composition: str = "rule-of-thirds wide",
                       palette: str = "deep teal + burnished gold", beats: str = "establish, escalate, payoff",
                       duration_s: int = 8, camera: str = "slow dolly-in",
                       name: str = "Quillan Hero", role: str = "protagonist", traits: str = "brave, scarred, teal cloak",
                       raw_prompt: str = "", persist: bool = True) -> Dict[str, Any]:
    """Chain image brief -> video brief -> character sheet -> prompt enhance."""
    image = media_image_brief(objective=objective, style=style, composition=composition, palette=palette)
    video = media_video_brief(beats=beats, duration_s=duration_s, camera=camera)
    character = media_character_sheet(name=name, role=role, traits=traits)
    enhance = media_prompt_enhance(raw_prompt=raw_prompt or objective)

    session_id = f"media-{int(time.time())}"
    result = {
        "status": "MEDIA_PIPELINE_COMPLETE",
        "session_id": session_id,
        "image_brief": image,
        "video_brief": video,
        "character_sheet": character,
        "prompt_enhance": enhance,
        "overall": "READY",
        "timestamp": time.time(),
    }

    if persist:
        try:
            p = SESSION_DIR / f"{session_id}.json"
            p.write_text(json.dumps({"type": "media_pipeline", "result": result}, indent=2), encoding="utf-8")
            result["persisted_to"] = str(p)
        except Exception as e:
            result["persist_error"] = str(e)

    return result

# ─── 6. ROUTE (cross-MCP) ───────────────────────────────────────────────────

@mcp.tool()
def media_route(target_mcp: str = "code", payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Cross-MCP routing payload for council/code/buddy integration."""
    if target_mcp.lower() == "council":
        return {"target": "QuillanCouncil", "suggested_tool": "council_full_deliberation",
                "payload": payload or {}, "note": "Route media design objective to council deliberation."}
    if target_mcp.lower() == "code":
        return {"target": "QuillanCode", "suggested_tool": "codeweaver_full_pipeline",
                "payload": payload or {}, "note": "Route media asset implementation to code scan pipeline."}
    if target_mcp.lower() == "buddy":
        return {"target": "RoninBuddy", "suggested_tool": "buddy_mirror_collaborate",
                "payload": payload or {}, "note": "Route media creative brief to mirror-collaboration."}
    return {"error": f"Unknown target '{target_mcp}'. Use council, code, or buddy."}

# ─── 8. HEALTH DASHBOARD ────────────────────────────────────────────────────

@mcp.tool()
def media_health() -> Dict[str, Any]:
    """Media project dashboard summary."""
    sessions = []
    try:
        for p in sorted(SESSION_DIR.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True):
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                sessions.append({
                    "session_id": data.get("session_id"),
                    "objective": (data.get("image_brief") or {}).get("Objective"),
                    "overall": data.get("overall"),
                    "timestamp": data.get("timestamp"),
                })
            except Exception:
                continue
    except Exception:
        pass
    return {
        "status": "MEDIA_HEALTH_READY",
        "sessions": sessions[:5],
        "council_note": "C22-AURELION aesthetics; C15-LUMINARIS clarity; C1-ASTRA vision.",
    }

# ─── 9. SESSION PERSISTENCE ─────────────────────────────────────────────────

@mcp.tool()
def media_session_list() -> Dict[str, Any]:
    """List persisted media sessions."""
    sessions = []
    for p in sorted(SESSION_DIR.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            sessions.append({
                "session_id": data.get("session_id"),
                "objective": (data.get("image_brief") or {}).get("Objective"),
                "overall": data.get("overall"),
                "timestamp": data.get("timestamp"),
                "file": str(p),
            })
        except Exception:
            continue
    return {"count": len(sessions), "sessions": sessions}

@mcp.tool()
def media_session_replay(session_id: str) -> Dict[str, Any]:
    """Replay a persisted media session by ID."""
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
