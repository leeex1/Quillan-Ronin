#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🎵 QUILLAN-RONIN AUDIO ENGINE — NATIVE FAST-MCP SERVER
=======================================================
Embodying C23-CADENCE (Rhythm/Audio), C8-METASYNTH (Fusion),
C22-AURELION (Aesthetics), C27-CHRONICLE (Narrative timing):
  1. audio_design_cue — mood -> stems, BPM, key, texture brief
  2. audio_analyze_track — lightweight Synesthesia-style analysis (librosa/whisper optional)
  3. audio_mix_plan — stems -> LUFS / headroom / bus plan
  4. audio_generate_sfx_brief — action+material -> layered SFX recipe
  5. audio_director_note — adaptive music state machine for games
  6. audio_full_pipeline — design cue -> mix plan -> director note (persisted)
  7. audio_route — cross-MCP routing payload (council/code/buddy)
  8. audio_health — audio project dashboard summary
  9. audio_session_list / audio_session_replay — persistence
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

mcp = FastMCP("QuillanAudio")
# [QUILLAN-v5.4.0-ONI-SHARED-WIRED]
try:
    from _shared import setup as _quillan_setup, tool_guard, ok_response, error_response, validate_string, validate_path, validate_int, read_file_safe
    from _shared.base_engines import PRISM_ENGINE, COUNCIL_ENGINE, GOVERNOR_ENGINE, SANDBOX_ENGINE
    _SHARED_CTX = _quillan_setup(server_name='quillan_audio', mcp_app=mcp)
except Exception as _shared_err:
    import sys as _sys
    print(f'[quillan_audio] _shared warning: {_shared_err}', file=_sys.stderr)


# ─── 1. DESIGN CUE ──────────────────────────────────────────────────────────

@mcp.tool()
def audio_design_cue(mood: str = "tense", scene: str = "boss gate", duration_sec: int = 60, stems: str = "pulse,texture,lead") -> Dict[str, Any]:
    """Design a music cue brief from mood + scene."""
    base = MOOD_MAP.get(mood.strip().lower(), {"bpm": 100, "key": "C major", "texture": "hybrid pulse + air"})
    stem_list = [s.strip() for s in stems.split(",") if s.strip()]
    return {
        "cue": f"{mood} / {scene}",
        "mood": mood,
        "scene": scene,
        "duration_sec": duration_sec,
        "bpm": base["bpm"],
        "key": base["key"],
        "texture": base["texture"],
        "stems": stem_list,
        "structure": ["0-8s establish pulse", "8-32s add tension layer", "32-48s peak + earcon", "48s-end loop tail with reverb wash"],
        "deliverables": [f"{s}.wav 48k/24b" for s in stem_list] + ["mix_ref.mp3", "loop_points.txt"],
        "council_note": "C23-CADENCE: keep low-end mono <120Hz; C22-AURELION: leave 2dB air for SFX.",
    }

# ─── 2. ANALYZE TRACK ───────────────────────────────────────────────────────

@mcp.tool()
def audio_analyze_track(source: str) -> Dict[str, Any]:
    """Lightweight Synesthesia-style analysis. Uses librosa/whisper if installed, else heuristic fallback."""
    is_url = source.lower().startswith(("http://", "https://"))
    result: Dict[str, Any] = {"source": source, "is_url": is_url}
    try:
        import librosa
        p = Path(source)
        if p.exists() and p.is_file():
            import numpy as np
            y, sr = librosa.load(str(p), sr=None, mono=True)
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            cent = float(librosa.feature.spectral_centroid(y=y, sr=sr).mean())
            result.update({"bpm": round(float(tempo), 1), "brightness_centroid": round(cent, 1), "engine": "librosa"})
            if cent < 1500:
                result["texture"] = "Heavy, bass-dominant, dark"
            elif cent <= 2500:
                result["texture"] = "Mid-range focused, balanced"
            else:
                result["texture"] = "Bright, treble-dominant"
            try:
                import whisper
                model = whisper.load_model("tiny")
                tr = model.transcribe(str(p))
                segs = [{"start": round(s.get("start", 0.0), 2), "text": s.get("text", "").strip()[:120]} for s in tr.get("segments", [])[:10]]
                result["transcript_preview"] = segs
            except Exception as e:
                result["transcript_note"] = f"whisper skipped: {e}"
            return result
    except Exception:
        pass
    result.update({
        "engine": "heuristic-fallback",
        "bpm": 100,
        "texture": "Unknown without local file — render stems and re-run locally for true BPM/centroid.",
        "action": "Download via yt-dlp to WAV, then re-run this tool on the local path for full analysis.",
    })
    return result

# ─── 3. MIX PLAN ────────────────────────────────────────────────────────────

@mcp.tool()
def audio_mix_plan(stems: str = "drums,bass,pads,sfx", target_lufs: float = -14.0) -> Dict[str, Any]:
    """Build a mix bus plan with headroom targets."""
    stem_list = [s.strip() for s in stems.split(",") if s.strip()]
    return {
        "target_integrated_lufs": target_lufs,
        "true_peak_ceiling_db": -1.0,
        "bus": ["DRY -> EQ hp20 -> COMP 2:1 slow -> SAT tape 5% -> LIMIT", "SFX sidechain -3dB duck on beat", "Reverb send 12%, delay 1/8 dotted 8%"],
        "per_stem_db": {s: (-8.0 if s.lower() in ("drums", "bass") else -12.0) for s in stem_list},
        "qc": ["mono-check low end", "phone-speaker check", "loop-click check at boundary"],
    }

# ─── 4. SFX BRIEF ───────────────────────────────────────────────────────────

@mcp.tool()
def audio_generate_sfx_brief(action: str = "dash", material: str = "cloth+wind", perspective: str = "third-person close") -> Dict[str, Any]:
    """Generate a layered SFX recipe."""
    return {
        "sfx": f"{action} / {material} / {perspective}",
        "layers": [f"L1 transient: {material} snap (5-15ms)", "L2 body: pitched whoosh down 300Hz", "L3 tail: room verb 0.6s"],
        "variation": "5 round-robins, pitch +-4%, no repetition within 3 plays",
        "implementation": "Wwise/FMOD event with RTPC speed->pitch, distance->lowpass",
    }

# ─── 5. DIRECTOR NOTE ───────────────────────────────────────────────────────

@mcp.tool()
def audio_director_note(states: str = "explore,combat,boss,resolve") -> Dict[str, Any]:
    """Design an adaptive music state machine for games."""
    sl = [s.strip() for s in states.split(",") if s.strip()]
    transitions = [{"from": sl[i], "to": sl[(i + 1) % len(sl)], "rule": "4-bar crossfade + stinger", "cooldown_s": 6} for i in range(len(sl))] if sl else []
    return {"states": sl, "transitions": transitions, "rule": "Never hard-cut; always carry pulse stem across states."}

# ─── 6. FULL PIPELINE (persisted) ───────────────────────────────────────────

@mcp.tool()
def audio_full_pipeline(mood: str = "tense", scene: str = "boss gate", stems: str = "pulse,texture,lead", target_lufs: float = -14.0, states: str = "explore,combat,boss,resolve", persist: bool = True) -> Dict[str, Any]:
    """Chain design_cue -> mix_plan -> director_note into one audio pipeline."""
    cue = audio_design_cue(mood=mood, scene=scene, stems=stems)
    mix = audio_mix_plan(stems=stems, target_lufs=target_lufs)
    director = audio_director_note(states=states)

    session_id = f"audio-{int(time.time())}"
    result = {
        "status": "AUDIO_PIPELINE_COMPLETE",
        "session_id": session_id,
        "cue": cue,
        "mix_plan": mix,
        "director_note": director,
        "overall": "READY",
        "timestamp": time.time(),
    }

    if persist:
        try:
            p = SESSION_DIR / f"{session_id}.json"
            p.write_text(json.dumps({"type": "audio_pipeline", "result": result}, indent=2), encoding="utf-8")
            result["persisted_to"] = str(p)
        except Exception as e:
            result["persist_error"] = str(e)

    return result

# ─── 7. ROUTE (cross-MCP) ───────────────────────────────────────────────────

@mcp.tool()
def audio_route(target_mcp: str = "code", payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Cross-MCP routing payload for council/code/buddy integration."""
    if target_mcp.lower() == "council":
        return {"target": "QuillanCouncil", "suggested_tool": "council_full_deliberation",
                "payload": payload or {}, "note": "Route audio design objective to council deliberation."}
    if target_mcp.lower() == "code":
        return {"target": "QuillanCode", "suggested_tool": "codeweaver_full_pipeline",
                "payload": payload or {}, "note": "Route audio implementation to code scan pipeline."}
    if target_mcp.lower() == "buddy":
        return {"target": "RoninBuddy", "suggested_tool": "buddy_mirror_collaborate",
                "payload": payload or {}, "note": "Route audio cue to mirror-collaboration."}
    return {"error": f"Unknown target '{target_mcp}'. Use council, code, or buddy."}

# ─── 8. HEALTH DASHBOARD ────────────────────────────────────────────────────

@mcp.tool()
def audio_health() -> Dict[str, Any]:
    """Audio project dashboard summary (deps + state)."""
    deps = {}
    try:
        import librosa
        deps["librosa"] = "installed"
    except Exception:
        deps["librosa"] = "missing"
    try:
        import whisper
        deps["whisper"] = "installed"
    except Exception:
        deps["whisper"] = "missing"
    sessions = []
    try:
        for p in sorted(SESSION_DIR.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True):
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                sessions.append({"session_id": data.get("session_id"), "cue": (data.get("cue") or {}).get("cue"), "timestamp": data.get("timestamp")})
            except Exception:
                continue
    except Exception:
        pass
    return {
        "status": "AUDIO_HEALTH_READY",
        "deps": deps,
        "sessions": sessions[:5],
        "council_note": "C23-CADENCE keeps rhythm; C22-AURELION keeps aesthetics.",
    }

# ─── 9. SESSION PERSISTENCE ─────────────────────────────────────────────────

@mcp.tool()
def audio_session_list() -> Dict[str, Any]:
    """List persisted audio sessions."""
    sessions = []
    for p in sorted(SESSION_DIR.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            sessions.append({
                "session_id": data.get("session_id"),
                "cue": (data.get("cue") or {}).get("cue"),
                "timestamp": data.get("timestamp"),
                "file": str(p),
            })
        except Exception:
            continue
    return {"count": len(sessions), "sessions": sessions}

@mcp.tool()
def audio_session_replay(session_id: str) -> Dict[str, Any]:
    """Replay a persisted audio session by ID."""
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
