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
# [QUILLAN-v5.4.0-ONI-SHARED-WIRED]
try:
    from _shared import setup as _quillan_setup, tool_guard, ok_response, error_response, validate_string, validate_path, validate_int, read_file_safe
    from _shared.base_engines import PRISM_ENGINE, COUNCIL_ENGINE, GOVERNOR_ENGINE, SANDBOX_ENGINE
    _SHARED_CTX = _quillan_setup(server_name='quillan_media', mcp_app=mcp)
except Exception as _shared_err:
    import sys as _sys
    print(f'[quillan_media] _shared warning: {_shared_err}', file=_sys.stderr)

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
        return {"target": "QuillanBuddy", "suggested_tool": "buddy_mirror_collaborate",
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


# ─── 10. IMAGE-TO-IMAGE TRANSFORMATION ───────────────────────────────────────

@mcp.tool()
def media_img2img(source_description: str, transform: str = "style_transfer", strength: float = 0.75, style: str = "oil painting") -> Dict[str, Any]:
    """Generate an image-to-image transformation brief."""
    return {
        "source": source_description,
        "transform": transform,
        "strength": max(0.0, min(1.0, strength)),
        "style": style,
        "params": {"denoising_strength": strength, "cfg_scale": 6.5, "steps": 35, "sampler": "DPM++ 2M Karras"},
        "negative": NEGATIVE_TAGS,
        "council_note": "C22-AURELION: aesthetic gate >=7.0; C1-ASTRA: structural preservation.",
    }

# ─── 11. STYLE TRANSFER ──────────────────────────────────────────────────────

@mcp.tool()
def media_style_transfer(content: str, style_reference: str = "van gogh starry night", intensity: float = 0.8) -> Dict[str, Any]:
    """Generate a style transfer brief from content + style reference."""
    return {
        "content": content,
        "style_reference": style_reference,
        "intensity": max(0.0, min(1.0, intensity)),
        "method": "neural style transfer (AdaIN) or diffusion img2img",
        "params": {"content_weight": 1.0, "style_weight": intensity * 10, "iterations": 500},
        "council_note": "C22-AURELION: style coherence; C8-METASYNTH: cross-modal fusion.",
    }

# ─── 12. UPSCALING ──────────────────────────────────────────────────────────

@mcp.tool()
def media_upscale(source_description: str, scale_factor: int = 4, method: str = "esrgan") -> Dict[str, Any]:
    """Generate an upscaling brief for an image."""
    methods = {"esrgan": "Real-ESRGAN (x4)", "swinir": "SwinIR (x4)", "stable_diffusion": "SD Upscale (x2-x4)", "lanczos": "Lanczos (fast, lower quality)"}
    return {
        "source": source_description,
        "scale_factor": scale_factor,
        "method": methods.get(method, methods["esrgan"]),
        "output_resolution": f"{scale_factor}x original",
        "params": {"denoise_strength": 0.35, "tile_size": 512},
        "council_note": "C22-AURELION: sharpness gate; C14-KAIDO: efficiency.",
    }

# ─── 13. INPAINTING / OUTPAINTING ───────────────────────────────────────────

@mcp.tool()
def media_inpaint(source_description: str, mask_description: str, mode: str = "inpaint", prompt: str = "") -> Dict[str, Any]:
    """Generate an inpainting/outpainting brief."""
    return {
        "source": source_description,
        "mask": mask_description,
        "mode": mode,
        "prompt": prompt or f"Fill {mask_description} seamlessly",
        "params": {"denoising_strength": 0.75 if mode == "inpaint" else 0.5, "cfg_scale": 7.0, "steps": 30},
        "negative": NEGATIVE_TAGS + ", seam, blur, artifact",
        "council_note": "C1-ASTRA: structural coherence; C22-AURELION: seamless blend.",
    }

# ─── 14. AUDIO-TO-VIDEO SYNC ─────────────────────────────────────────────────

@mcp.tool()
def media_audio_sync(audio_description: str, video_description: str, sync_mode: str = "beat_match") -> Dict[str, Any]:
    """Generate an audio-to-video synchronization brief."""
    return {
        "audio": audio_description,
        "video": video_description,
        "sync_mode": sync_mode,
        "methods": {"beat_match": "Cut on beat markers", "lip_sync": "Phoneme-based lip sync", "mood_sync": "Mood-driven color grading"},
        "output": {"format": "mp4", "codec": "h264", "audio_codec": "aac", "fps": 24},
        "council_note": "C23-CADENCE: rhythm sync; C27-CHRONICLE: narrative timing.",
    }

# ─── 15. 3D MODEL GENERATION ─────────────────────────────────────────────────

@mcp.tool()
def media_3d_generate(description: str, format: str = "glb", poly_count: str = "medium", style: str = "realistic") -> Dict[str, Any]:
    """Generate a 3D model generation brief."""
    poly_ranges = {"low": "1k-5k", "medium": "5k-20k", "high": "20k-100k", "ultra": "100k+"}
    return {
        "description": description,
        "format": format,
        "poly_count": poly_ranges.get(poly_count, poly_ranges["medium"]),
        "style": style,
        "pipeline": ["mesh generation", "UV unwrapping", "texture baking", "rigging (optional)", "export"],
        "params": {"method": "text-to-3D (Shap-E / DreamGaussian)", "texture_resolution": 1024},
        "council_note": "C26-TECHNE: geometry constraints; C1-ASTRA: shape coherence.",
    }

# ─── 16. TEXTURE GENERATION ──────────────────────────────────────────────────

@mcp.tool()
def media_texture_generate(material: str = "wood", resolution: int = 1024, seamless: bool = True, pbr: bool = True) -> Dict[str, Any]:
    """Generate a PBR texture set brief."""
    maps = ["albedo", "normal", "roughness", "metallic", "ao", "height"] if pbr else ["albedo"]
    return {
        "material": material,
        "resolution": resolution,
        "seamless": seamless,
        "pbr_maps": maps,
        "format": "PNG (lossless) or KTX2 (compressed)",
        "params": {"tiling": "seamless" if seamless else "single", "color_space": "sRGB for albedo, linear for others"},
        "council_note": "C22-AURELION: texture quality; C14-KAIDO: memory optimization.",
    }

# ─── 17. ANIMATION KEYFRAME PLANNING ─────────────────────────────────────────

@mcp.tool()
def media_animation_keyframes(action: str = "walk cycle", fps: int = 24, duration_s: float = 1.0, style: str = "realistic") -> Dict[str, Any]:
    """Generate an animation keyframe planning brief."""
    frame_count = int(fps * duration_s)
    return {
        "action": action,
        "fps": fps,
        "duration_s": duration_s,
        "frame_count": frame_count,
        "keyframes": [
            {"frame": 0, "pose": "start", "note": "contact pose"},
            {"frame": frame_count // 4, "pose": "passing", "note": "mid-stride"},
            {"frame": frame_count // 2, "pose": "contact", "note": "opposite foot"},
            {"frame": 3 * frame_count // 4, "pose": "passing", "note": "mid-stride"},
            {"frame": frame_count, "pose": "start", "note": "loop close"},
        ],
        "principles": ["squash & stretch", "anticipation", "follow-through", "timing", "exaggeration"],
        "council_note": "C23-CADENCE: motion timing; C1-ASTRA: pose coherence.",
    }

# ─── 18. MULTI-ANGLE CONSISTENCY ─────────────────────────────────────────────

@mcp.tool()
def media_multi_angle(subject: str, angles: str = "front,side,back,three-quarter", style: str = "consistent") -> Dict[str, Any]:
    """Generate a multi-angle consistency brief for a subject."""
    angle_list = [a.strip() for a in angles.split(",") if a.strip()]
    return {
        "subject": subject,
        "angles": angle_list,
        "style": style,
        "anchor_prompt": f"{subject}, consistent identity across all angles, model sheet, {', '.join(angle_list)} views",
        "negative": NEGATIVE_TAGS + ", inconsistent features, different face",
        "rule": "Use same seed + anchor_prompt for all angles. Vary only camera angle.",
        "council_note": "C1-ASTRA: structural consistency; C22-AURELION: aesthetic coherence.",
    }

# ─── 19. BATCH GENERATION ────────────────────────────────────────────────────

@mcp.tool()
def media_batch_generate(prompts: str, count: int = 4, seed: int = -1, variation: str = "seed") -> Dict[str, Any]:
    """Generate a batch generation brief for multiple images."""
    prompt_list = [p.strip() for p in prompts.split("||") if p.strip()]
    return {
        "prompts": prompt_list,
        "count": count,
        "seed": seed,
        "variation": variation,
        "total_images": len(prompt_list) * count,
        "params": {"cfg_scale": 6.5, "steps": 35, "sampler": "DPM++ 2M Karras"},
        "naming": "batch_{index:04d}_{prompt_slug}.png",
        "council_note": "C14-KAIDO: batch efficiency; C22-AURELION: quality gate.",
    }

# ─── 20. MEDIA FORMAT CONVERSION ─────────────────────────────────────────────

@mcp.tool()
def media_convert(source_format: str = "png", target_format: str = "webp", quality: int = 90) -> Dict[str, Any]:
    """Generate a media format conversion brief."""
    conversions = {
        ("png", "webp"): "Lossless-ish, smaller file size",
        ("png", "jpg"): "Lossy, smaller file size",
        ("mp4", "webm"): "VP9/AV1 codec, smaller size",
        ("wav", "mp3"): "Lossy audio compression",
        ("glb", "gltf"): "Split binary + JSON",
    }
    return {
        "source_format": source_format,
        "target_format": target_format,
        "quality": quality,
        "description": conversions.get((source_format.lower(), target_format.lower()), "Standard conversion"),
        "tools": ["ffmpeg (video/audio)", "ImageMagick (images)", "Blender (3D)"],
        "council_note": "C14-KAIDO: format optimization; C26-TECHNE: compatibility.",
    }

# ─── 21. DEPTH MAP / NORMAL MAP GENERATION ───────────────────────────────────

@mcp.tool()
def media_depth_map(source_description: str, method: str = "midas") -> Dict[str, Any]:
    """Generate a depth map / normal map brief for 3D reconstruction."""
    return {
        "source": source_description,
        "method": method,
        "outputs": ["depth_map (16-bit)", "normal_map (RGB)", "point_cloud (optional)"],
        "params": {"model": "MiDaS v3.1" if method == "midas" else method, "resolution": "match source"},
        "use_cases": ["3D reconstruction", "parallax effect", "relighting", "depth-based blur"],
        "council_note": "C1-ASTRA: depth coherence; C26-TECHNE: 3D pipeline.",
    }

# ─── 22. VIDEO FRAME EXTRACTION ──────────────────────────────────────────────

@mcp.tool()
def media_frame_extract(video_description: str, fps: int = 24, format: str = "png") -> Dict[str, Any]:
    """Generate a video frame extraction brief."""
    return {
        "video": video_description,
        "fps": fps,
        "format": format,
        "naming": "frame_{index:06d}.png",
        "use_cases": ["training data", "sprite extraction", "storyboard", "thumbnail selection"],
        "tool": "ffmpeg -i input.mp4 -vf fps=24 frame_%06d.png",
        "council_note": "C27-CHRONICLE: frame sequencing; C14-KAIDO: batch efficiency.",
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
