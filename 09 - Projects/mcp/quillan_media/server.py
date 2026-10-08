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
"""

import sys
from typing import Dict, Any, List

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from fastmcp import FastMCP

mcp = FastMCP("QuillanMedia")

QUALITY_TAGS = "masterpiece, best quality, ultra-detailed, 8k, sharp focus, cinematic lighting, raytraced subsurface scattering"
NEGATIVE_TAGS = "low quality, worst quality, blurry, bad anatomy, deformed, extra limbs, floating limbs, disconnected limbs, mutation, gross proportions, missing fingers, malformed hands, watermark, signature, username, amateur render, flat shading, washed out"


@mcp.tool()
def media_image_brief(objective: str, style: str = "stylized-painterly", composition: str = "rule-of-thirds wide", palette: str = "deep teal + burnished gold") -> Dict[str, Any]:
    """Build a strict image-generation brief (tool-ingestible JSON schema).

    Args:
        objective: What the image must communicate.
        style: Render style.
        composition: Framing / lens.
        palette: Quillan palette hint.
    """
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


@mcp.tool()
def media_video_brief(beats: str = "establish, escalate, payoff", duration_s: int = 8, camera: str = "slow dolly-in") -> Dict[str, Any]:
    """Build a short video/shot brief.

    Args:
        beats: Comma-separated story beats.
        duration_s: Total seconds.
        camera: Camera move.
    """
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


@mcp.tool()
def media_character_sheet(name: str, role: str = "protagonist", traits: str = "brave, scarred, teal cloak") -> Dict[str, Any]:
    """Create a reusable character bible for consistency across generations.

    Args:
        name: Character name.
        role: Story role.
        traits: Comma-separated visual traits.
    """
    trait_list = [t.strip() for t in traits.split(",") if t.strip()]
    return {
        "name": name,
        "role": role,
        "signature_traits": trait_list,
        "anchor_prompt": f"{name}, {role}, {', '.join(trait_list)}, consistent face, consistent costume, model sheet, front/side/three-quarter",
        "Negative": NEGATIVE_TAGS,
        "usage": "Paste anchor_prompt into every image brief Content[0] to lock identity.",
    }


@mcp.tool()
def media_prompt_enhance(raw_prompt: str) -> Dict[str, Any]:
    """Enhance a raw prompt with Quillan aesthetic boosters.

    Args:
        raw_prompt: User draft prompt.
    """
    enhanced = f"{raw_prompt.strip()}, {QUALITY_TAGS}"
    return {"original": raw_prompt, "enhanced": enhanced, "negative": NEGATIVE_TAGS, "cfg": 6.5, "steps": 35, "sampler": "DPM++ 2M Karras"}


if __name__ == "__main__":
    mcp.run()
