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
"""

import sys
from pathlib import Path
from typing import Dict, Any, List, Optional

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from fastmcp import FastMCP

mcp = FastMCP("QuillanAudio")

MOOD_MAP = {
    "tense": {"bpm": 132, "key": "D minor", "texture": "low pulse + high shimmer, sidechained noise"},
    "cozy": {"bpm": 76, "key": "F major pentatonic", "texture": "felt piano + tape wow, soft room"},
    "epic": {"bpm": 120, "key": "C minor -> Eb major lift", "texture": "taikos + ostinato strings + braam"},
    "mysterious": {"bpm": 90, "key": "A dorian", "texture": "bowed glass + sub drops, long tails"},
    "combat": {"bpm": 148, "key": "E phrygian", "texture": "double-kick + distorted bass, stingers on hit"},
}


@mcp.tool()
def audio_design_cue(mood: str = "tense", scene: str = "boss gate", duration_sec: int = 60, stems: str = "pulse,texture,lead") -> Dict[str, Any]:
    """Design a music cue brief from mood + scene.

    Args:
        mood: tense, cozy, epic, mysterious, combat (or free text).
        scene: Where it plays.
        duration_sec: Loop length.
        stems: Comma-separated stem list.
    """
    base = MOOD_MAP.get(mood.strip().lower(), {"bpm": 100, "key": "C major", "texture": "hybrid pulse + air"})
    stem_list = [s.strip() for s in stems.split(",") if s.strip()]
    return {
        "cue": f"{mood} / {scene}",
        "bpm": base["bpm"],
        "key": base["key"],
        "texture": base["texture"],
        "stems": stem_list,
        "structure": ["0-8s establish pulse", "8-32s add tension layer", "32-48s peak + earcon", "48s-end loop tail with reverb wash"],
        "deliverables": [f"{s}.wav 48k/24b" for s in stem_list] + ["mix_ref.mp3", "loop_points.txt"],
        "council_note": "C23-CADENCE: keep low-end mono <120Hz; C22-AURELION: leave 2dB air for SFX.",
    }


@mcp.tool()
def audio_analyze_track(source: str) -> Dict[str, Any]:
    """Lightweight Synesthesia-style analysis. Uses librosa/whisper if installed, else heuristic fallback.

    Args:
        source: YouTube URL or local audio path.
    """
    is_url = source.lower().startswith(("http://", "https://"))
    result: Dict[str, Any] = {"source": source, "is_url": is_url}
    # Optional heavy path
    try:
        import librosa  # type: ignore
        p = Path(source)
        if p.exists() and p.is_file():
            import numpy as np  # type: ignore
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
                import whisper  # type: ignore
                model = whisper.load_model("tiny")
                tr = model.transcribe(str(p))
                segs = [{"start": round(s.get("start", 0.0), 2), "text": s.get("text", "").strip()[:120]} for s in tr.get("segments", [])[:10]]
                result["transcript_preview"] = segs
            except Exception as e:
                result["transcript_note"] = f"whisper skipped: {e}"
            return result
    except Exception:
        pass
    # Fallback (no deps / URL mode): deterministic guidance so tool never hard-fails
    result.update({
        "engine": "heuristic-fallback",
        "bpm": 100,
        "texture": "Unknown without local file — render stems and re-run locally for true BPM/centroid.",
        "action": "Download via yt-dlp to WAV, then re-run this tool on the local path for full analysis.",
    })
    return result


@mcp.tool()
def audio_mix_plan(stems: str = "drums,bass,pads,sfx", target_lufs: float = -14.0) -> Dict[str, Any]:
    """Build a mix bus plan with headroom targets.

    Args:
        stems: Comma-separated stems.
        target_lufs: Integrated loudness target.
    """
    stem_list = [s.strip() for s in stems.split(",") if s.strip()]
    return {
        "target_integrated_lufs": target_lufs,
        "true_peak_ceiling_db": -1.0,
        "bus": ["DRY -> EQ hp20 -> COMP 2:1 slow -> SAT tape 5% -> LIMIT", "SFX sidechain -3dB duck on beat", "Reverb send 12%, delay 1/8 dotted 8%"],
        "per_stem_db": {s: (-8.0 if s.lower() in ("drums", "bass") else -12.0) for s in stem_list},
        "qc": ["mono-check low end", "phone-speaker check", "loop-click check at boundary"],
    }


@mcp.tool()
def audio_generate_sfx_brief(action: str = "dash", material: str = "cloth+wind", perspective: str = "third-person close") -> Dict[str, Any]:
    """Generate a layered SFX recipe.

    Args:
        action: Game action.
        material: Foleys/materials.
        perspective: Mix perspective.
    """
    return {
        "sfx": f"{action} / {material} / {perspective}",
        "layers": [f"L1 transient: {material} snap (5-15ms)", "L2 body: pitched whoosh down 300Hz", "L3 tail: room verb 0.6s"],
        "variation": "5 round-robins, pitch +-4%, no repetition within 3 plays",
        "implementation": "Wwise/FMOD event with RTPC speed->pitch, distance->lowpass",
    }


@mcp.tool()
def audio_director_note(states: str = "explore,combat,boss,resolve") -> Dict[str, Any]:
    """Design an adaptive music state machine for games.

    Args:
        states: Comma-separated music states.
    """
    sl = [s.strip() for s in states.split(",") if s.strip()]
    transitions = [{"from": sl[i], "to": sl[(i + 1) % len(sl)], "rule": "4-bar crossfade + stinger", "cooldown_s": 6} for i in range(len(sl))] if sl else []
    return {"states": sl, "transitions": transitions, "rule": "Never hard-cut; always carry pulse stem across states."}


if __name__ == "__main__":
    mcp.run()
