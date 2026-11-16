#!/usr/bin/env python3
"""
Quillan Voice & Audio Intelligence Pipeline
===========================================
Integrates audio processing, lyrics analysis, and speech/music alignment
leveraging NVIDIA NIM multimodal capabilities and local feature extractors.
"""

import asyncio
import os
import json
import logging
import random
from pathlib import Path
from typing import Optional, List, Dict
import httpx

NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY", "").strip()
NIM_BASE = "https://integrate.api.nvidia.com/v1"
REASONING_MODEL = "nvidia/nemotron-3.5-lightning-30b-a3b"
NIM_TIMEOUT = httpx.Timeout(connect=10.0, read=90.0, write=30.0, pool=10.0)

def get_nvidia_api_key() -> str:
    key = os.environ.get("NVIDIA_API_KEY", "").strip()
    if key:
        return key
    for env_path in (Path(r"C:\02_QUILLAN\.env"), Path(__file__).resolve().parents[3] / ".env"):
        if env_path.exists():
            for line in env_path.read_text(encoding="utf-8", errors="ignore").splitlines():
                if line.strip().startswith("NVIDIA_API_KEY="):
                    val = line.split("=", 1)[1].strip().strip('"\'')
                    if val:
                        os.environ["NVIDIA_API_KEY"] = val
                        return val
    return ""

logging.basicConfig(level=logging.INFO, format="%(asctime)s [Audio] %(message)s")
log = logging.getLogger("quillan_audio")

async def analyze_track_metadata(audio_path: str, lyrics_text: Optional[str] = None) -> Dict[str, any]:
    """
    Extracts structural audio metadata and conducts stylistic & semantic synthesis.
    """
    path = Path(audio_path)
    if not path.exists():
        return {"error": f"Audio file not found: {audio_path}"}

    file_size_mb = round(path.stat().st_size / (1024 * 1024), 2)
    ext = path.suffix.lower()

    summary = {
        "filename": path.name,
        "extension": ext,
        "size_mb": file_size_mb,
        "has_lyrics": bool(lyrics_text),
    }

    try:
        import sys as _sys
        _mcp_dir = r"C:\02_QUILLAN\09 - Projects\mcp"
        if _mcp_dir not in _sys.path:
            _sys.path.insert(0, _mcp_dir)
        from _shared.media_utils import decode_audio_pyav
        decoded = decode_audio_pyav(path, max_seconds=15.0)
        summary.update({
            "codec": decoded["codec"],
            "sample_rate": decoded["source_sample_rate"],
            "channels": decoded["channels"],
            "duration_sec": decoded["duration_sec"],
            "rms_db": decoded["rms_db"],
            "bpm": decoded["bpm"],
            "brightness_centroid": decoded["brightness_centroid"],
        })
    except Exception as _dec_err:
        summary["audio_decode_note"] = str(_dec_err)

    if lyrics_text:
        # Perform poetic & harmonic analysis via NIM
        prompt = f"""You are Quillan's Music Architect & Lyricist Specialist.
Analyze the following track lyrics for structural flow, meter, emotional resonance, and genre signature.

Track: {path.name}
Lyrics:
{lyrics_text[:3000]}

Provide:
1. Genre & Rhythm Signature
2. Emotional Arc & Theme
3. Key Hooks / Rhyme Scheme Quality Score (1-10)
"""
        api_key = get_nvidia_api_key()
        if not api_key:
            summary["analysis"] = "NVIDIA_API_KEY is not set and was not found in C:\\02_QUILLAN\\.env"
            return summary

        last_exc = None
        async with httpx.AsyncClient(timeout=NIM_TIMEOUT) as client:
            for attempt in range(3):
                try:
                    resp = await client.post(
                        f"{NIM_BASE}/chat/completions",
                        headers={
                            "Authorization": f"Bearer {api_key}",
                            "Content-Type": "application/json"
                        },
                        json={
                            "model": REASONING_MODEL,
                            "messages": [{"role": "user", "content": prompt}],
                            "max_tokens": 512,
                            "temperature": 0.3
                        }
                    )
                    if resp.status_code in (429, 500, 502, 503, 504) and attempt < 2:
                        await asyncio.sleep((2 ** attempt) + random.uniform(0.1, 0.5))
                        continue
                    if resp.status_code == 200:
                        msg = resp.json()["choices"][0]["message"]
                        summary["analysis"] = msg.get("content") or msg.get("reasoning_content") or ""
                    else:
                        summary["analysis"] = f"Analysis failed with status code {resp.status_code}"
                    break
                except (httpx.TimeoutException, httpx.TransportError) as exc:
                    last_exc = exc
                    if attempt == 2:
                        raise
                    await asyncio.sleep((2 ** attempt) + random.uniform(0.1, 0.5))

    return summary

if __name__ == "__main__":
    print("Quillan Audio Service initialized.")
