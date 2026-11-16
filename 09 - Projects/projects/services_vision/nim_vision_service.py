#!/usr/bin/env python3
"""
Quillan Multimodal & Vision Intelligence Service
================================================
Powered by NVIDIA NIM:
- Primary Vision: meta/llama-3.2-11b-vision-instruct / meta/llama-3.2-90b-vision-instruct
- Analyzes diagrams, model architectures, 3D meshes/renderings, and album artwork.
"""

import asyncio
import os
import base64
import logging
import random
from pathlib import Path
from typing import Optional
import httpx

NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY", "").strip()
NIM_BASE = "https://integrate.api.nvidia.com/v1"
PRIMARY_VISION_MODEL = "nvidia/nemotron-3.5-super-vl-120b-a12b"
MAX_IMAGE_BYTES = 10 * 1024 * 1024
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s [Vision] %(message)s")
log = logging.getLogger("quillan_vision")

def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

async def analyze_image(
    image_path: str,
    prompt: str = "Analyze this image in detail and describe its core components, structure, and artistic/technical intent.",
    model: str = PRIMARY_VISION_MODEL,
    max_tokens: int = 1024
) -> str:
    path = Path(image_path)
    if not path.exists():
        return f"Error: Image file not found at {image_path}"
    if path.stat().st_size > MAX_IMAGE_BYTES:
        return f"Error: Image exceeds {MAX_IMAGE_BYTES} byte limit: {image_path}"
    
    b64_img = encode_image(str(path))
    ext = path.suffix.lower().replace(".", "")
    if ext == "jpg": ext = "jpeg"
    media_type = f"image/{ext}" if ext in ["png", "jpeg", "webp", "gif"] else "image/png"

    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{media_type};base64,{b64_img}"
                        }
                    }
                ]
            }
        ],
        "max_tokens": max_tokens,
        "temperature": 0.2
    }

    api_key = get_nvidia_api_key()
    if not api_key:
        raise RuntimeError("NVIDIA_API_KEY is not set and was not found in C:\\02_QUILLAN\\.env")

    async with httpx.AsyncClient(timeout=NIM_TIMEOUT) as client:
        for attempt in range(3):
            try:
                resp = await client.post(
                    f"{NIM_BASE}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json"
                    },
                    json=payload
                )
                if resp.status_code in (429, 500, 502, 503, 504) and attempt < 2:
                    await asyncio.sleep((2 ** attempt) + random.uniform(0.1, 0.5))
                    continue
                resp.raise_for_status()
                msg = resp.json()["choices"][0]["message"]
                return msg.get("content") or msg.get("reasoning_content") or ""
            except (httpx.TimeoutException, httpx.TransportError):
                if attempt == 2:
                    raise
                await asyncio.sleep((2 ** attempt) + random.uniform(0.1, 0.5))

if __name__ == "__main__":
    import asyncio
    print("Quillan Vision Service initialized.")
