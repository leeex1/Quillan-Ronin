#!/usr/bin/env python3
"""
Quillan Multimodal & Vision Intelligence Service
================================================
Powered by NVIDIA NIM:
- Primary Vision: meta/llama-3.2-11b-vision-instruct / meta/llama-3.2-90b-vision-instruct
- Analyzes diagrams, model architectures, 3D meshes/renderings, and album artwork.
"""

import os
import base64
import logging
from pathlib import Path
from typing import Optional
import httpx

NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY", "")
NIM_BASE = "https://integrate.api.nvidia.com/v1"
PRIMARY_VISION_MODEL = "meta/llama-3.2-11b-vision-instruct"

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

    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(
            f"{NIM_BASE}/chat/completions",
            headers={
                "Authorization": f"Bearer {NVIDIA_API_KEY}",
                "Content-Type": "application/json"
            },
            json=payload
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

if __name__ == "__main__":
    import asyncio
    print("Quillan Vision Service initialized.")
