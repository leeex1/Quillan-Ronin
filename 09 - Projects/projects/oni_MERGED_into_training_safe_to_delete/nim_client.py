#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
👑 QUILLAN-RONIN v5.4.0 ONI — NVIDIA NIM CLIENT ADAPTER
=============================================================================
Unified OpenAI / LangChain-compatible interface for NVIDIA NIM endpoints.
Default Model: meta/muse-glimmer-30b (with function calling & streaming).
"""

import os
from typing import Generator, Dict, Any, List, Optional
import httpx

NIM_BASE_URL = os.environ.get("NIM_BASE_URL", "https://integrate.api.nvidia.com/v1")
DEFAULT_MODEL = os.environ.get("NIM_MODEL", "meta/muse-glimmer-30b")

# Canonical NIM Model Rotation Roster
NIM_ROSTER = [
    "meta/muse-glimmer-30b",
    "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning",
    "nvidia/nemotron-3.5-lightning-30b-a3b",
    "mistralai/mistral-large-2-instruct",
    "meta/llama-3.2-11b-vision-instruct",
    "meta/llama-3.2-90b-vision-instruct",
    "deepseek-ai/deepseek-coder-6.7b-instruct",
]

def get_nvidia_key() -> str:
    """Retrieve key from environment, Windows registry, or local git-ignored .env."""
    key = os.environ.get("NVIDIA_API_KEY", "")
    if key: return key.strip()
    if os.name == "nt":
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Environment") as rk:
                val, _ = winreg.QueryValueEx(rk, "NVIDIA_API_KEY")
                if val: return str(val).strip()
        except Exception:
            pass
    from pathlib import Path
    env_file = Path(r"C:\02_QUILLAN\.env")
    if env_file.exists():
        try:
            for line in env_file.read_text(encoding="utf-8").splitlines():
                if line.startswith("NVIDIA_API_KEY="):
                    return line.split("=", 1)[1].strip().strip("\"'")
        except Exception:
            pass
    return ""

class QuillanNIMClient:
    """High-throughput NIM integration client for Muse Glimmer, Nemotron, and Llama."""
    def __init__(self, api_key: Optional[str] = None, base_url: str = NIM_BASE_URL):
        self.api_key = api_key or get_nvidia_key()
        self.base_url = base_url.rstrip("/")
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    def chat_stream(
        self,
        messages: List[Dict[str, str]],
        model: str = DEFAULT_MODEL,
        temperature: float = 0.7,
        top_p: float = 0.95,
        max_tokens: int = 4096,
        tools: Optional[List[Dict[str, Any]]] = None
    ) -> Generator[str, None, None]:
        """Stream token chunks from NIM endpoint."""
        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "stream": True
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        with httpx.stream("POST", url, headers=self.headers, json=payload, timeout=60.0) as response:
            if response.status_code != 200:
                raise RuntimeError(f"NIM Error {response.status_code}: {response.read().decode('utf-8')}")
            for line in response.iter_lines():
                if line.startswith("data: "):
                    data_str = line[6:].strip()
                    if data_str == "[DONE]":
                        break
                    import json
                    try:
                        chunk = json.loads(data_str)
                        delta = chunk["choices"][0].get("delta", {})
                        content = delta.get("content")
                        if content:
                            yield content
                    except Exception:
                        continue

    def chat_complete(
        self,
        messages: List[Dict[str, str]],
        model: str = DEFAULT_MODEL,
        temperature: float = 0.7,
        max_tokens: int = 2048
    ) -> str:
        """Standard synchronous completion."""
        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False
        }
        with httpx.Client(timeout=30.0) as client:
            r = client.post(url, headers=self.headers, json=payload)
            if r.status_code != 200:
                raise RuntimeError(f"NIM Error {r.status_code}: {r.text}")
            return r.json()["choices"][0]["message"]["content"]

    def chat_stream_cascade(
        self,
        messages: List[Dict[str, str]],
        roster: Optional[List[str]] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        tools: Optional[List[Dict[str, Any]]] = None
    ) -> Generator[str, None, None]:
        """Automatically cascade through candidate models in roster if a model is busy or cold-starting."""
        models_to_try = roster or NIM_ROSTER
        last_error = None
        for candidate in models_to_try:
            try:
                for token in self.chat_stream(
                    messages=messages,
                    model=candidate,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    tools=tools
                ):
                    yield token
                return
            except Exception as e:
                last_error = e
                continue
        if last_error:
            raise RuntimeError(f"All candidate NIM models failed. Last error: {last_error}")
