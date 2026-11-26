#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Ollama-compatible HTTP front-end for the custom Quillan-Ronin-Mini arch.

Runs the REAL ``QuillanRoninSovereign`` checkpoint in PyTorch. It is NOT a proxy
to llama.cpp or Ollama: export_phase5_gguf.py:157-177 falls back to .pt for this
architecture, so no GGUF graph exists to register.

Endpoints: GET /health, /api/version, /api/tags, /api/show; POST /api/generate.

Launch (exact):
  & 'C:\\Users\\Admin\\.local\\bin\\python3.11.exe' -m uvicorn \
      quillan_serve.server:app --app-dir \
      'C:\\02_QUILLAN\\09 - Projects\\mcp' --host 127.0.0.1 --port 11435
"""

from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from engine import (
    ACTIVE_WINDOW_TOKENS,
    DECLARED_NUM_CTX,
    MODEL_NAME,
    MODEL_TAG,
    ContextWindowError,
    QuillanServingEngine,
)

_engine: Optional[QuillanServingEngine] = None


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global _engine
    _engine = QuillanServingEngine()
    try:
        yield
    finally:
        del _engine
        _engine = None


app = FastAPI(
    title="Quillan-Ronin-Mini Custom-Architecture Serving",
    version="1.0.0",
    description=(
        "PyTorch-native serving for QuillanRoninSovereign (815M, sha256 53e60d33...). "
        f"Declared num_ctx={DECLARED_NUM_CTX}; ACTIVE KV window={ACTIVE_WINDOW_TOKENS}; "
        "prompt is never truncated."
    ),
    lifespan=lifespan,
)


def _engine_or_503() -> QuillanServingEngine:
    if _engine is None:
        raise HTTPException(status_code=503, detail="model engine not loaded")
    return _engine


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class _Options(BaseModel):
    model_config = ConfigDict(extra="allow")
    num_ctx: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    num_predict: Optional[int] = None
    repeat_penalty: Optional[float] = None
    stop: Optional[List[str]] = None
    seed: Optional[int] = None


class GenerateRequest(BaseModel):
    model_config = ConfigDict(extra="allow")
    model: str = MODEL_NAME
    prompt: str
    system: Optional[str] = None
    suffix: Optional[str] = None
    stream: bool = False
    raw: bool = False
    format: Any = None
    keep_alive: Any = None
    options: Optional[_Options] = None
    stop: Optional[List[str]] = None


def _resolve(engine: QuillanServingEngine, req: GenerateRequest) -> Dict[str, Any]:
    opts = req.options or _Options()
    warnings: List[str] = []

    # Hard constraint: declared num_ctx is 2,097,152. A client asking for less
    # is NOT honoured as a backend shortcut -- we keep the declaration and say so.
    if opts.num_ctx is not None and int(opts.num_ctx) != DECLARED_NUM_CTX:
        warnings.append(
            f"num_ctx={opts.num_ctx} override ignored: declared context is fixed "
            f"at {DECLARED_NUM_CTX}; no Ollama backend shortcut is used."
        )

    system = req.system or ""
    prompt = (system + "\n" + req.prompt) if system else req.prompt
    stops = list(req.stop or opts.stop or [])

    return {
        "prompt": prompt,
        "num_predict": int(opts.num_predict if opts.num_predict is not None else 64),
        "temperature": float(opts.temperature if opts.temperature is not None else 0.0),
        "top_p": float(opts.top_p if opts.top_p is not None else 0.9),
        "repetition_penalty": float(
            opts.repeat_penalty if opts.repeat_penalty is not None else 1.15
        ),
        "stop": stops,
        "warnings": warnings,
    }


@app.get("/health")
def health() -> Dict[str, Any]:
    return _engine_or_503().health()


@app.get("/api/version")
def version() -> Dict[str, str]:
    return {"version": "0.0.0-quillan-custom-arch"}


@app.get("/api/tags")
def tags() -> Dict[str, Any]:
    engine = _engine_or_503()
    h = engine.health()
    return {
        "models": [
            {
                "name": MODEL_TAG,
                "model": MODEL_TAG,
                "modified_at": "2026-11-20T00:00:00Z",
                "size": h["checkpoint_bytes"],
                "digest": "sha256:" + h["sha256"],
                "details": {
                    "format": "safetensors",
                    "family": "quillan-ronin-mini",
                    "parameter_size": "815M",
                    "quantization_level": "bitnet1.58-int8+per-tensor-scale",
                    "num_ctx_declared": DECLARED_NUM_CTX,
                    "active_window_tokens": ACTIVE_WINDOW_TOKENS,
                    "prompt_truncated": False,
                    "backend": "pytorch-custom-arch",
                },
            }
        ]
    }


@app.get("/api/show")
def show(model: str = MODEL_NAME) -> Dict[str, Any]:
    _engine_or_503()
    if model not in (MODEL_NAME, MODEL_TAG, "ronin-mini", ""):
        raise HTTPException(status_code=404, detail=f"unknown model: {model}")
    return _engine_or_503().describe()


@app.post("/api/generate")
def generate(req: GenerateRequest, request: Request):
    engine = _engine_or_503()
    args = _resolve(engine, req)

    if not req.stream:
        try:
            result = engine.generate(
                args["prompt"],
                num_predict=args["num_predict"],
                temperature=args["temperature"],
                top_p=args["top_p"],
                repetition_penalty=args["repetition_penalty"],
                stop=args["stop"],
                warnings=args["warnings"],
            )
        except ContextWindowError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        payload = result.to_dict()
        payload["warnings"] = list(dict.fromkeys(payload["warnings"]))
        return payload

    def _chunks():
        try:
            for chunk in engine.stream_generate(
                args["prompt"],
                num_predict=args["num_predict"],
                temperature=args["temperature"],
                top_p=args["top_p"],
                repetition_penalty=args["repetition_penalty"],
                stop=args["stop"],
                warnings=args["warnings"],
            ):
                yield json.dumps(chunk, ensure_ascii=False) + "\n"
        except ContextWindowError as exc:
            yield (
                json.dumps(
                    {"error": str(exc), "done": True, "done_reason": "context_window"}
                )
                + "\n"
            )

    return StreamingResponse(_chunks(), media_type="application/x-ndjson")


@app.get("/")
def root() -> Dict[str, Any]:
    return {
        "service": "quillan-ronin-mini-custom-arch",
        "declared_num_ctx": DECLARED_NUM_CTX,
        "active_window_tokens": ACTIVE_WINDOW_TOKENS,
        "docs": "/docs",
        "health": "/health",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=os.getenv("QUILLAN_SERVE_HOST", "127.0.0.1"),
        port=int(os.getenv("QUILLAN_SERVE_PORT", "11435")),
        log_level="info",
    )
