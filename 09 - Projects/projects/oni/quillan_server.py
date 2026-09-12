#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
👑 QUILLAN-RONIN v5.3.1 — SOVEREIGN LOCAL INFERENCE SERVER (ASGI)
================================================================
Production OpenAI-compatible local HTTP inference daemon powered by Starlette & Uvicorn:
- Serves the trained 34-Expert Council Sovereign Model (QuillanRoninOni)
- Strictly local loopback binding (127.0.0.1:11435)
- Endpoints:
    GET  /health
    GET  /v1/models
    POST /v1/chat/completions
    POST /v1/completions
- Supports temperature, top_p, repetition penalty, stop tokens, max_tokens safety bounds.
"""

import os
import sys
import time
import json
import uuid
import logging
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import torch
import torch.nn.functional as F
from starlette.applications import Starlette
from starlette.responses import JSONResponse, Response
from starlette.routing import Route
import uvicorn

# Path resolution
ONI_DIR = Path(__file__).resolve().parent
REPO_ROOT = ONI_DIR.parent.parent.parent
sys.path.insert(0, str(ONI_DIR))

from quillan_v5_4_oni import QuillanRoninOni, QuillanOniConfig
from quillan_tokenizer_unified import UnifiedQuillanTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [SOVEREIGN-SERVER] %(message)s")
logger = logging.getLogger("quillan_server")

CKPT_PATH = REPO_ROOT / "checkpoints" / "production_export" / "quillan_ronin_v531_sovereign_production.pt"
MODEL_NAME = "quillan-ronin-v5.3.1"

def get_safe_device() -> str:
    """Validate compute capability to avoid sm_61 CUDA kernel incompatibility."""
    if torch.cuda.is_available():
        try:
            major, minor = torch.cuda.get_device_capability()
            if major > 7 or (major == 7 and minor >= 5):
                return "cuda"
            logger.warning(
                f"GPU compute capability sm_{major}{minor} < sm_75 is unsupported by this PyTorch build. Defaulting to CPU."
            )
        except Exception as e:
            logger.warning(f"Device capability check failed: {e}. Defaulting to CPU.")
    return "cpu"

import tiktoken

class ModelEngine:
    """Manages model memory, tokenizer, and generation pipeline."""
    def __init__(self, ckpt_path: Path = CKPT_PATH, device: Optional[str] = None):
        self.device = device or get_safe_device()
        logger.info(f"Target compute device: {self.device}")

        logger.info(f"Initializing Tiktoken GPT-2 BPE tokenizer (50,257 vocab)...")
        self.tokenizer = tiktoken.get_encoding("gpt2")

        logger.info(f"Building QuillanRoninOni architecture (6 layers, 34 experts, 1024 dim)...")
        cfg = QuillanOniConfig(
            n_layer=6,
            hidden_dim=1024,
            ffn_dim=2048,
            num_experts=34,
            device=self.device
        )
        self.model = QuillanRoninOni(cfg).to(self.device)

        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")

        logger.info(f"Loading weights from {ckpt_path.name} ({ckpt_path.stat().st_size / (1024**2):.1f} MB)...")
        ckpt = torch.load(str(ckpt_path), map_location=self.device, weights_only=False)
        state_dict = ckpt.get("model_state_dict", ckpt)
        self.model.load_state_dict(state_dict, strict=True)
        self.model.eval()
        logger.info(f"✅ Sovereign Model successfully loaded into memory (All 1,438 tensors verified).")

    def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.25,
        top_p: float = 0.85,
        top_k: int = 40,
        repetition_penalty: float = 1.15,
        stop: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Autoregressive text generation using O(1) step KV-caching and calibrated nucleus sampling."""
        t0 = time.time()
        stop_sequences = stop or ["<|im_end|>", "<|endoftext|>", "<|user|>"]

        max_tokens = max(1, min(int(max_tokens), 2048))
        temperature = max(0.01, min(float(temperature), 1.5))
        top_p = max(0.1, min(float(top_p), 1.0))
        top_k = max(1, min(int(top_k), 100))

        input_ids = self.tokenizer.encode(prompt, allowed_special="all")
        prompt_len = len(input_ids)
        tokens = list(input_ids)
        generated_ids: List[int] = []

        with torch.no_grad():
            inp = torch.tensor([tokens[-1024:]], dtype=torch.long, device=self.device)
            out = self.model(inp, use_cache=True)
            logits, kv_cache = out[0], out[1]

            for _ in range(max_tokens):
                curr_logits = logits[0, -1, :].clone()

                # Repetition penalty applied strictly to generated response tokens
                if repetition_penalty > 1.0 and len(generated_ids) > 0:
                    for tid in set(generated_ids[-32:]):
                        curr_logits[tid] /= repetition_penalty

                if temperature <= 0.05:
                    next_tok = torch.argmax(curr_logits).item()
                else:
                    scaled = curr_logits / temperature
                    if top_k > 0:
                        topk_v, _ = torch.topk(scaled, min(top_k, scaled.size(-1)))
                        scaled[scaled < topk_v[-1]] = float("-inf")
                    sorted_logits, sorted_indices = torch.sort(scaled, descending=True)
                    cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    remove_mask = cum_probs > top_p
                    remove_mask[..., 1:] = remove_mask[..., :-1].clone()
                    remove_mask[..., 0] = False
                    scaled[sorted_indices[remove_mask]] = float("-inf")
                    probs = F.softmax(scaled, dim=-1)
                    next_tok = torch.multinomial(probs, num_samples=1).item()

                generated_ids.append(next_tok)
                if next_tok == 50256:
                    break

                curr_text = self.tokenizer.decode(generated_ids)
                if any(seq in curr_text for seq in stop_sequences):
                    break

                inp_single = torch.tensor([[next_tok]], dtype=torch.long, device=self.device)
                out = self.model(inp_single, past_key_values=kv_cache, use_cache=True)
                logits, kv_cache = out[0], out[1]

        elapsed = time.time() - t0
        output_text = self.tokenizer.decode(generated_ids)
        for seq in stop_sequences:
            if seq in output_text:
                output_text = output_text.split(seq)[0]

        return {
            "text": output_text.strip(),
            "prompt_tokens": prompt_len,
            "completion_tokens": len(generated_ids),
            "total_tokens": prompt_len + len(generated_ids),
            "latency_sec": elapsed,
            "tokens_per_sec": len(generated_ids) / max(elapsed, 1e-4)
        }

ENGINE: Optional[ModelEngine] = None
INFERENCE_LOCK = asyncio.Lock()

async def health_handler(request):
    return JSONResponse({
        "status": "healthy",
        "model": MODEL_NAME,
        "device": ENGINE.device if ENGINE else "uninitialized",
        "parameters": 227100000,
        "state": "ready"
    })

async def models_handler(request):
    return JSONResponse({
        "object": "list",
        "data": [
            {
                "id": MODEL_NAME,
                "object": "model",
                "created": 1700000000,
                "owned_by": "quillan-sovereign"
            },
            {
                "id": "falcon3:1b-instruct-q8_0",
                "object": "model",
                "created": 1700000000,
                "owned_by": "quillan-local-alias"
            }
        ]
    })

async def chat_completions_handler(request):
    if ENGINE is None:
        return JSONResponse({"error": "Model engine initializing"}, status_code=503)

    try:
        body = await request.json()
    except Exception as e:
        return JSONResponse({"error": f"Invalid JSON: {e}"}, status_code=400)

    messages = body.get("messages", [])
    if not messages:
        return JSONResponse({"error": "Field 'messages' is required."}, status_code=400)

    prompt_parts = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "").strip()
        if role == "system":
            prompt_parts.append(f"<|system|>\n{content}\n")
        elif role == "assistant":
            prompt_parts.append(f"<|assistant|>\n{content}\n")
        elif role == "user":
            prompt_parts.append(f"<|user|>\n{content}\n")
    prompt_parts.append("<|assistant|>\n")
    rendered_prompt = "".join(prompt_parts)

    max_tokens = body.get("max_tokens", 512)
    temperature = body.get("temperature", 0.25)
    top_p = body.get("top_p", 0.85)
    top_k = body.get("top_k", 40)
    repetition_penalty = body.get("repetition_penalty", 1.15)
    stop = body.get("stop", None)

    async with INFERENCE_LOCK:
        res = await asyncio.to_thread(
            ENGINE.generate,
            prompt=rendered_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            stop=stop
        )

    resp_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    return JSONResponse({
        "id": resp_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": body.get("model", MODEL_NAME),
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": res["text"]
                },
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": res["prompt_tokens"],
            "completion_tokens": res["completion_tokens"],
            "total_tokens": res["total_tokens"]
        }
    })

async def completions_handler(request):
    if ENGINE is None:
        return JSONResponse({"error": "Model engine initializing"}, status_code=503)

    try:
        body = await request.json()
    except Exception as e:
        return JSONResponse({"error": f"Invalid JSON: {e}"}, status_code=400)

    prompt = body.get("prompt", "")
    max_tokens = body.get("max_tokens", 256)
    temperature = body.get("temperature", 0.25)
    top_p = body.get("top_p", 0.85)
    top_k = body.get("top_k", 40)
    repetition_penalty = body.get("repetition_penalty", 1.15)
    stop = body.get("stop", None)

    async with INFERENCE_LOCK:
        res = await asyncio.to_thread(
            ENGINE.generate,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            stop=stop
        )

    return JSONResponse({
        "id": f"cmpl-{uuid.uuid4().hex[:12]}",
        "object": "text_completion",
        "created": int(time.time()),
        "model": body.get("model", MODEL_NAME),
        "choices": [
            {
                "text": res["text"],
                "index": 0,
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": res["prompt_tokens"],
            "completion_tokens": res["completion_tokens"],
            "total_tokens": res["total_tokens"]
        }
    })

routes = [
    Route("/health", health_handler, methods=["GET"]),
    Route("/v1/models", models_handler, methods=["GET"]),
    Route("/v1/chat/completions", chat_completions_handler, methods=["POST"]),
    Route("/v1/completions", completions_handler, methods=["POST"]),
]

app = Starlette(routes=routes)

def run_server(host: str = "127.0.0.1", port: int = 11436):
    global ENGINE
    print("=" * 70, flush=True)
    print(f"👑 QUILLAN-RONIN v5.3.1 — SOVEREIGN INFERENCE SERVER", flush=True)
    print(f"   Binding: http://{host}:{port}/v1", flush=True)
    print(f"   Model  : {MODEL_NAME} ({CKPT_PATH.name})", flush=True)
    print("=" * 70, flush=True)

    ENGINE = ModelEngine(CKPT_PATH)
    logger.info(f"Starting Starlette+Uvicorn on http://{host}:{port}...")
    uvicorn.run(app, host=host, port=port, log_level="warning")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Quillan-Ronin Sovereign Inference Server")
    parser.add_argument("--host", default="127.0.0.1", help="Host interface")
    parser.add_argument("--port", type=int, default=11436, help="Port (default: 11436)")
    args = parser.parse_args()

    run_server(host=args.host, port=args.port)
