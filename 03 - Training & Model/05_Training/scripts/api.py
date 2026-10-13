#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
👑 QUILLAN-RONIN v5.3.1 — ZERO-DEPENDENCY OPENAI REST API SERVER
========================================================================================
Production-grade, stdlib-native HTTP service providing:
- OpenAI-compatible /v1/chat/completions (with optional SSE streaming).
- Models catalog (/v1/models) & Telemetry (/health).
- Standard CORS headers for web dashboards & browser extensions.
- Zero external package dependencies (built entirely on Python stdlib).
========================================================================================
"""

from __future__ import annotations

import os
import sys
import json
import time
import logging
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
from typing import Dict, Any, Optional, List

SCRIPTS_DIR = Path("C:/02_QUILLAN/scripts")
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from sovereign_master_orchestrator import get_master_orchestrator, SovereignMasterOrchestrator
from sovereign_inference_engine import SamplingParams

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

LOGGER = logging.getLogger("quillan.api")
if not LOGGER.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", "%H:%M:%S"))
    LOGGER.addHandler(handler)
    LOGGER.setLevel(logging.INFO)


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    """Multi-threaded HTTP server handling concurrent requests deterministically."""
    daemon_threads = True
    allow_reuse_address = True


class SovereignAPIHandler(BaseHTTPRequestHandler):
    """HTTP Request Handler implementing OpenAI v1 API endpoints."""

    def _set_cors_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")

    def do_OPTIONS(self):
        self.send_response(204)
        self._set_cors_headers()
        self.end_headers()

    def do_GET(self):
        if self.path == "/health":
            self.send_response(200)
            self._set_cors_headers()
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            resp = {
                "status": "healthy",
                "model": "quillan-ronin-v5.3.1",
                "engine": "sovereign-master-orchestrator",
                "timestamp": time.time()
            }
            self.wfile.write(json.dumps(resp).encode("utf-8"))
            return

        if self.path in ("/v1/models", "/models"):
            self.send_response(200)
            self._set_cors_headers()
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            resp = {
                "object": "list",
                "data": [
                    {
                        "id": "quillan-ronin-v5.3.1",
                        "object": "model",
                        "created": 1777000000,
                        "owned_by": "quillan-foundation",
                        "description": "34-Expert MoE Sovereign Model (479M params, 9-Vector Prism, 4-Stage Output)",
                    }
                ]
            }
            self.wfile.write(json.dumps(resp).encode("utf-8"))
            return

        self.send_response(404)
        self._set_cors_headers()
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b'{"error": "Endpoint not found"}')

    def do_POST(self):
        if self.path in ("/v1/chat/completions", "/chat/completions"):
            content_len = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_len)
            try:
                data = json.loads(body.decode("utf-8"))
            except Exception as e:
                self.send_response(400)
                self._set_cors_headers()
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": f"Invalid JSON: {e}"}).encode("utf-8"))
                return

            messages = data.get("messages", [])
            stream = bool(data.get("stream", False))
            max_tokens = int(data.get("max_tokens", 180))
            temperature = float(data.get("temperature", 0.65))

            # Extract user query
            user_query = ""
            for msg in reversed(messages):
                if msg.get("role") == "user":
                    user_query = str(msg.get("content", ""))
                    break
            if not user_query and messages:
                user_query = str(messages[-1].get("content", ""))

            orchestrator = get_master_orchestrator()
            params = SamplingParams(
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_k=40,
                top_p=0.85,
                min_p=0.05,
                repetition_penalty=1.25,
                frequency_penalty=0.35,
                no_repeat_ngram_size=3,
                use_kv_cache=True,
            )

            chat_id = f"chatcmpl-{int(time.time() * 1000)}"
            created_ts = int(time.time())

            if stream:
                self.send_response(200)
                self._set_cors_headers()
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "keep-alive")
                self.end_headers()

                full_response = orchestrator.generate_full_sovereign_response(user_query, params=params)
                chunk_size = 8
                words = full_response.split(" ")
                for i in range(0, len(words), chunk_size):
                    chunk_text = " ".join(words[i:i+chunk_size]) + " "
                    payload = {
                        "id": chat_id,
                        "object": "chat.completion.chunk",
                        "created": created_ts,
                        "model": "quillan-ronin-v5.3.1",
                        "choices": [{"index": 0, "delta": {"content": chunk_text}, "finish_reason": None}]
                    }
                    self.wfile.write(f"data: {json.dumps(payload)}\n\n".encode("utf-8"))
                    self.wfile.flush()
                    time.sleep(0.01)

                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
                return

            # Non-streaming response
            full_response = orchestrator.generate_full_sovereign_response(user_query, params=params)
            resp = {
                "id": chat_id,
                "object": "chat.completion",
                "created": created_ts,
                "model": "quillan-ronin-v5.3.1",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": full_response},
                        "finish_reason": "stop"
                    }
                ],
                "usage": {
                    "prompt_tokens": len(user_query.split()),
                    "completion_tokens": len(full_response.split()),
                    "total_tokens": len(user_query.split()) + len(full_response.split())
                }
            }
            self.send_response(200)
            self._set_cors_headers()
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(resp).encode("utf-8"))
            return

        self.send_response(404)
        self._set_cors_headers()
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b'{"error": "Endpoint not found"}')


def run_api_server(host: str = "127.0.0.1", port: int = 8000):
    """Launches the production-grade stdlib HTTP API server."""
    # Pre-warm model in memory
    LOGGER.info("Pre-warming Sovereign Master Orchestrator...")
    get_master_orchestrator()

    server_address = (host, port)
    httpd = ThreadedHTTPServer(server_address, SovereignAPIHandler)
    LOGGER.info("👑 Quillan-Ronin API Gateway listening on http://%s:%d", host, port)
    LOGGER.info("Ready for requests from QuillanWorker, Web UIs, and MCP agents.")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        LOGGER.info("Shutting down API server.")
        httpd.server_close()


if __name__ == "__main__":
    run_api_server()
