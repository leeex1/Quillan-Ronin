#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""In-process ASGI smoke test of the Ollama-compatible endpoints.

Uses FastAPI TestClient (ASGI transport, NO socket, NO network, NO Ollama).

  & 'C:\\Users\\Admin\\.local\\bin\\python3.11.exe' \\
      'C:\\02_QUILLAN\\09 - Projects\\mcp\\quillan_serve\\smoke_test_api.py'
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from fastapi.testclient import TestClient  # noqa: E402

from server import app  # noqa: E402

PROMPT = "Ronin:"
NUM_PREDICT = 2


def main() -> int:
    with TestClient(app) as client:
        health = client.get("/health")
        print("HEALTH_STATUS=" + str(health.status_code))
        print("HEALTH_DIGEST=" + health.json()["sha256"][:16])

        tags = client.get("/api/tags")
        model = tags.json()["models"][0]
        print("TAGS_STATUS=" + str(tags.status_code))
        print("TAGS_NAME=" + model["name"])
        print("TAGS_DECLARED_CTX=" + str(model["details"]["num_ctx_declared"]))
        print("TAGS_ACTIVE_WINDOW=" + str(model["details"]["active_window_tokens"]))

        gen = client.post(
            "/api/generate",
            json={
                "model": "quillan-ronin-mini:latest",
                "prompt": PROMPT,
                "stream": False,
                "options": {
                    "num_ctx": 2048,
                    "num_predict": NUM_PREDICT,
                    "temperature": 0.0,
                },
            },
        )
        print("GENERATE_STATUS=" + str(gen.status_code))
        body = gen.json()
        print("EXACT_OUTPUT=" + repr(body["response"]))
        print("EVAL_COUNT=" + str(body["eval_count"]))
        print("TOKENS_PER_SEC=" + f"{body['eval_tokens_per_second']:.4f}")
        print("WARNINGS=" + repr(body["warnings"]))

        over = client.post(
            "/api/generate",
            json={
                "model": "quillan-ronin-mini:latest",
                "prompt": "x",
                "options": {"num_predict": 300000},
            },
        )
        print("OVER_WINDOW_STATUS=" + str(over.status_code))

        ok = (
            health.status_code == 200
            and tags.status_code == 200
            and gen.status_code == 200
            and body["done"] is True
            and body["warnings"]
            and model["details"]["num_ctx_declared"] == 2000000
            and model["details"]["active_window_tokens"] == 262144
            and over.status_code == 400
        )
        print("API_SMOKE_PASS=" + str(ok))
        return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
