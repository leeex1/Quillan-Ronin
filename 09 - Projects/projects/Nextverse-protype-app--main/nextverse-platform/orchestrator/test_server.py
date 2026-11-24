"""Live HTTP integration test for the orchestrator API."""

from __future__ import annotations

import pytest
from aiohttp.test_utils import TestClient, TestServer

from server import create_app


@pytest.mark.anyio
async def test_http_health_and_execute() -> None:
    async with TestClient(TestServer(create_app())) as client:
        response = await client.get(
            "/health", headers={"Origin": "http://127.0.0.1:3010"}
        )
        assert response.status == 200
        assert response.headers.get("Access-Control-Allow-Origin") == "*"
        body = await response.json()
        assert body["status"] == "healthy"
        assert body["available_count"] == 7

        response = await client.post(
            "/execute",
            json={"action": "jqld", "payload": {"C": 1000.0, "exponent": 10.0}},
        )
        assert response.status == 200
        result = await response.json()
        assert result["status"] == "success"
        assert result["result"]["value"] == pytest.approx(1024000.0, rel=1e-9)

        response = await client.post(
            "/execute", json={"action": "unknown", "payload": {}}
        )
        assert response.status == 404
    print("[PASS] orchestrator HTTP health + execute + error handling")
