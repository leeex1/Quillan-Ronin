#!/usr/bin/env python3
"""HTTP API for the NextVerse system orchestrator (aiohttp, already installed)."""

from __future__ import annotations

import os

from aiohttp import web

from system import SystemOrchestrator

orchestrator = SystemOrchestrator()


async def health(request: web.Request) -> web.Response:
    return web.json_response(orchestrator.health())


async def services(request: web.Request) -> web.Response:
    return web.json_response({"services": orchestrator.services()})


async def execute(request: web.Request) -> web.Response:
    try:
        payload = await request.json()
        result = orchestrator.execute(
            str(payload.get("action", "")), payload.get("payload", {})
        )
        return web.json_response({"status": "success", "result": result})
    except KeyError as error:
        return web.json_response({"status": "error", "message": str(error)}, status=404)
    except Exception as error:
        return web.json_response({"status": "error", "message": str(error)}, status=400)


@web.middleware
async def cors_middleware(request: web.Request, handler) -> web.StreamResponse:
    """Allow the local web UI (different port) to read real benchmark data."""
    if request.method == "OPTIONS":
        response = web.Response(status=204)
    else:
        response = await handler(request)
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return response


def create_app() -> web.Application:
    app = web.Application(middlewares=[cors_middleware])
    app.router.add_get("/health", health)
    app.router.add_get("/services", services)
    app.router.add_post("/execute", execute)
    return app


if __name__ == "__main__":  # pragma: no cover
    web.run_app(
        create_app(),
        host=os.environ.get("HOST", "127.0.0.1"),
        port=int(os.environ.get("PORT", 3006)),
    )
