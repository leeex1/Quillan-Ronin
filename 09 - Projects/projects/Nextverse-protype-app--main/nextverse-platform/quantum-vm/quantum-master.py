#!/usr/bin/env python3
import asyncio
import contextvars
import json
import aiohttp
from aiohttp import web

# Per-task session: concurrent coordinate_task calls must never share a
# ClientSession, or one call's cleanup closes another call's connector.
_SESSION = contextvars.ContextVar("nextverse_vm_session", default=None)
import logging


class QuantumMaster:
    def __init__(self, copies=11):
        self.copies = copies
        self.instances = {}

    async def coordinate_task(self, task):
        """Distribute a task across all instances without losing data.

        Floor-division chunking silently dropped the remainder whenever
        len(data) was not divisible by ``copies``.
        """
        data = task.get("data", []) or []
        base, remainder = divmod(len(data), self.copies)
        start = 0
        requests = []
        for i in range(self.copies):
            size = base + (1 if i < remainder else 0)
            chunk = data[start : start + size]
            start += size
            task_data = {
                "chunk_id": i,
                "total_chunks": self.copies,
                "data": chunk,
                "is_empty": size == 0,
                "operation": task.get("operation", "compute"),
            }
            requests.append(
                self.send_to_instance(f"http://localhost:{7000 + i + 1}", task_data)
            )

        session = aiohttp.ClientSession()
        token = _SESSION.set(session)
        try:
            raw_results = await asyncio.gather(*requests, return_exceptions=True)
        finally:
            await session.close()
            _SESSION.reset(token)
        results = [
            {"status": "error", "message": str(item)}
            if isinstance(item, BaseException)
            else item
            for item in raw_results
        ]
        return self.merge_results(results)

    async def send_to_instance(self, url, data):
        # Reuse a coordinator session when present; one session per chunk wastes
        # sockets and leaks connectors under load.
        session = _SESSION.get()
        if session is not None and not session.closed:
            async with session.post(f"{url}/execute", json=data) as resp:
                return await resp.json()
        async with aiohttp.ClientSession() as temporary_session:
            async with temporary_session.post(f"{url}/execute", json=data) as resp:
                return await resp.json()

    def merge_results(self, results):
        """Merge results from all quantum instances"""
        merged = {"status": "success", "total_boost": 0, "results": []}

        for result in results:
            if result.get("status") == "success":
                merged["total_boost"] += result.get("boost_applied", 0)
                merged["results"].extend(result.get("data", []))

        return merged

    async def health_check(self, request):
        return web.json_response({"status": "healthy", "instances": self.copies})

    async def execute(self, request):
        data = await request.json()
        result = await self.coordinate_task(data)
        return web.json_response(result)


if __name__ == "__main__":
    master = QuantumMaster()
    app = web.Application()
    app.router.add_get("/health", master.health_check)
    app.router.add_post("/execute", master.execute)

    # Locale-safe startup banner: emoji crash on Windows cp1252 consoles.
    print("Quantum Master running on http://localhost:7777")
    web.run_app(app, host="localhost", port=7777)
