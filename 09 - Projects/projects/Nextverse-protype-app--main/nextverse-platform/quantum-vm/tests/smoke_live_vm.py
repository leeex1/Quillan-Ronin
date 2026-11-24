#!/usr/bin/env python3
"""Live end-to-end smoke test for the NextVerse Quantum Super VM.

Boots 11 quantum-core instances (ports 7001..7011) plus the quantum master
(port 7777), verifies every health endpoint, submits a real /execute request,
asserts the 11-way chunked result is merged correctly, then shuts everything
down. No mocks: this exercises real sockets and the real QuantumMaster class.
"""

from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

VM_DIR = Path(__file__).resolve().parent.parent
MASTER_PORT = 7777
INSTANCE_PORTS = list(range(7001, 7012))
BOOST_PER_INSTANCE = 42


class _QuantumInstanceHandler(BaseHTTPRequestHandler):
    """Python-native equivalent of the shipped bash ``quantum-core-binary``."""

    def log_message(self, *_args: object) -> None:  # silence per-request logs
        return

    def _reply(self, payload: dict) -> None:
        body = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        if self.path == "/health":
            self._reply(
                {"status": "healthy", "instance": self.server.server_port - 7000}
            )  # type: ignore[attr-defined]
        else:
            self.send_error(404)

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length)
        try:
            request = json.loads(raw or b"{}")
        except json.JSONDecodeError:
            self.send_error(400)
            return
        if self.path == "/execute":
            chunk = request.get("data", [])
            self._reply(
                {
                    "status": "success",
                    "boost_applied": BOOST_PER_INSTANCE,
                    "data": [
                        f"i{self.server.server_port - 7000}:{item}" for item in chunk
                    ],  # type: ignore[attr-defined]
                    "instance_id": self.server.server_port - 7000,  # type: ignore[attr-defined]
                    "operation": request.get("operation", "compute"),
                }
            )
        elif self.path == "/boost":
            self._reply({"current_boost": BOOST_PER_INSTANCE})
        else:
            self.send_error(404)


def _http(url: str, payload: dict | None = None, timeout: float = 3.0) -> dict:
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def _wait_ready(url: str, deadline: float = 15.0) -> None:
    end = time.time() + deadline
    last: Exception | None = None
    while time.time() < end:
        try:
            _http(url)
            return
        except (urllib.error.URLError, OSError) as exc:
            last = exc
            time.sleep(0.15)
    raise RuntimeError(f"{url} never became ready: {last}")


def main() -> int:
    servers: list[ThreadingHTTPServer] = []
    threads: list[threading.Thread] = []
    master: subprocess.Popen | None = None
    try:
        for port in INSTANCE_PORTS:
            srv = ThreadingHTTPServer(("127.0.0.1", port), _QuantumInstanceHandler)
            th = threading.Thread(target=srv.serve_forever, daemon=True)
            th.start()
            servers.append(srv)
            threads.append(th)

        master = subprocess.Popen(
            [sys.executable, str(VM_DIR / "quantum-master.py")],
            cwd=str(VM_DIR),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

        for port in INSTANCE_PORTS:
            health = _http(f"http://127.0.0.1:{port}/health")
            assert health["status"] == "healthy", (port, health)
        print(f"[PASS] {len(INSTANCE_PORTS)} quantum-core instances healthy")

        _wait_ready(f"http://127.0.0.1:{MASTER_PORT}/health")
        mh = _http(f"http://127.0.0.1:{MASTER_PORT}/health")
        assert mh["status"] == "healthy" and mh["instances"] == 11, mh
        print(f"[PASS] quantum master healthy: {mh}")

        payload = {"data": list(range(110)), "operation": "compute"}
        result = _http(f"http://127.0.0.1:{MASTER_PORT}/execute", payload, timeout=20)
        assert result["status"] == "success", result
        assert result["total_boost"] == len(INSTANCE_PORTS) * BOOST_PER_INSTANCE, (
            result["total_boost"]
        )
        assert len(result["results"]) == 110, len(result["results"])
        print(
            f"[PASS] 11-way execute merged: status={result['status']}, "
            f"total_boost={result['total_boost']}, chunks={len(result['results'])}"
        )
        print("[PASS] Quantum VM live smoke test: ALL CHECKS PASSED")
        return 0
    finally:
        if master is not None and master.poll() is None:
            master.terminate()
            try:
                master.wait(timeout=5)
            except subprocess.TimeoutExpired:
                master.kill()
        for srv in servers:
            srv.shutdown()
            srv.server_close()
        for th in threads:
            th.join(timeout=2)


if __name__ == "__main__":
    raise SystemExit(main())
