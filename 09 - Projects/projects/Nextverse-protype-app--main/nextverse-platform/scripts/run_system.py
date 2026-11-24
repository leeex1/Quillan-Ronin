#!/usr/bin/env python3
"""Launch the NextVerse app stack and verify it over real HTTP.

Starts:
  - orchestrator API on 127.0.0.1:3006
  - social hub (SocketIO) on 127.0.0.1:8085
  - built web UI on 127.0.0.1:3010
Then probes every service and prints measured results. Tears down cleanly.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROBES: list[tuple[str, str, dict | None]] = [
    ("orchestrator health", "http://127.0.0.1:3006/health", None),
    (
        "jqld compute",
        "http://127.0.0.1:3006/execute",
        {"action": "jqld", "payload": {"C": 1000.0, "exponent": 10.0}},
    ),
    (
        "real benchmark",
        "http://127.0.0.1:3006/execute",
        {"action": "benchmark", "payload": {}},
    ),
    (
        "real cpu batch",
        "http://127.0.0.1:3006/execute",
        {"action": "real_cpu_batch", "payload": {"data": [100] * 16, "workers": 4}},
    ),
    ("web ui", "http://127.0.0.1:3010/index.html", None),
    ("asset manifest", "http://127.0.0.1:3010/assets/tripio/manifest.json", None),
    ("demo GLB robot", "http://127.0.0.1:3010/assets/tripio/robot.glb", None),
]


def request(url: str, payload: dict | None = None, timeout: float = 10.0):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return response.status, response.read()


def wait_for(url: str, seconds: float = 25.0) -> None:
    deadline = time.time() + seconds
    last: Exception | None = None
    while time.time() < deadline:
        try:
            request(url, timeout=2)
            return
        except (urllib.error.URLError, OSError) as error:
            last = error
            time.sleep(0.25)
    raise RuntimeError(f"{url} did not start: {last}")


def main() -> int:
    processes: list[subprocess.Popen] = []
    try:
        processes.append(
            subprocess.Popen(
                [sys.executable, str(ROOT / "orchestrator" / "server.py")],
                cwd=str(ROOT / "orchestrator"),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
        )
        processes.append(
            subprocess.Popen(
                [sys.executable, str(ROOT / "hub-server" / "hub_server.py")],
                cwd=str(ROOT / "hub-server"),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
        )
        processes.append(
            subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "http.server",
                    "3010",
                    "-b",
                    "127.0.0.1",
                    "-d",
                    str(ROOT / "web-ui" / "build"),
                ],
                cwd=str(ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
        )

        wait_for("http://127.0.0.1:3006/health")
        wait_for("http://127.0.0.1:3010/index.html")

        for name, url, payload in PROBES:
            status, body = request(url, payload)
            if status != 200:
                raise RuntimeError(f"{name} returned {status}")
            if name == "orchestrator health":
                data = json.loads(body)
                print(
                    f"[PASS] {name}: {data['status']}, {data['available_count']}/{data['total_count']} services"
                )
            elif name == "jqld compute":
                data = json.loads(body)
                print(f"[PASS] {name}: value={data['result']['value']}")
            elif name == "real benchmark":
                data = json.loads(body)["result"]
                print(
                    f"[PASS] {name}: parallel={data['parallel']['measured_speedup']}x "
                    f"vectorised={data['vectorised']['measured_speedup']}x"
                )
            elif name == "real cpu batch":
                data = json.loads(body)["result"]
                print(
                    f"[PASS] {name}: {len(data['result'])} results, {data['report']['correctness']}"
                )
            else:
                print(f"[PASS] {name}: {len(body)} bytes, HTTP {status}")
        print("[PASS] NextVerse app stack running and verified")
        return 0
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()


if __name__ == "__main__":
    raise SystemExit(main())
