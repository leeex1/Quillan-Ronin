"""Boot-test every stdio MCP server in the registry and check cross-platform parity.

Each server is spawned, sent a JSON-RPC `initialize`, and its whole process tree is
killed afterwards (no ghost processes). Exit code 0 only if all servers respond and
every platform config contains every canonical server (modulo aliases).
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

LOG = logging.getLogger("mcp-verify")
REPO = Path(r"C:\02_QUILLAN")
REGISTRY = REPO / "09 - Projects" / "mcp" / "mcp_registry.json"
ANTIGRAVITY = Path.home() / ".gemini" / "config" / "mcp_config.json"
TIMEOUT_S = 40
MAX_WORKERS = 4

INIT = json.dumps({
    "jsonrpc": "2.0", "id": 1, "method": "initialize",
    "params": {"protocolVersion": "2024-11-05", "capabilities": {},
               "clientInfo": {"name": "registry-verify", "version": "1.0"}},
}) + "\n"


def _kill_tree(proc: subprocess.Popen) -> None:
    if proc.poll() is None:
        if os.name == "nt":
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                           capture_output=True, check=False)
        else:
            proc.kill()


def boot(name: str, spec: dict) -> tuple[str, bool, str]:
    env = os.environ.copy()
    # Do not leak the launching interpreter's environment into child servers.
    for var in ("PYTHONHOME", "PYTHONPATH", "VIRTUAL_ENV", "PYTHONNOUSERSITE", "__PYVENV_LAUNCHER__"):
        env.pop(var, None)
    env.update(spec.get("env", {}))
    proc = subprocess.Popen([spec["command"], *spec["args"]], stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, env=env)
    result: list[str] = []
    t = threading.Thread(target=lambda: result.append(proc.stdout.readline()), daemon=True)
    try:
        proc.stdin.write(INIT)
        proc.stdin.flush()
        t.start()
        t.join(TIMEOUT_S)
        line = result[0].strip() if result else ""
        if line:
            try:
                if "result" in json.loads(line):
                    return name, True, "ok"
            except json.JSONDecodeError:
                pass
            return name, False, f"bad response: {line[:120]}"
        return name, False, "no response (timeout or crash)"
    except OSError as exc:
        return name, False, f"spawn error: {exc}"
    finally:
        _kill_tree(proc)


def parity(servers: dict) -> list[str]:
    problems: list[str] = []
    checks = {
        "opencode.json": (REPO / "opencode.json", lambda n, s: n),
        ".idx/mcp.json": (REPO / ".idx" / "mcp.json", lambda n, s: n),
        "antigravity": (ANTIGRAVITY, lambda n, s: s.get("aliases", {}).get("antigravity", n)),
    }
    for label, (path, namer) in checks.items():
        if not path.exists():
            problems.append(f"{label}: file missing ({path})")
            continue
        have = set(json.loads(path.read_text(encoding="utf-8")).get("mcpServers", {}))
        for n, s in servers.items():
            expected = namer(n, s)
            if expected not in have and not (label == "antigravity" and s["transport"] == "remote"):
                problems.append(f"{label}: missing {expected}")
    return problems


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    servers = json.loads(REGISTRY.read_text(encoding="utf-8"))["servers"]
    stdio = {n: s for n, s in servers.items() if s["transport"] == "stdio"}
    LOG.info("Booting %d stdio servers (%d remote skipped)...", len(stdio), len(servers) - len(stdio))
    failures = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        for name, ok, msg in pool.map(lambda kv: boot(*kv), stdio.items()):
            LOG.info("  %-24s %s %s", name, "OK  " if ok else "FAIL", "" if ok else msg)
            failures += not ok
    problems = parity(servers)
    for p in problems:
        LOG.info("  PARITY %s", p)
    LOG.info("boot failures=%d parity problems=%d", failures, len(problems))
    return 1 if failures or problems else 0


if __name__ == "__main__":
    sys.exit(main())

