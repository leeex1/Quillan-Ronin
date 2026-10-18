#!/usr/bin/env python3
"""Persistent-session probe: keep stdin open, wait up to 90s for tools/list."""
import json
import subprocess
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")
PY = r"C:\Users\Admin\AppData\Local\Programs\Python\Python314\python.exe"
p = subprocess.Popen(
    [PY, r"C:\02_QUILLAN\09 - Projects\mcp\quillan_rag\server.py"],
    stdin=subprocess.PIPE, stdout=subprocess.PIPE,
    stderr=subprocess.DEVNULL, bufsize=1)


def send(obj):
    p.stdin.write((json.dumps(obj) + "\n").encode())
    p.stdin.flush()


send({"jsonrpc": "2.0", "id": 1, "method": "initialize",
      "params": {"protocolVersion": "2024-11-05", "capabilities": {},
                 "clientInfo": {"name": "p", "version": "1"}}})
send({"jsonrpc": "2.0", "method": "notifications/initialized"})
send({"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}})

import select  # noqa: E402
deadline = time.time() + 90
buf = b""
got = False
try:
    while time.time() < deadline:
        chunk = p.stdout.read1(4096) if hasattr(p.stdout, "read1") else p.stdout.read(4096)
        if not chunk:
            break
        buf += chunk
        while b"\n" in buf:
            line, buf = buf.split(b"\n", 1)
            try:
                d = json.loads(line.decode("utf-8", errors="replace"))
            except ValueError:
                continue
            if isinstance(d, dict) and d.get("id") == 2:
                r = d.get("result") or {}
                tools = r.get("tools") or []
                print("TOOLS:", [t.get("name") for t in tools])
                got = True
                break
        if got:
            break
    if not got:
        print("NO tools/list response in 90s (hangs under persistent session too)")
finally:
    p.kill()
