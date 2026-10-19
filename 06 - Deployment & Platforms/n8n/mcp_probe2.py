#!/usr/bin/env python3
"""Re-probe rag + thinking_engine with byte-safe IO."""
import json
import subprocess
import sys

sys.stdout.reconfigure(encoding="utf-8")
PY = r"C:\Users\Admin\AppData\Local\Programs\Python\Python314\python.exe"
BASE = r"C:\02_QUILLAN\09 - Projects\mcp"
msgs = ("\n".join([
    json.dumps({"jsonrpc": "2.0", "id": 1, "method": "initialize",
                "params": {"protocolVersion": "2024-11-05",
                           "capabilities": {},
                           "clientInfo": {"name": "probe", "version": "1"}}}),
    json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}),
    json.dumps({"jsonrpc": "2.0", "id": 2, "method": "tools/list",
                "params": {}})]) + "\n").encode()

for s in ("quillan_rag", "quillan_thinking_engine"):
    try:
        p = subprocess.run(
            [PY, f"{BASE}\\{s}\\server.py"], input=msgs, capture_output=True,
            timeout=25)
        out = p.stdout.decode("utf-8", errors="replace")
        err = p.stderr.decode("utf-8", errors="replace")
        tools = None
        for line in out.splitlines():
            try:
                d = json.loads(line)
            except ValueError:
                continue
            if isinstance(d, dict) and d.get("id") == 2:
                r = d.get("result") or {}
                tools = r.get("tools")
                if tools is None:
                    print(s, "id:2 result keys:", list(r.keys())[:6])
                    print(s, "raw:", str(d)[:300])
        if isinstance(tools, list):
            print(s, "OK:", [t.get("name") for t in tools])
        elif tools is None:
            print(s, "NO id:2 response. stderr tail:",
                  err.strip().splitlines()[-2:])
    except subprocess.TimeoutExpired:
        print(s, "HANG")
