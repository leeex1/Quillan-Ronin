#!/usr/bin/env python3
"""MCP stdio handshake probe: initialize + tools/list, 25s per server."""
import json
import subprocess
import sys

sys.stdout.reconfigure(encoding="utf-8")
PY = r"C:\Users\Admin\AppData\Local\Programs\Python\Python314\python.exe"
BASE = r"C:\02_QUILLAN\09 - Projects\mcp"
SERVERS = ["quillan_rag", "quillan_thinking_engine", "quillan_samurai_engine",
           "quillan_buddy", "quillan_code", "quillan_model_lab",
           "quillan_council", "quillan_governor", "quillan_prism"]

msgs = ("\n".join([
    json.dumps({"jsonrpc": "2.0", "id": 1, "method": "initialize",
                "params": {"protocolVersion": "2024-11-05",
                           "capabilities": {},
                           "clientInfo": {"name": "probe",
                                           "version": "1"}}}),
    json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}),
    json.dumps({"jsonrpc": "2.0", "id": 2, "method": "tools/list",
                "params": {}})]) + "\n")

for s in SERVERS:
    try:
        p = subprocess.run(
            [PY, f"{BASE}\\{s}\\server.py"], input=msgs, capture_output=True,
            text=True, encoding="utf-8", errors="replace", timeout=25)
        tools = 0
        for line in p.stdout.splitlines():
            try:
                d = json.loads(line)
            except ValueError:
                continue
            if isinstance(d, dict) and d.get("id") == 2:
                tools = len(((d.get("result") or {}).get("tools")) or [])
        print(("OK   " if tools else "EMPTY"), f"{s}: {tools} tools",
              (p.stderr.strip().splitlines() or [""])[-1][:100]
              if not tools else "")
    except subprocess.TimeoutExpired:
        print("HANG", s, "(no response in 25s)")
    except Exception as e:
        print("FAIL", s, str(e)[:120])
