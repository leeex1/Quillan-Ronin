#!/usr/bin/env python3
"""Post a diagnostic event to the local Quillan collector.
Usage: diag_emit.py <agent> <type> <detail> [stage] [status] [tokens]
Localhost only. Stdlib only.
"""
import json
import sys
import urllib.request

BASE = "http://127.0.0.1:8777/api/events"

agent, etype, detail = sys.argv[1], sys.argv[2], sys.argv[3]
ev = {"agent": agent, "type": etype, "detail": detail}
if len(sys.argv) > 4 and sys.argv[4]:
    ev["stage"] = sys.argv[4]
if len(sys.argv) > 5 and sys.argv[5]:
    ev["status"] = sys.argv[5]
if len(sys.argv) > 6:
    ev["tokens"] = int(sys.argv[6])

req = urllib.request.Request(
    BASE, data=json.dumps(ev).encode(),
    headers={"Content-Type": "application/json"}, method="POST")
print(urllib.request.urlopen(req, timeout=8).read().decode())
