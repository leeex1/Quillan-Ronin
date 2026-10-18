#!/usr/bin/env python3
"""Current status: latest run + voices + one real OR response sample."""
import json
import sqlite3
import sys
import urllib.request

sys.stdout.reconfigure(encoding="utf-8")
db = sqlite3.connect(r"C:\Users\Admin\.n8n\database.sqlite")
for r in db.execute(
        "SELECT workflowId, status, mode, startedAt, stoppedAt "
        "FROM execution_entity ORDER BY startedAt DESC LIMIT 2"):
    print(f"{r[3][:19]}Z | {r[1]} | {r[0][:8]} | stop={str(r[4])[:19]}")
ev = json.load(urllib.request.urlopen(
    "http://127.0.0.1:8777/api/events", timeout=8))["events"]
live = [e for e in ev if e.get("agent") == "council-live"]
print("live voices:", len(live))
real = [e for e in live if len(str(e.get("detail") or "")) > 20]
print("non-empty voices:", len(real))
for e in real[-3:]:
    print(" -", str(e.get("detail"))[:150].replace(chr(10), " "))
