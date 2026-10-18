#!/usr/bin/env python3
"""Dig the error out of list-format execution data."""
import json
import sqlite3
import sys

sys.stdout.reconfigure(encoding="utf-8")
db = sqlite3.connect(r"C:\Users\Admin\.n8n\database.sqlite")
row = db.execute(
    "SELECT id FROM execution_entity ORDER BY startedAt DESC LIMIT 1").fetchone()
det = db.execute("SELECT data FROM execution_data WHERE executionId=?",
                 (row[0],)).fetchone()[0]
d = json.loads(det)
print("type:", type(d).__name__, "len:", len(d))
for i, e in enumerate(d):
    if isinstance(e, dict) and (e.get("error") or e.get("status") == "error"):
        print(f"--- item {i} keys:", list(e.keys())[:10])
        print(json.dumps(e, indent=1)[:800])
        break
else:
    print("no error item; first item keys:", list(d[0].keys()) if d else None)
