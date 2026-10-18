#!/usr/bin/env python3
"""Dump execution_data structure."""
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
for i, e in enumerate(d):
    if isinstance(e, dict):
        print(i, "dict keys:", list(e.keys())[:8])
    elif isinstance(e, list):
        print(i, f"list len {len(e)}",
              ("first: " + str(e[0])[:100]) if e else "")
    else:
        print(i, type(e).__name__, str(e)[:100])
