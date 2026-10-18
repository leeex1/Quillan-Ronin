#!/usr/bin/env python3
"""Resolve chunked execution refs."""
import json
import sqlite3
import sys

sys.stdout.reconfigure(encoding="utf-8")
db = sqlite3.connect(r"C:\Users\Admin\.n8n\database.sqlite")
row = db.execute(
    "SELECT id FROM execution_entity ORDER BY startedAt DESC LIMIT 1").fetchone()
d = json.loads(db.execute(
    "SELECT data FROM execution_data WHERE executionId=?",
    (row[0],)).fetchone()[0])
ref = d[2]
print("lastNodeExecuted ->", d[int(ref["lastNodeExecuted"])])
print("error ->", str(d[int(ref["error"])])[:400])
rd = d[int(ref["runData"])]
print("runData keys:", len(rd) if isinstance(rd, dict) else type(rd))
if isinstance(rd, dict):
    print("nodes:", sorted(rd.keys())[-8:])
