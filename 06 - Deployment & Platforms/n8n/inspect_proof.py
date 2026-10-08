#!/usr/bin/env python3
"""Inspect the proof workflow as imported."""
import json
import sqlite3
import sys

sys.stdout.reconfigure(encoding="utf-8")
db = sqlite3.connect(r"C:\Users\Admin\.n8n\database.sqlite")
row = db.execute(
    "SELECT nodes, connections FROM workflow_entity WHERE id=?",
    ("b7c9d0e1-2f3a-4b5c-8d9e-0f123456789a",)).fetchone()
nodes = json.loads(row[0])
for n in nodes:
    print("-", n["name"], "|", n["type"], "| v" + str(n.get("typeVersion")),
          "| creds:", list((n.get("credentials") or {}).keys()))
print("conns:", json.dumps(json.loads(row[1]))[:500])
