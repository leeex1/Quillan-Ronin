#!/usr/bin/env python3
"""Inspect webhook registrations + workflow active state."""
import sqlite3
import sys

sys.stdout.reconfigure(encoding="utf-8")
db = sqlite3.connect(r"C:\Users\Admin\.n8n\database.sqlite")
print("-- workflows --")
for r in db.execute("SELECT id, name, active FROM workflow_entity"):
    print(("ACTIVE " if r[2] else "off    ") + (r[1][:40] + " " + r[0][:8]))
print("-- webhook_entity --")
try:
    rows = db.execute(
        "SELECT webhookPath, method, workflowId FROM webhook_entity").fetchall()
    for r in rows:
        print(r)
    if not rows:
        print("(empty - nothing registered)")
except Exception as e:
    print("tbl fail:", e)
