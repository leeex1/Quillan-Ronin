#!/usr/bin/env python3
"""List recent executions with statuses."""
import sqlite3
import sys

sys.stdout.reconfigure(encoding="utf-8")
db = sqlite3.connect(r"C:\Users\Admin\.n8n\database.sqlite")
names = {r[0]: r[1][:30] for r in db.execute(
    "SELECT id, name FROM workflow_entity")}
for r in db.execute(
        "SELECT workflowId, status, mode, startedAt, stoppedAt "
        "FROM execution_entity ORDER BY startedAt DESC LIMIT 6"):
    print(f"{r[3][:19]}Z | {names.get(r[0], '?'):30} | {r[1]:9} | {r[2]} | stop={str(r[4])[:19]}")
