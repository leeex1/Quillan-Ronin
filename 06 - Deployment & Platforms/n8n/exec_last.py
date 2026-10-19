#!/usr/bin/env python3
"""Show the latest execution and its error."""
import json
import sqlite3
import sys

sys.stdout.reconfigure(encoding="utf-8")
db = sqlite3.connect(r"C:\Users\Admin\.n8n\database.sqlite")
row = db.execute(
    "SELECT id, workflowId, status, mode, startedAt, stoppedAt "
    "FROM execution_entity ORDER BY startedAt DESC LIMIT 1").fetchone()
print("latest:", row[2], "| mode:", row[3], "| wf:", row[1][:8], "| at:", row[4])
det = db.execute("SELECT data FROM execution_data WHERE executionId=?",
                 (row[0],)).fetchone()
if not det or not det[0]:
    print("no detail stored")
    sys.exit()
d = json.loads(det[0])
if isinstance(d, list):
    print("detail is list, len", len(d))
    sys.exit()
res = (d.get("resultData") or {})
err = res.get("error") or {}
print("ERROR:", str(err.get("message"))[:300])
node = err.get("node") or {}
print("NODE:", node.get("name") if isinstance(node, dict) else node)
rd = res.get("runData") or {}
print("nodes ran:", len(rd))
