#!/usr/bin/env python3
"""Read n8n execution history: what the user actually ran and how it ended."""
import sqlite3
import sys

sys.stdout.reconfigure(encoding="utf-8")
db = sqlite3.connect(r"C:\Users\Admin\.n8n\database.sqlite")
tbls = [r[0] for r in db.execute(
    "SELECT name FROM sqlite_master WHERE type='table'")]
print("exec tables:", [t for t in tbls if "exec" in t.lower()])
names = {r[0]: r[1] for r in db.execute(
    "SELECT id, name FROM workflow_entity")}
try:
    rows = db.execute(
        "SELECT id, workflowId, status, mode, startedAt, stoppedAt "
        "FROM execution_entity ORDER BY startedAt DESC LIMIT 12").fetchall()
except Exception as e:
    print("query fail:", e)
    sys.exit()
for r in rows:
    print(f"{r[4][:19]} | {names.get(r[1], r[1][:8])[:34]:34} | {r[2]:8} | mode={r[3]}")
    det = db.execute(
        "SELECT data FROM execution_data WHERE executionId=?", (r[0],)).fetchone()
    if det and det[0]:
        import json
        try:
            d = json.loads(det[0])
            err = ((d.get("resultData") or {}).get("error") or {})
            if err:
                print("   ERROR:", str(err.get("message"))[:220])
        except Exception as e:
            print("   (unparseable detail)", e)
