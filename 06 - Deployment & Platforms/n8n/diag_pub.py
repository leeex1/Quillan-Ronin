#!/usr/bin/env python3
"""Active flag + recent publish events + snapshot content check."""
import json
import sqlite3
import sys

sys.stdout.reconfigure(encoding="utf-8")
db = sqlite3.connect(r"C:\Users\Admin\.n8n\database.sqlite")
print("active now:",
      db.execute("SELECT name, active FROM workflow_entity "
                 "WHERE id='EmfoGhyIzSrktmIe'").fetchall())
print("recent publish events:")
for r in db.execute(
        "SELECT workflowId, versionId, event, userId, createdAt "
        "FROM workflow_publish_history ORDER BY createdAt DESC LIMIT 6"):
    print("  ", r[0][:8], (r[1] or "")[:8], r[2], r[3], r[4])
print("history cols:",
      [c[1] for c in db.execute("PRAGMA table_info(workflow_history)")])
