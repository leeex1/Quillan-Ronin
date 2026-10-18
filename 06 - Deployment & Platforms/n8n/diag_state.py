#!/usr/bin/env python3
"""Quick state: MAIN row, active, webhooks."""
import sqlite3
import sys

sys.stdout.reconfigure(encoding="utf-8")
db = sqlite3.connect(r"C:\Users\Admin\.n8n\database.sqlite")
print("MAIN:",
      db.execute("SELECT name, active FROM workflow_entity "
                 "WHERE id='EmfoGhyIzSrktmIe'").fetchall())
print("webhooks:",
      db.execute("SELECT webhookPath, method, workflowId "
                 "FROM webhook_entity").fetchall())
