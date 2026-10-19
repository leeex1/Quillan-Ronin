#!/usr/bin/env python3
"""Full publish: snapshot live canvas into workflow_history + point
published_version at it. Creates rows if missing."""
import datetime
import json
import shutil
import sqlite3
import sys
import uuid

sys.stdout.reconfigure(encoding="utf-8")
shutil.copy(r"C:\Users\Admin\.n8n\database.sqlite",
            r"C:\Users\Admin\.n8n\database.sqlite.bak-pub")
db = sqlite3.connect(r"C:\Users\Admin\.n8n\database.sqlite")
WID = "EmfoGhyIzSrktmIe"
live = db.execute(
    "SELECT nodes, connections, name FROM workflow_entity WHERE id=?",
    (WID,)).fetchone()
now = datetime.datetime.now(datetime.timezone.utc).isoformat()
vid = str(uuid.uuid4())
old = db.execute(
    "SELECT versionId, autosaved, description, nodeGroups FROM workflow_history "
    "WHERE workflowId=? ORDER BY updatedAt DESC LIMIT 1",
    (WID,)).fetchone()
auto, desc, groups = (old[1], old[2], old[3]) if old else (0, "", "[]")
db.execute(
    "INSERT INTO workflow_history (versionId, workflowId, authors, createdAt, "
    "updatedAt, nodes, connections, name, autosaved, description, nodeGroups) "
    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
    (vid, WID, "quillan-cli", now, now, live[0], live[1], live[2],
     auto, desc, groups))
db.execute("INSERT INTO workflow_published_version "
           "(workflowId, publishedVersionId, createdAt, updatedAt) "
           "VALUES (?, ?, ?, ?) "
           "ON CONFLICT(workflowId) DO UPDATE SET publishedVersionId="
           "excluded.publishedVersionId, updatedAt=excluded.updatedAt",
           (WID, vid, now, now))
db.commit()
print("published snapshot:", vid[:8])
print("verify:",
      db.execute("SELECT publishedVersionId FROM workflow_published_version "
                 "WHERE workflowId=?", (WID,)).fetchone()[0][:8])
