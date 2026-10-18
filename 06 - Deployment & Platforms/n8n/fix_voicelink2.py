#!/usr/bin/env python3
"""Re-attach voice lane + SAVE (the missing commit)."""
import json
import shutil
import sqlite3
import sys

sys.stdout.reconfigure(encoding="utf-8")
shutil.copy(r"C:\Users\Admin\.n8n\database.sqlite",
            r"C:\Users\Admin\.n8n\database.sqlite.bak-voicelink2")
db = sqlite3.connect(r"C:\Users\Admin\.n8n\database.sqlite")
row = db.execute(
    "SELECT nodes, connections FROM workflow_entity WHERE id=?",
    ("EmfoGhyIzSrktmIe",)).fetchone()
nodes = json.loads(row[0])
conns = json.loads(row[1])
assert "Build Voice Requests" in {n["name"] for n in nodes}
dout = conns.get("Debate Round 2", {}).get("main", [[[]]])[0]
print("before:", [t["node"] for t in dout])
if not any(t.get("node") == "Build Voice Requests" for t in dout):
    dout.append({"node": "Build Voice Requests", "type": "main", "index": 0})
    conns["Debate Round 2"] = {"main": [dout]}
db.execute("UPDATE workflow_entity SET connections=? WHERE id=?",
           (json.dumps(conns), "EmfoGhyIzSrktmIe"))
db.commit()
chk = json.loads(db.execute(
    "SELECT connections FROM workflow_entity WHERE id=?",
    ("EmfoGhyIzSrktmIe",)).fetchone()[0])
print("after:",
      [t["node"] for t in chk.get("Debate Round 2", {}).get("main", [[[]]])[0]])
