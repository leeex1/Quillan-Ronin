#!/usr/bin/env python3
"""Surgical DB fix: re-attach the voice lane. No file round-trips."""
import json
import shutil
import sqlite3
import sys

sys.stdout.reconfigure(encoding="utf-8")
shutil.copy(r"C:\Users\Admin\.n8n\database.sqlite",
            r"C:\Users\Admin\.n8n\database.sqlite.bak-voicelink")
db = sqlite3.connect(r"C:\Users\Admin\.n8n\database.sqlite")
row = db.execute(
    "SELECT nodes, connections FROM workflow_entity WHERE id=?",
    ("EmfoGhyIzSrktmIe",)).fetchone()
nodes = json.loads(row[0])
conns = json.loads(row[1])
assert "Build Voice Requests" in {n["name"] for n in nodes}, "voice lane missing!"

outs = conns.get("Debate Round 2", {}).get("main", [[[]]])[0]
print("debate outs before:", [t["node"] for t in outs])
if not any(t.get("node") == "Build Voice Requests" for t in outs):
    outs.append({"node": "Build Voice Requests", "type": "main", "index": 0})
    conns["Debate Round 2"] = {"main": [outs]}
    print("voice lane re-attached")
else:
    print("already attached")

# full end-to-end trace from both triggers
def trace(start, seen=None):
    seen = seen or set()
    if start in seen:
        return seen
    seen.add(start)
    for t in [x["node"] for o in conns.get(start, {}).get("main", [])
              for x in o]:
        trace(t, seen)
    return seen


manual = trace("When clicking \u2018Execute workflow\u2019")
web = trace("\U0001f6f0 Quillan Intake Webhook")
print("manual reaches:", len(manual), "nodes; throne:", "C0-QUILLAN Throne" in manual,
      "; file1:", "Convert to File1" in manual)
print("webhook reaches:", len(web), "nodes; throne:", "C0-QUILLAN Throne" in web,
      "; respond:", "\U0001f4e4 Respond" in web)
db.execute("UPDATE workflow_entity SET connections=? WHERE id=?",
           (json.dumps(conns), "EmfoGhyIzSrktmIe"))
db.commit()
print("saved")
