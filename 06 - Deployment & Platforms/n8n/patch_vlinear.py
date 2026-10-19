#!/usr/bin/env python3
"""Voice lane linear: Debate -> Emit Council -> Build -> Call -> Shape -> Log."""
import json
import shutil
import sqlite3
import sys

sys.stdout.reconfigure(encoding="utf-8")
shutil.copy(r"C:\Users\Admin\.n8n\database.sqlite",
            r"C:\Users\Admin\.n8n\database.sqlite.bak-vlin")
db = sqlite3.connect(r"C:\Users\Admin\.n8n\database.sqlite")
row = db.execute(
    "SELECT nodes, connections FROM workflow_entity WHERE id=?",
    ("EmfoGhyIzSrktmIe",)).fetchone()
nodes = {n["name"]: n for n in json.loads(row[0])}
conns = json.loads(row[1])

# debate drops the third fan-out
dout = [t for o in conns.get("Debate Round 2", {}).get("main", [])
        for t in o if t.get("node") != "Build Voice Requests"]
conns["Debate Round 2"] = {"main": [dout]}
# emit council feeds build (leaf -> chain link)
eout = [t for o in conns.get("Emit Council", {}).get("main", [])
        for t in o]
if not any(t.get("node") == "Build Voice Requests" for t in eout):
    eout.append({"node": "Build Voice Requests", "type": "main", "index": 0})
conns["Emit Council"] = {"main": [eout]}
# build reads debate state via ref (emit's HTTP response carries nothing)
bjs = nodes["Build Voice Requests"]["parameters"]["jsCode"]
old_head = "const d = $input.first().json;"
assert old_head in bjs, "build head not found: " + bjs[:120]
nodes["Build Voice Requests"]["parameters"]["jsCode"] = bjs.replace(
    old_head, "const d = $('Debate Round 2').first().json;", 1)
print("debate outs:", [t["node"] for t in dout])
print("emit-council outs:", [t["node"] for t in eout])
db.execute("UPDATE workflow_entity SET nodes=?, connections=? WHERE id=?",
           (json.dumps(list(nodes.values())), json.dumps(conns),
            "EmfoGhyIzSrktmIe"))
db.commit()
print("saved")
