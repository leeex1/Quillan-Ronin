#!/usr/bin/env python3
"""Voice lane v3: Build -> Call(batched 5/15s) -> Shape -> Log. No cycles."""
import json
import shutil
import sqlite3
import sys

sys.stdout.reconfigure(encoding="utf-8")
shutil.copy(r"C:\Users\Admin\.n8n\database.sqlite",
            r"C:\Users\Admin\.n8n\database.sqlite.bak-v3")
db = sqlite3.connect(r"C:\Users\Admin\.n8n\database.sqlite")
row = db.execute(
    "SELECT nodes, connections FROM workflow_entity WHERE id=?",
    ("EmfoGhyIzSrktmIe",)).fetchone()
nodes = {n["name"]: n for n in json.loads(row[0])}
conns = json.loads(row[1])

nodes.pop("Batch Voices", None)
conns.pop("Batch Voices", None)
for k, v in list(conns.items()):
    for out in v.get("main", []):
        out[:] = [t for t in out
                  if t.get("node") not in ("Batch Voices",)]
conns["Build Voice Requests"] = {"main": [[{"node": "Call OpenRouter",
                                            "type": "main", "index": 0}]]}
nodes["Call OpenRouter"]["parameters"]["options"] = {
    "timeout": 90000, "batchSize": 5, "batchInterval": 15000}
print("lane: build -> call(batched) -> shape -> log; batch node deleted")
db.execute("UPDATE workflow_entity SET nodes=?, connections=? WHERE id=?",
           (json.dumps(list(nodes.values())), json.dumps(conns),
            "EmfoGhyIzSrktmIe"))
db.commit()
print("saved", len(nodes), "nodes")
