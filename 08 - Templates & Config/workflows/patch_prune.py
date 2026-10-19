#!/usr/bin/env python3
"""Prune dead weight from MAIN. Everything removed is preserved in
YdcY canvas + .bak files + exports. Backup first."""
import json
import shutil
import sqlite3
import sys

sys.stdout.reconfigure(encoding="utf-8")
db = sqlite3.connect(r"C:\Users\Admin\.n8n\database.sqlite")
row = db.execute(
    "SELECT nodes, connections FROM workflow_entity WHERE id=?",
    ("EmfoGhyIzSrktmIe",)).fetchone()
nodes = {n["name"]: n for n in json.loads(row[0])}
conns = json.loads(row[1])

DEAD = (["AI Agent"] + [f"AI Agent{i}" for i in range(1, 34)]
        + ["C0-QUILLAN \u00b7LIVE", "OpenRouter Chat Model",
           "Log Live Voice", "Read/Write Files from Disk",
           "Convert to File", "Zone: parking"])
gone = [n for n in DEAD if n in nodes]
for n in gone:
    del nodes[n]
    conns.pop(n, None)
for k, v in list(conns.items()):
    for out in v.get("main", []):
        out[:] = [t for t in out if t.get("node") not in set(DEAD)]
print("removed:", len(gone))
print("remaining nodes:", len(nodes))

# dead-end audit of what is left
outs = {k: [t["node"] for o in v.get("main", []) for t in o]
        for k, v in conns.items()}
terms = sorted(n for n in nodes if not outs.get(n))
print("terminals left:", terms)

P = r"C:\02_QUILLAN\08 - Templates & Config\workflows\emfo-main.json"
shutil.copy(P, P + ".bak-prune")
open(P, "w", encoding="utf-8").write(json.dumps(
    {"id": "EmfoGhyIzSrktmIe", "name": "Quillan Sovereign \u2014 MAIN",
     "nodes": list(nodes.values()), "connections": conns,
     "settings": {"executionOrder": "v1"}, "staticData": None, "pinData": {}},
    indent=2))
print("saved")
