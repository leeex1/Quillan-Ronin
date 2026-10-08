#!/usr/bin/env python3
"""Delete all unutilized nodes from MAIN. Backups hold everything."""
import json
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
           "Convert to File", "Zone: parking", "Sticky TODO export"])
# only delete if truly unused (no live links either direction)
for n in DEAD:
    if n not in nodes:
        print("already gone:", n)
        continue
    inl = [k for k, v in conns.items()
           for o in v.get("main", []) for t in o if t.get("node") == n]
    outl = [t.get("node") for o in conns.get(n, {}).get("main", [])
            for t in o]
    if inl or outl:
        print("STILL WIRED, keeping:", n, "in:", inl, "out:", outl)
        continue
    del nodes[n]
    conns.pop(n, None)
    print("deleted:", n)

# confirm zero islands remain (stickies exempt)
outs = {k: [t["node"] for o in v.get("main", []) for t in o]
        for k, v in conns.items()}
inns = set()
for v in conns.values():
    for o in v.get("main", []):
        for t in o:
            inns.add(t.get("node"))
islands = [n for n in nodes
           if n not in conns or not outs.get(n)]
islands = [n for n in islands
           if n not in inns and "tick" not in n.lower()
           and nodes[n]["type"] != "n8n-nodes-base.stickyNote"]
print("unwired non-sticky nodes left:", islands if islands else "NONE - clean")

P = r"C:\02_QUILLAN\08 - Templates & Config\workflows\emfo-main.json"
open(P, "w", encoding="utf-8").write(json.dumps(
    {"id": "EmfoGhyIzSrktmIe", "name": "Quillan Sovereign \u2014 MAIN",
     "nodes": list(nodes.values()), "connections": conns,
     "settings": {"executionOrder": "v1"}, "staticData": None, "pinData": {}},
    indent=2))
print("saved", len(nodes), "nodes")
