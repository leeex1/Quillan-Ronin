#!/usr/bin/env python3
"""Remove the stale EmitCouncil->Throne link + sim orphans. One parent per node."""
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

# 1. throne waits ONLY for the gate chain
ec = conns.get("Emit Council", {}).get("main", [[[]]])[0]
conns["Emit Council"] = {"main": [[t for t in ec
                                   if t.get("node") != "C0-QUILLAN Throne"]]}
print("emit council now feeds:",
      [t["node"] for t in conns["Emit Council"]["main"][0]])

# 2. delete sim orphans still on canvas
ORPHANS = ["\U0001f44d Throne Synthesis"]
for n in ORPHANS:
    if n in nodes:
        del nodes[n]
        conns.pop(n, None)
        print("deleted orphan:", n)

# 3. verify single-parent rule on the core chain
core = ["Debate Round 2", "Diffusion Refine", "Gate Logic C7",
        "Gate Ethics VIR", "Gate Truth C18", "Gate Clarity C15",
        "Gate Paradox C17", "C0-QUILLAN Throne"]
parents = {}
for k, v in conns.items():
    for o in v.get("main", []):
        for t in o:
            parents.setdefault(t.get("node"), []).append(k)
for c in core:
    print(f"parents({c}) =", parents.get(c, []))

P = r"C:\02_QUILLAN\08 - Templates & Config\workflows\emfo-charter.json"
open(P, "w", encoding="utf-8").write(json.dumps(
    {"id": "EmfoGhyIzSrktmIe", "name": "Quillan Sovereign \u2014 MAIN",
     "nodes": list(nodes.values()), "connections": conns,
     "settings": {"executionOrder": "v1"}, "staticData": None, "pinData": {}},
    indent=2))
print("saved", len(nodes), "nodes")
