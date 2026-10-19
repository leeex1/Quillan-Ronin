#!/usr/bin/env python3
"""Single-end tail: Throne -> KillChecks -> Post -> Recollect -> EmitArb
-> Aggregate -> Code -> Code1 -> File1. No forks that don't rejoin."""
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


def outs(k):
    return [t["node"] for o in conns.get(k, {}).get("main", [])
            for t in o]


print("throne feeds:", outs("C0-QUILLAN Throne"))
# 1. throne -> killchecks ONLY
conns["C0-QUILLAN Throne"] = {"main": [[{"node": "Emit Kill-Checks",
                                         "type": "main", "index": 0}]]}
# 2. arbitration -> aggregate (tail continues, single line)
arb = outs("Emit Arbitration")
if "Aggregate" not in arb:
    arb.append({"node": "Aggregate", "type": "main", "index": 0})
    conns["Emit Arbitration"] = {"main": [arb]}
print("throne feeds now:", outs("C0-QUILLAN Throne"))
print("arb feeds now:", outs("Emit Arbitration"))
# verify full tail chain
chain = ["C0-QUILLAN Throne", "Emit Kill-Checks", "Post Kill-Checks",
         "Recollect Throne", "Emit Arbitration", "Aggregate",
         "Code in Python", "Code in Python1", "Convert to File1"]
ok = True
for a, b in zip(chain, chain[1:]):
    if b not in outs(a):
        print("BROKEN:", a, "->", b)
        ok = False
print("tail single-line:", ok)

P = r"C:\02_QUILLAN\08 - Templates & Config\workflows\emfo-main.json"
open(P, "w", encoding="utf-8").write(json.dumps(
    {"id": "EmfoGhyIzSrktmIe", "name": "Quillan Sovereign \u2014 MAIN",
     "nodes": list(nodes.values()), "connections": conns,
     "settings": {"executionOrder": "v1"}, "staticData": None, "pinData": {}},
    indent=2))
print("saved")
