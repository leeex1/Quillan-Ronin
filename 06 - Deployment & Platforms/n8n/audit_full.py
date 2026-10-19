#!/usr/bin/env python3
"""Ground-truth audit: every link, every island, every trigger path."""
import json
import sqlite3
import sys

sys.stdout.reconfigure(encoding="utf-8")
db = sqlite3.connect(r"C:\Users\Admin\.n8n\database.sqlite")
row = db.execute(
    "SELECT nodes, connections FROM workflow_entity WHERE id=?",
    ("EmfoGhyIzSrktmIe",)).fetchone()
nodes = {n["name"]: (n["type"], n.get("disabled", False))
         for n in json.loads(row[0])}
conns = json.loads(row[1])
print("nodes:", len(nodes))

outs = {}
for k, v in conns.items():
    if not isinstance(v, dict):
        print("ODD conns value:", k, type(v))
        continue
    outs[k] = [t["node"] for o in v.get("main", []) for t in o
               if isinstance(o, list)]
inns = {}
for k, tgts in outs.items():
    for t in tgts:
        inns.setdefault(t, []).append(k)

print("\n== triggers and their reach ==")
for n, (t, _) in nodes.items():
    if "rigger" in n or "ebhook" in n:
        print(f"TRIGGER {n} [{t}] -> {outs.get(n, [])}")

print("\n== nodes with NO incoming (roots) ==")
for n in nodes:
    if n not in inns and "tick" not in n.lower():
        print("  ROOT:", n, "| out:", outs.get(n, [])[:3])

print("\n== nodes with NO outgoing (ends) ==")
for n in nodes:
    if not outs.get(n):
        print("  END:", n)

print("\n== disabled nodes ==")
print(" ", [n for n, (t, d) in nodes.items() if d] or "none")
