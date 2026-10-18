#!/usr/bin/env python3
"""Audit every node: triggers, unwired, parked, terminals."""
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

outs = {k: [t["node"] for o in v.get("main", []) for t in o]
        for k, v in conns.items()}
inns = {}
for k, v in conns.items():
    for o in v.get("main", []):
        for t in o:
            inns.setdefault(t["node"], []).append(k)

print("== triggers ==")
for n in nodes:
    if "rigger" in n or "ebhook" in n:
        print(" ", n, "->", outs.get(n, []))
print("== nodes with NO incoming (excluding triggers/stickies) ==")
for n in nodes:
    if not inns.get(n) and "rigger" not in n and "ebhook" not in n \
            and "tick" not in n.lower():
        print(" ", n, "| out:", outs.get(n, [])[:3])
print("== nodes with NO outgoing (excluding terminals/stickies/parked) ==")
terms = [n for n in nodes if not outs.get(n)]
print(" ", len(terms), "terminals")
for n in sorted(terms):
    print("   ", n)
