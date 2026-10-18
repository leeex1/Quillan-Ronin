#!/usr/bin/env python3
"""Respond uses its direct input (C0 Throne), not the deleted sim node."""
import json
import sqlite3
import sys

sys.stdout.reconfigure(encoding="utf-8")
db = sqlite3.connect(r"C:\Users\Admin\.n8n\database.sqlite")
row = db.execute(
    "SELECT nodes FROM workflow_entity WHERE id=?",
    ("EmfoGhyIzSrktmIe",)).fetchone()
nodes = json.loads(row[0])
for n in nodes:
    if n["name"] == "\U0001f4e4 Respond":
        n["parameters"]["responseBody"] = "={{ $json }}"
        print("respond reads direct input")
db.execute("UPDATE workflow_entity SET nodes=? WHERE id=?",
           (json.dumps(nodes), "EmfoGhyIzSrktmIe"))
db.commit()
P = r"C:\02_QUILLAN\08 - Templates & Config\workflows\emfo-main.json"
wf = json.load(open(P, encoding="utf-8"))
w = wf[0] if isinstance(wf, list) else wf
for n in w["nodes"]:
    if n["name"] == "\U0001f4e4 Respond":
        n["parameters"]["responseBody"] = "={{ $json }}"
open(P, "w", encoding="utf-8").write(
    json.dumps(w if not isinstance(wf, list) else [w], indent=2))
print("file synced")
