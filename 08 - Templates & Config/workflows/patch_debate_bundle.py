#!/usr/bin/env python3
"""Debate forwards the bundle (query/prism/personas) downstream to gates."""
import json
import sqlite3
import sys

sys.stdout.reconfigure(encoding="utf-8")
db = sqlite3.connect(r"C:\Users\Admin\.n8n\database.sqlite")
row = db.execute(
    "SELECT nodes FROM workflow_entity WHERE id=?",
    ("EmfoGhyIzSrktmIe",)).fetchone()
nodes = {n["name"]: n for n in json.loads(row[0])}
djs = nodes["Debate Round 2"]["parameters"]["jsCode"]
old = "verdicts: round2,"
assert old in djs
nodes["Debate Round 2"]["parameters"]["jsCode"] = djs.replace(
    old,
    "verdicts: round2,\n  bundle: incomingDebate.bundle || null,", 1)
print("debate forwards bundle")
P = r"C:\02_QUILLAN\08 - Templates & Config\workflows\emfo-main.json"
open(P, "w", encoding="utf-8").write(json.dumps(
    {"id": "EmfoGhyIzSrktmIe", "name": "Quillan Sovereign \u2014 MAIN",
     "nodes": list(nodes.values()),
     "connections": json.loads(db.execute(
         "SELECT connections FROM workflow_entity WHERE id=?",
         ("EmfoGhyIzSrktmIe",)).fetchone()[0]),
     "settings": {"executionOrder": "v1"}, "staticData": None, "pinData": {}},
    indent=2))
print("saved")
