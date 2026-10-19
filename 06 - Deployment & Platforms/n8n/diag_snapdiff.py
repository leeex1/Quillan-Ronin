#!/usr/bin/env python3
"""Diff published snapshot vs live workflow row."""
import hashlib
import json
import sqlite3
import sys

sys.stdout.reconfigure(encoding="utf-8")


def h(o):
    return hashlib.md5(json.dumps(o, sort_keys=True).encode()).hexdigest()[:12]


db = sqlite3.connect(r"C:\Users\Admin\.n8n\database.sqlite")
pub = db.execute(
    "SELECT publishedVersionId FROM workflow_published_version "
    "WHERE workflowId='EmfoGhyIzSrktmIe'").fetchone()[0]
snap = json.loads(db.execute(
    "SELECT nodes, connections FROM workflow_history WHERE versionId=?",
    (pub,)).fetchone()[0] and json.dumps({}) or "{}") if False else None
row = db.execute(
    "SELECT nodes, connections FROM workflow_history WHERE versionId=?",
    (pub,)).fetchone()
snodes = {n["name"]: n for n in json.loads(row[0])}
sconns = json.loads(row[1])
live = db.execute(
    "SELECT nodes, connections FROM workflow_entity WHERE id=?",
    ("EmfoGhyIzSrktmIe",)).fetchone()
lnodes = {n["name"]: n for n in json.loads(live[0])}
lconns = json.loads(live[1])
print("snap nodes:", len(snodes), "| live nodes:", len(lnodes))
only_live = sorted(set(lnodes) - set(snodes))
only_snap = sorted(set(snodes) - set(lnodes))
print("only live:", only_live if only_live else "none")
print("only snap:", only_snap if only_snap else "none")
diff = 0
for k in sorted(set(snodes) & set(lnodes)):
    if h(snodes[k].get("parameters", {})) != h(lnodes[k].get("parameters", {})):
        diff += 1
        if diff <= 10:
            print("PARAMS DIFF:", k)
print("param diffs:", diff)
print("conns equal:", h(sconns) == h(lconns))
