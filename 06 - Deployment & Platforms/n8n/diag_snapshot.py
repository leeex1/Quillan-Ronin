#!/usr/bin/env python3
"""What code does the PUBLISHED snapshot hold?"""
import json
import sqlite3
import sys

sys.stdout.reconfigure(encoding="utf-8")
db = sqlite3.connect(r"C:\Users\Admin\.n8n\database.sqlite")
pub = db.execute(
    "SELECT publishedVersionId FROM workflow_published_version "
    "WHERE workflowId='EmfoGhyIzSrktmIe'").fetchone()[0]
print("published version:", pub)
row = db.execute(
    "SELECT nodes FROM workflow_history WHERE versionId=?", (pub,)).fetchone()
if not row:
    print("NO snapshot row!")
    sys.exit()
nodes = {n["name"]: n for n in json.loads(row[0])}
print("snapshot nodes:", len(nodes))
for nm in ("Debate Round 2", "C0-QUILLAN Throne", "Gate Ethics VIR",
           "Prepare 34 Assignments"):
    js = (nodes.get(nm) or {}).get("parameters", {}).get("jsCode", "")
    marks = []
    if "bundle: incomingDebate" in js:
        marks.append("BUNDLE-OK")
    if "$('Debate Round 2')" in js:
        marks.append("THRONE-FROM-DEBATE")
    if "s.bundle && s.bundle.prism" in js:
        marks.append("ETHICS-HARDENED")
    print(f"{nm}: {' '.join(marks) if marks else 'MISSING-FIXES'}")
