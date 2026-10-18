#!/usr/bin/env python3
"""Version tables + latest run bundle check."""
import json
import sqlite3
import sys

sys.stdout.reconfigure(encoding="utf-8")
db = sqlite3.connect(r"C:\Users\Admin\.n8n\database.sqlite")
tbls = [r[0] for r in db.execute(
    "SELECT name FROM sqlite_master WHERE type='table'")]
print("version-ish tables:",
      [t for t in tbls if "version" in t.lower() or "histor" in t.lower()])
row = db.execute(
    "SELECT id, startedAt FROM execution_entity ORDER BY startedAt DESC LIMIT 1"
).fetchone()
print("latest exec:", row[1])
print("published versions:")
for r in db.execute(
        "SELECT workflowId, versionId, createdAt FROM workflow_published_version"):
    print(" ", r[0][:8], r[1], r[2])
d = json.loads(db.execute(
    "SELECT data FROM execution_data WHERE executionId=?",
    (row[0],)).fetchone()[0])
ROOT = d if isinstance(d, list) else []


def res(o, depth=0):
    if isinstance(o, str) and o.isdigit() and depth < 12:
        try:
            return res(ROOT[int(o)], depth + 1)
        except (IndexError, ValueError):
            return o
    if isinstance(o, dict):
        return {k: res(v, depth + 1) for k, v in o.items()}
    if isinstance(o, list):
        return [res(v, depth + 1) for v in o]
    return o


def walk(o, depth=0):
    if isinstance(o, dict):
        if "Debate Round 2" in o:
            return o
        if depth < 8:
            for v in o.values():
                r = walk(v, depth + 1)
                if r is not None:
                    return r
    elif isinstance(o, list) and depth < 8:
        for v in o:
            r = walk(v, depth + 1)
            if r is not None:
                return r
    return None


rd = res(walk(d)) or {}
e = rd.get("Debate Round 2")
ex = (e[0] if isinstance(e, list) else e) if e else {}
try:
    items = (ex.get("data") or {}).get("main", [[[]]])[0] or []
    j = items[0].get("json", {}) if items else {}
    print("debate out keys:", list(j.keys())[:10])
except Exception as ex2:
    print("parse issue:", str(ex2)[:100])
