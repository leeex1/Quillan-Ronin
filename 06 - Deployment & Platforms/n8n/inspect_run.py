#!/usr/bin/env python3
"""Inspect shell + log node outputs in the latest run."""
import json
import sqlite3
import sys

sys.stdout.reconfigure(encoding="utf-8")
db = sqlite3.connect(r"C:\Users\Admin\.n8n\database.sqlite")
row = db.execute(
    "SELECT id FROM execution_entity ORDER BY startedAt DESC LIMIT 1").fetchone()
d = json.loads(db.execute(
    "SELECT data FROM execution_data WHERE executionId=?",
    (row[0],)).fetchone()[0])


def find_run_data(obj):
    if isinstance(obj, dict):
        if "AI Agent" in obj:
            return obj
        for v in obj.values():
            r = find_run_data(v)
            if r is not None:
                return r
    elif isinstance(obj, list):
        for v in obj:
            r = find_run_data(v)
            if r is not None:
                return r
    return None


# runData is referenced; find the dict containing node names
def walk(o, depth=0):
    if isinstance(o, dict):
        if "AI Agent" in o and "Debate Round 2" in o:
            return o
        for v in o.values():
            r = walk(v, depth + 1) if depth < 6 else None
            if r is not None:
                return r
    elif isinstance(o, list):
        for v in o:
            r = walk(v, depth + 1) if depth < 6 else None
            if r is not None:
                return r
    return None


rd = walk(d)
if not rd:
    print("runData not found via walk")
    sys.exit()
ROOT = d if isinstance(d, list) else []


def res(o, depth=0):
    if isinstance(o, str) and o.isdigit() and depth < 8:
        return res(ROOT[int(o)], depth + 1)
    if isinstance(o, dict):
        return {k: res(v, depth + 1) for k, v in o.items()}
    if isinstance(o, list):
        return [res(v, depth + 1) for v in o]
    return o


print("nodes in runData:", len(rd))
for name in ("AI Agent", "AI Agent1", "Log Live Voice",
             "OpenRouter Chat Model", "Debate Round 2"):
    e = res(rd.get(name, []))
    outs = []
    for ex in (e if isinstance(e, list) else [e]):
        data = (ex.get("data") or {}).get("main", [[[]]])[0]
        outs.append(data[0].get("json", {}) if data and isinstance(data[0], dict) else data)
    first = outs[0]
    if isinstance(first, dict):
        print(f"{name}: ran, keys={list(first.keys())[:6]}",
              ("output=" + str(first.get("output"))[:130]) if "output" in first else "")
    else:
        print(f"{name}: ran, items={len(outs)}")
