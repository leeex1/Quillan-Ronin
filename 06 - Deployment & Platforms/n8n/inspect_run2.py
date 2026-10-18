#!/usr/bin/env python3
"""Shell outputs in latest run (ref-resolving)."""
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
ROOT = d if isinstance(d, list) else []


def res(o, depth=0):
    if isinstance(o, str) and o.isdigit() and depth < 10:
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
        if "AI Agent" in o and "Debate Round 2" in o:
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


rd = res(walk(d))
names = ["AI Agent", "AI Agent1", "AI Agent33", "C0-QUILLAN \u00b7LIVE",
         "Log Live Voice", "Debate Round 2"]
for name in names:
    e = rd.get(name)
    if not e:
        print(f"{name}: NOT RAN")
        continue
    ex = e[0] if isinstance(e, list) else e
    if isinstance(ex, dict) and ex.get("error"):
        print(f"{name}: ERROR:", str(ex["error"].get("message"))[:120])
        continue
    try:
        items = (ex.get("data") or {}).get("main", [[]])[0] or []
        j = items[0].get("json", {}) if items else {}
        if "output" in j:
            print(f"{name}: LLM TEXT: {str(j['output'])[:150]}")
        else:
            print(f"{name}: keys={list(j.keys())[:6]}")
    except Exception as ex2:
        print(f"{name}: parse issue {str(ex2)[:100]}")
