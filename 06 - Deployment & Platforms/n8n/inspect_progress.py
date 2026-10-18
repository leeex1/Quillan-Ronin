#!/usr/bin/env python3
"""Progress of the running execution: which shells finished, any LLM text?"""
import json
import sqlite3
import sys

sys.stdout.reconfigure(encoding="utf-8")
db = sqlite3.connect(r"C:\Users\Admin\.n8n\database.sqlite")
row = db.execute(
    "SELECT id FROM execution_entity WHERE status='running' "
    "ORDER BY startedAt DESC LIMIT 1").fetchone()
if not row:
    print("no running execution")
    sys.exit()
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


rd = res(walk(d)) or {}
print("nodes completed so far:", len(rd))
shells = [k for k in rd if k == "AI Agent"
          or (k.startswith("AI Agent") and k[8:].isdigit())
          or k == "C0-QUILLAN \u00b7LIVE"]
print("shells completed:", len(shells))
for name in sorted(shells)[:6]:
    e = rd[name][0] if isinstance(rd[name], list) else rd[name]
    if isinstance(e, dict) and e.get("error"):
        print(name, "ERROR:", str(e["error"].get("message"))[:90])
        break
    try:
        items = (e.get("data") or {}).get("main", [[[]]])[0] or []
        j = items[0].get("json", {}) if items else {}
        out = str(j.get("output", ""))[:100]
        print(name, "->", out if out else f"keys={list(j.keys())[:5]}")
    except Exception as ex:
        print(name, "parse issue", str(ex)[:80])
