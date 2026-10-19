#!/usr/bin/env python3
"""Dump Gate Logic output item keys (ref-tolerant)."""
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
    if isinstance(o, str) and o.isdigit() and depth < 14:
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
        if "Gate Logic C7" in o:
            return o
        if depth < 9:
            for v in o.values():
                r = walk(v, depth + 1)
                if r is not None:
                    return r
    elif isinstance(o, list) and depth < 9:
        for v in o:
            r = walk(v, depth + 1)
            if r is not None:
                return r
    return None


rd = res(walk(d)) or {}
e = rd.get("Gate Logic C7")
print("entry type:", type(e).__name__)
if isinstance(e, list):
    print("execs:", len(e))
    e = e[0]
if isinstance(e, dict):
    data = e.get("data") or {}
    mains = data.get("main", [])
    print("outputs:", len(mains))
    if mains and mains[0]:
        it = mains[0][0]
        print("item type:", type(it).__name__)
        if isinstance(it, dict):
            print("item keys:", list(it.keys()))
            j = it.get("json", {})
            print("json keys:", list(j.keys()) if isinstance(j, dict) else type(j))
