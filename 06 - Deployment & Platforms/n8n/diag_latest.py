#!/usr/bin/env python3
"""Diagnose the LATEST execution error precisely."""
import json
import sqlite3
import sys

sys.stdout.reconfigure(encoding="utf-8")
db = sqlite3.connect(r"C:\Users\Admin\.n8n\database.sqlite")
erow = db.execute(
    "SELECT id, startedAt, status FROM execution_entity "
    "ORDER BY startedAt DESC LIMIT 1").fetchone()
print("latest:", erow[1], erow[2])
d = json.loads(db.execute(
    "SELECT data FROM execution_data WHERE executionId=?",
    (erow[0],)).fetchone()[0])
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
        if "lastNodeExecuted" in o and "error" in o:
            return o
        if depth < 10:
            for v in o.values():
                r = walk(v, depth + 1)
                if r is not None:
                    return r
    elif isinstance(o, list) and depth < 10:
        for v in o:
            r = walk(v, depth + 1)
            if r is not None:
                return r
    return None


info = res(walk(d)) or {}
print("lastNode:", info.get("lastNodeExecuted"))
err = res(info.get("error", {}))
print("err name:", err.get("name"))
print("err msg:", str(err.get("message"))[:300])
node = res(err.get("node", {}))
print("node:", node.get("name") if isinstance(node, dict) else node)
