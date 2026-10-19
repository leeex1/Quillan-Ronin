#!/usr/bin/env python3
"""Was the voice lane in the latest run?"""
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
        if "Build Voice Requests" in o or "Debate Round 2" in o:
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
for name in ("Debate Round 2", "Build Voice Requests", "Batch Voices",
             "Call OpenRouter", "Shape Voice Events", "Log Live Voices"):
    print(("RAN " if name in rd else "SKIP ") + name)
