#!/usr/bin/env python3
"""Voice lane: parallel burst -> sequential loop (free-tier friendly)."""
import json
import sqlite3
import sys
import uuid

sys.stdout.reconfigure(encoding="utf-8")
db = sqlite3.connect(r"C:\Users\Admin\.n8n\database.sqlite")
row = db.execute(
    "SELECT nodes, connections FROM workflow_entity WHERE id=?",
    ("EmfoGhyIzSrktmIe",)).fetchone()
nodes = {n["name"]: n for n in json.loads(row[0])}
conns = json.loads(row[1])

bx, by = nodes["Build Voice Requests"]["position"]
split = {"parameters": {"batchSize": 1, "options": {}},
         "id": str(uuid.uuid4()), "name": "Batch Voices",
         "type": "n8n-nodes-base.splitInBatches", "typeVersion": 3,
         "position": [bx + 220, by]}
nodes["Batch Voices"] = split

conns["Build Voice Requests"] = {"main": [[{"node": "Batch Voices",
                                            "type": "main", "index": 0}]]}
conns["Batch Voices"] = {"main": [[], [{"node": "Call OpenRouter",
                                         "type": "main", "index": 0}]]}
conns["Log Live Voices"] = {"main": [[{"node": "Batch Voices",
                                       "type": "main", "index": 0}]]}
print("loop wired: build -> batch -[loop]-> call -> shape -> log -> batch")

P = r"C:\02_QUILLAN\08 - Templates & Config\workflows\emfo-main.json"
open(P, "w", encoding="utf-8").write(json.dumps(
    {"id": "EmfoGhyIzSrktmIe", "name": "Quillan Sovereign \u2014 MAIN",
     "nodes": list(nodes.values()), "connections": conns,
     "settings": {"executionOrder": "v1"}, "staticData": None, "pinData": {}},
    indent=2))
print("saved", len(nodes), "nodes")
