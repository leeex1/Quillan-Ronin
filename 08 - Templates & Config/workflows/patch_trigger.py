#!/usr/bin/env python3
"""Restore the missing manual trigger node (canvas can't draw wires from ghosts)."""
import json
import sys
import uuid

sys.stdout.reconfigure(encoding="utf-8")
P = r"C:\02_QUILLAN\08 - Templates & Config\workflows\emfo-main.json"
wf = json.load(open(P, encoding="utf-8"))
w = wf[0] if isinstance(wf, list) else wf
nodes = {n["name"]: n for n in w["nodes"]}
TRIG = "When clicking \u2018Execute workflow\u2019"
if TRIG not in nodes:
    qx, qy = nodes["Query Input"]["position"]
    trig = {"parameters": {}, "id": str(uuid.uuid4()), "name": TRIG,
            "type": "n8n-nodes-base.manualTrigger", "typeVersion": 1,
            "position": [qx - 224, qy]}
    w["nodes"].append(trig)
    print("trigger restored at", trig["position"])
else:
    print("trigger already present")
conns = w["connections"]
assert conns.get(TRIG, {}).get("main", [[[]]])[0][0].get("node") == "Query Input", \
    "trigger link wrong: " + str(conns.get(TRIG))
print("trigger -> Query Input confirmed")
open(P, "w", encoding="utf-8").write(
    json.dumps(w if not isinstance(wf, list) else [w], indent=2))
print("saved", len(w["nodes"]), "nodes")
