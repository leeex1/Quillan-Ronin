#!/usr/bin/env python3
"""Unify the merged canvas: make the council path runnable, park dead shells.
Backup first. Idempotent-ish. Run once."""
import json
import shutil
import sys

sys.stdout.reconfigure(encoding="utf-8")
P = r"C:\02_QUILLAN\08 - Templates & Config\workflows\starter odea workflow.json"
shutil.copy(P, P + ".bak-unify")
print("backup written")

wf = json.load(open(P, encoding="utf-8"))
wf["name"] = "Quillan Sovereign (unified canvas)"
nodes = {n["name"]: n for n in wf["nodes"]}
conns = wf["connections"]

TRIG = "When clicking \u2018Execute workflow\u2019"
assert TRIG in nodes, "trigger not found"

# 1. Trigger -> Query Input (the council chain currently has no live trigger)
conns[TRIG] = {"main": [[{"node": "Query Input", "type": "main", "index": 0}]]}
print("trigger wired -> Query Input")

# 2. Park dead shells: file node loses all targets; shells lose Aggregate link
dead = [k for k in conns if k.startswith("AI Agent")]
file_node = "Read/Write Files from Disk"
if file_node in conns:
    conns[file_node] = {"main": [[]]}
    print("parked:", file_node)
for k in dead:
    conns[k] = {"main": [[]]}
print("shells unplugged:", len(dead))

# 3. Prepare must read Parse output (it sits behind Emit Intake's HTTP response)
prep = nodes["Prepare 34 Assignments"]
js = prep["parameters"]["jsCode"]
old = "$input.first().json"
assert old in js
prep["parameters"]["jsCode"] = js.replace(
    old, "$('Parse Identities JSON').first().json", 1)
print("prepare source fixed")

# 4. Sticky TODO over the parked export leg
agg = nodes.get("Aggregate", {})
ax, ay = agg.get("position", [3000, 600])
wf["nodes"].append({
    "parameters": {"content": (
        "# TODO export leg (parked, safe)\n\n"
        "Aggregate + Python + File nodes are INCOMPLETE - left unwired on purpose so runs stay green.\n\n"
        "To finish: complete the Python shaping, then connect C0-QUILLAN Throne -> Aggregate.\n"
        "Data files for n8n must live under C:/Users/Admin/.n8n-files/ (server allowlist)."
    ), "height": 200, "width": 420, "color": 4},
    "id": "todo-export-leg", "name": "Sticky TODO export",
    "type": "n8n-nodes-base.stickyNote", "typeVersion": 1,
    "position": [ax - 460, ay - 40]})

json.dump(wf, open(P, "w", encoding="utf-8"), indent=2)
print("saved unified canvas")
