#!/usr/bin/env python3
"""Add swarm emit + kill-check fanout to the live canvas council tail.

S34 -> Emit Swarm -> Debate  (was S34 -> Debate)
Throne -> Emit Kill-Checks (1->4 items) -> Post Kill-Checks -> Recollect Throne -> Emit Arbitration
Backup first.
"""
import json
import shutil
import sys
import uuid

sys.stdout.reconfigure(encoding="utf-8")
P = r"C:\02_QUILLAN\08 - Templates & Config\workflows\starter odea workflow.json"
shutil.copy(P, P + ".bak-killchecks")
print("backup written")

wf = json.load(open(P, encoding="utf-8"))
nodes = {n["name"]: n for n in wf["nodes"]}
conns = wf["connections"]


def nid():
    return str(uuid.uuid4())


def http_node(name, body_expr, pos):
    return {"parameters": {"method": "POST", "url": "http://127.0.0.1:8777/api/events",
                           "sendBody": True, "specifyBody": "json", "jsonBody": body_expr,
                           "options": {"timeout": 3000}},
            "id": nid(), "name": name, "type": "n8n-nodes-base.httpRequest",
            "typeVersion": 4.2, "position": pos, "onError": "continueRegularOutput"}


KILL_JS = """const s = $input.first().json;
await new Promise((r) => setTimeout(r, 200));
const risk = String(s.query).match(/hack|harm|attack|bypass|illegal|weapon|exploit|steal/gi) || [];
const uniq = [...new Set(risk.map((w) => w.toLowerCase()))];
const items = [
  { agent: 'council-34', type: 'stage', stage: 'predator', status: s.gates.logic.pass ? 'pass' : 'fail',
    detail: 'C34-PREDATOR kill-check: weakest ' + s.debate.weakest + (s.gates.logic.pass ? ' survived' : ' killed') },
  { agent: 'council-34', type: 'stage', stage: 'vir', status: s.gates.ethics.pass ? 'pass' : 'fail',
    detail: 'C2-VIR ethical veto: ' + (s.gates.ethics.pass ? 'clear' : 'VETO engaged') },
  { agent: 'council-34', type: 'stage', stage: 'warden', status: 'pass',
    detail: 'C13-WARDEN threat scan: ' + (uniq.length ? 'flags [' + uniq.join(', ') + ']' : 'clean') },
  { agent: 'council-34', type: 'stage', stage: 'vigil', status: s.confidence >= 0.6 ? 'pass' : 'fail',
    detail: 'C19-VIGIL drift anchor: consistency ' + s.confidence }
];
return items.map((e) => ({ json: e }));"""

RECOLLECT_JS = """return [{ json: $('C0-QUILLAN Throne').first().json }];"""

# --- 1. swarm emit between S34 and Debate ---
s34 = "C34-PREDATOR-swarm"
assert s34 in nodes and "Debate Round 2" in nodes
dpos = nodes["Debate Round 2"]["position"]
spos = nodes[s34]["position"]
mx, my = (spos[0] + dpos[0]) // 2, (spos[1] + dpos[1]) // 2
nodes["Emit Swarm"] = http_node(
    "Emit Swarm",
    "={{ {'agent': 'council-34', 'type': 'stage', 'stage': 'swarm', 'status': 'pass', "
    "'detail': 'swarm accumulation complete: ' + $json.verdicts.length + ' verdicts'} }}",
    [mx, my])
wf["nodes"].append(nodes["Emit Swarm"])
conns[s34] = {"main": [[{"node": "Emit Swarm", "type": "main", "index": 0}]]}
conns["Emit Swarm"] = {"main": [[{"node": "Debate Round 2", "type": "main", "index": 0}]]}
print("swarm emit inserted")

# --- 2. kill-check fanout after throne ---
tpos = nodes["C0-QUILLAN Throne"]["position"]
epos = nodes["Emit Arbitration"]["position"]
kx = tpos[0] + 220
kill = {"parameters": {"mode": "runOnceForAllItems", "jsCode": KILL_JS},
        "id": nid(), "name": "Emit Kill-Checks",
        "type": "n8n-nodes-base.code", "typeVersion": 2, "position": [kx, tpos[1]]}
post = http_node("Post Kill-Checks", "={{ $json }}", [kx + 220, tpos[1]])
recol = {"parameters": {"mode": "runOnceForAllItems", "jsCode": RECOLLECT_JS},
         "id": nid(), "name": "Recollect Throne",
         "type": "n8n-nodes-base.code", "typeVersion": 2, "position": [kx + 440, tpos[1]]}
for nd in (kill, post, recol):
    nodes[nd["name"]] = nd
    wf["nodes"].append(nd)
nodes["Emit Arbitration"]["position"] = [kx + 660, tpos[1]]
conns["C0-QUILLAN Throne"] = {"main": [[{"node": "Emit Kill-Checks", "type": "main", "index": 0}]]}
conns["Emit Kill-Checks"] = {"main": [[{"node": "Post Kill-Checks", "type": "main", "index": 0}]]}
conns["Post Kill-Checks"] = {"main": [[{"node": "Recollect Throne", "type": "main", "index": 0}]]}
conns["Recollect Throne"] = {"main": [[{"node": "Emit Arbitration", "type": "main", "index": 0}]]}
print("kill-check fanout wired")
print("NOTE: old Throne->Emit Arbitration link replaced; epos was", epos)

json.dump(wf, open(P, "w", encoding="utf-8"), indent=2)
print("saved")
