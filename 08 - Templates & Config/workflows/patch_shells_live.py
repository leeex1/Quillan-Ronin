#!/usr/bin/env python3
"""Wire the 34 parked shells + new C0 shell as LIVE sequential voices.

Reads Emfo fresh from DB (positions/layout preserved), writes emfo-main.json.
- Each shell: identity systemMessage + prompt via $('Debate Round 2') refs.
- Chain: Debate -> Sh1 -> Sh2 ... -> Sh34 -> ShC0 -> (also each -> Log Voices).
- Shared OpenRouter model (nemotron free) + shared log node, onError continue.
- Throne/Debate/council logic untouched.
"""
import json
import shutil
import sqlite3
import sys
import uuid

sys.stdout.reconfigure(encoding="utf-8")
OR_CRED = "e6a2a4ff-6f9f-4c66-a817-221922428a98"
MODEL = open(r"C:\02_QUILLAN\06 - Deployment & Platforms\n8n\working_model.txt",
             encoding="utf-8").read().strip()
IDENT = json.load(open(
    r"C:\02_QUILLAN\08 - Templates & Config\workflows\council-identities.json",
    encoding="utf-8"))
PERSONAS = IDENT["personas"]  # C1..C34 order
assert len(PERSONAS) == 34
THRONE = IDENT["throne"]

db = sqlite3.connect(r"C:\Users\Admin\.n8n\database.sqlite")
row = db.execute(
    "SELECT nodes, connections FROM workflow_entity WHERE id=?",
    ("EmfoGhyIzSrktmIe",)).fetchone()
nodes = {n["name"]: n for n in json.loads(row[0])}
conns = json.loads(row[1])
print("loaded Emfo from DB:", len(nodes), "nodes")

# pasted shell order maps to council order C1..C34
PASTED = ["AI Agent"] + [f"AI Agent{i}" for i in range(1, 34)]
for nm in PASTED:
    assert nm in nodes, f"shell missing: {nm}"

for idx, (shell, p) in enumerate(zip(PASTED, PERSONAS)):
    nodes[shell]["parameters"] = {
        "promptType": "define",
        "text": ("={{ 'Council query: ' + $('Debate Round 2').first().json.query "
                 "+ '. Consensus ' + $('Debate Round 2').first().json.debate.agreement "
                 "+ ', veto=' + $('Debate Round 2').first().json.debate.veto "
                 "+ '. Give YOUR verdict in 2 sentences. Begin with your ID.' }}"),
        "options": {
            "systemMessage": (
                f"You are {p['id']}-{p['name']}, {p['role']} on the "
                f"Quillan-Ronin council. {p['system_prompt']}"),
            "maxIterations": 2},
    }
    nodes[shell]["onError"] = "continueRegularOutput"
print("34 shells given brains")


def nid():
    return str(uuid.uuid4())


# C0 live voice (Quillan himself)
c34 = dict(nodes[PASTED[-1]])
c0pos = [c34["position"][0], c34["position"][1] + 220]
c0 = {"parameters": {
    "promptType": "define",
    "text": ("={{ 'Council query: ' + $('Debate Round 2').first().json.query "
             "+ '. You are the throne. Speak your synthesis in 2 sentences.' }}"),
    "options": {
        "systemMessage": f"You are {THRONE['id']}-{THRONE['name']}, {THRONE['role']}. {THRONE['system_prompt']}",
        "maxIterations": 2},
    "onError": "continueRegularOutput",
}, "id": nid(), "name": "C0-QUILLAN ·LIVE",
    "type": "@n8n/n8n-nodes-langchain.agent", "typeVersion": 3.1,
    "position": c0pos}
nodes[c0["name"]] = c0

# shared model + shared log voice node near the shells' bbox
xs = [nodes[s]["position"][0] for s in PASTED]
ys = [nodes[s]["position"][1] for s in PASTED]
mx, my = (min(xs) + max(xs)) // 2, max(ys) + 260
model = {"parameters": {"model": MODEL, "options": {}},
         "id": nid(), "name": "OpenRouter Chat Model",
         "type": "@n8n/n8n-nodes-langchain.lmChatOpenRouter",
         "typeVersion": 1, "position": [mx - 140, my],
         "credentials": {"openRouterApi": {"id": OR_CRED, "name": "OpenRouter"}}}
log = {"parameters": {"method": "POST",
                      "url": "http://127.0.0.1:8777/api/events",
                      "sendBody": True, "specifyBody": "json",
                      "jsonBody": ("={{ {'agent': 'council-live', 'type': 'log', "
                                   "'detail': $json.output} }}"),
                      "options": {"timeout": 10000}},
       "id": nid(), "name": "Log Live Voice",
       "type": "n8n-nodes-base.httpRequest", "typeVersion": 4.2,
       "position": [mx + 140, my], "onError": "continueRegularOutput"}
nodes[model["name"]] = model
nodes[log["name"]] = log
print("model + log nodes added:", MODEL)

# chain: Debate -> Sh1 -> Sh2 ... -> Sh34 -> C0 ; each -> Log
order = PASTED + [c0["name"]]
dout = conns.get("Debate Round 2", {}).get("main", [[[]]])[0]
if not any(t.get("node") == order[0] for t in dout):
    dout.append({"node": order[0], "type": "main", "index": 0})
    conns["Debate Round 2"] = {"main": [dout]}
for a, b in zip(order, order[1:]):
    # single main output fanning to BOTH next-in-chain and the log node
    conns[a] = {"main": [[{"node": b, "type": "main", "index": 0},
                           {"node": log["name"], "type": "main", "index": 0}]]}
conns[order[-1]] = {"main": [[{"node": log["name"], "type": "main", "index": 0}]]}
# shared model link for every shell (keeps the mains set above)
for sname in order:
    conns[sname]["ai_languageModel"] = [[{"node": model["name"],
                                          "type": "ai_languageModel", "index": 0}]]
print("chain + model links wired")

out_nodes = list(nodes.values())
P = r"C:\02_QUILLAN\08 - Templates & Config\workflows\emfo-main.json"
shutil.copy(P, P + ".bak-shells")
open(P, "w", encoding="utf-8").write(json.dumps(
    {"id": "EmfoGhyIzSrktmIe", "name": "Quillan Sovereign \u2014 MAIN",
     "nodes": out_nodes, "connections": conns,
     "settings": {"executionOrder": "v1"}, "staticData": None, "pinData": {}},
    indent=2))
print("saved", len(out_nodes), "nodes")
