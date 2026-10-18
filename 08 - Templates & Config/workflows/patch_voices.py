#!/usr/bin/env python3
"""HTTP live voices: 4 nodes, parallel, no sub-node mystery.

Reads MAIN fresh from DB, preserves everything, adds voice lane + parks shells.
Also builds the HttpHeaderAuth credential import file (deleted after import).
"""
import json
import re
import shutil
import sqlite3
import sys
import uuid

sys.stdout.reconfigure(encoding="utf-8")
OR_TXT = open(r"C:\Users\Admin\AppData\Local\hermes\.env", encoding="utf-8",
              errors="ignore").read()
OR_KEY = re.search(r"OPENROUTER_API_KEY\s*[:=]\s*['\"]?([^\s'\";]+)",
                   OR_TXT).group(1)
MODEL = open(r"C:\02_QUILLAN\06 - Deployment & Platforms\n8n\working_model.txt",
             encoding="utf-8").read().strip()
IDENT = json.load(open(
    r"C:\02_QUILLAN\08 - Templates & Config\workflows\council-identities.json",
    encoding="utf-8"))

# 1. credential import file
cid = str(uuid.uuid4())
creds = [{"id": cid, "name": "OpenRouter Bearer", "type": "httpHeaderAuth",
          "data": {"name": "Authorization", "value": f"Bearer {OR_KEY}"}}]
open(r"C:\02_QUILLAN\06 - Deployment & Platforms\n8n\creds-http.json",
     "w", encoding="utf-8").write(json.dumps(creds))
print("cred file built, id:", cid[:8])

# 2. patch MAIN
db = sqlite3.connect(r"C:\Users\Admin\.n8n\database.sqlite")
row = db.execute(
    "SELECT nodes, connections FROM workflow_entity WHERE id=?",
    ("EmfoGhyIzSrktmIe",)).fetchone()
nodes = {n["name"]: n for n in json.loads(row[0])}
conns = json.loads(row[1])


def nid():
    return str(uuid.uuid4())


BUILD_JS = """const d = $input.first().json;
await new Promise((r) => setTimeout(r, 300));
const personas = __PERSONAS__;
const q = d.query, agree = d.debate.agreement, veto = d.debate.veto;
return personas.map((p) => ({ json: {
  cid: p.id + '-' + p.name,
  model: '__MODEL__',
  system: 'You are ' + p.id + '-' + p.name + ', ' + p.role + ' on the Quillan-Ronin council. ' + p.prompt,
  user: 'Council query: ' + q + '. Consensus ' + agree + ', veto=' + veto + '. Give YOUR verdict in 2 sentences. Begin with your ID.'
}}));""".replace("__PERSONAS__", json.dumps(
    [{"id": p["id"], "name": p["name"], "role": p["role"],
      "prompt": p["system_prompt"]} for p in IDENT["personas"]])).replace(
    "__MODEL__", MODEL)

SHAPE_JS = """const items = $input.all().map((i) => i.json);
return items.map((r) => {
  try {
    const txt = (((r.choices || [])[0] || {}).message || {}).content || '';
    return { json: { agent: 'council-live', type: 'log',
                     detail: (r.cid || 'voice') + ': ' + String(txt).slice(0, 600) } };
  } catch (e) {
    return { json: { agent: 'council-live', type: 'log',
                     detail: (r.cid || 'voice') + ': (no answer this run)' } };
  }
});"""

dpos = nodes["Debate Round 2"]["position"]
bx, by = dpos[0], dpos[1] + 260
build = {"parameters": {"mode": "runOnceForAllItems", "jsCode": BUILD_JS},
         "id": nid(), "name": "Build Voice Requests",
         "type": "n8n-nodes-base.code", "typeVersion": 2,
         "position": [bx, by]}
call = {"parameters": {
    "authentication": "genericAuth", "genericAuthType": "httpHeaderAuth",
    "method": "POST", "url": "https://openrouter.ai/api/v1/chat/completions",
    "sendBody": True, "specifyBody": "json",
    "jsonBody": ("={{ {model: $json.model, messages: "
                 "[{role: 'system', content: $json.system}, "
                 "{role: 'user', content: $json.user}], max_tokens: 120} }}"),
    "options": {"timeout": 90000}},
    "id": nid(), "name": "Call OpenRouter",
    "type": "n8n-nodes-base.httpRequest", "typeVersion": 4.2,
    "position": [bx + 220, by], "onError": "continueRegularOutput",
    "credentials": {"httpHeaderAuth": {"id": cid,
                                       "name": "OpenRouter Bearer"}}}
shape = {"parameters": {"mode": "runOnceForAllItems", "jsCode": SHAPE_JS},
         "id": nid(), "name": "Shape Voice Events",
         "type": "n8n-nodes-base.code", "typeVersion": 2,
         "position": [bx + 440, by]}
logv = {"parameters": {
    "method": "POST", "url": "http://127.0.0.1:8777/api/events",
    "sendBody": True, "specifyBody": "json", "jsonBody": "={{ $json }}",
    "options": {"timeout": 10000}},
    "id": nid(), "name": "Log Live Voices",
    "type": "n8n-nodes-base.httpRequest", "typeVersion": 4.2,
    "position": [bx + 660, by], "onError": "continueRegularOutput"}
for nd in (build, call, shape, logv):
    nodes[nd["name"]] = nd

# Debate fans to Emit Council (kept) + voice lane
dout = conns.get("Debate Round 2", {}).get("main", [[[]]])[0]
if not any(t.get("node") == "Build Voice Requests" for t in dout):
    dout.append({"node": "Build Voice Requests", "type": "main", "index": 0})
    conns["Debate Round 2"] = {"main": [dout]}
conns["Build Voice Requests"] = {"main": [[{"node": "Call OpenRouter",
                                            "type": "main", "index": 0}]]}
conns["Call OpenRouter"] = {"main": [[{"node": "Shape Voice Events",
                                       "type": "main", "index": 0}]]}
conns["Shape Voice Events"] = {"main": [[{"node": "Log Live Voices",
                                          "type": "main", "index": 0}]]}
print("voice lane wired")

# park the shell chain (shells keep brains for future UI wiring)
PASTED = ["AI Agent"] + [f"AI Agent{i}" for i in range(1, 34)] + ["C0-QUILLAN \u00b7LIVE"]
for sname in PASTED + ["OpenRouter Chat Model", "Log Live Voice"]:
    if sname in conns:
        conns[sname] = {"main": [[]]}
# Debate no longer feeds the old shell chain head
dout2 = conns.get("Debate Round 2", {}).get("main", [[[]]])[0]
conns["Debate Round 2"] = {"main": [[t for t in dout2
                                      if t.get("node") not in PASTED]]}
print("old shell chain parked")

P = r"C:\02_QUILLAN\08 - Templates & Config\workflows\emfo-live.json"
open(P, "w", encoding="utf-8").write(json.dumps(
    {"id": "EmfoGhyIzSrktmIe", "name": "Quillan Sovereign \u2014 MAIN",
     "nodes": list(nodes.values()), "connections": conns,
     "settings": {"executionOrder": "v1"}, "staticData": None, "pinData": {}},
    indent=2))
print("saved", len(nodes), "nodes")
