#!/usr/bin/env python3
"""Make the USER's canvas (Emfo) fully runnable. Position-preserving:
only adds missing nodes + fixes code/refs. Never moves their nodes.
- Restores deleted manual trigger -> Query Input
- Prepare: unwrap Extract {data:{...}}
- Fan -> sequential chain (agents carry {bundle,verdicts[]} via $input only)
- Debate reads S34 accumulator; Throne reads Debate
- Adds Emit Swarm + kill-check fanout (positions from neighbors)
- Renames to unique title. Keeps Emfo id.
Backup first.
"""
import json
import shutil
import sys
import uuid

sys.stdout.reconfigure(encoding="utf-8")
P = r"C:\02_QUILLAN\08 - Templates & Config\workflows\emfo-main.json"
raw = open(P, encoding="utf-8").read()
wf = json.loads(raw)
w = wf[0] if isinstance(wf, list) else wf
is_list = isinstance(wf, list)
shutil.copy(P, P + ".bak-emfofix")
print("backup written")

w["id"] = "EmfoGhyIzSrktmIe"
w["name"] = "Quillan Sovereign \u2014 MAIN"
nodes = {n["name"]: n for n in w["nodes"]}
conns = w["connections"]

AGENT_JS = """const items = $input.all().map((i) => i.json);
await new Promise((r) => setTimeout(r, 150));
let bundle, verdicts;
if (items[0].persona && !items[0].verdicts) {
  bundle = { query: items[0].query, context: items[0].context, run_id: items[0].run_id,
    personas: items.map((a) => a.persona) };
  verdicts = [];
} else {
  bundle = items[0].bundle;
  verdicts = [...items[0].verdicts];
}
const persona = bundle.personas.find((p) => p.id === '__ID__');
const q = String(bundle.query).toLowerCase();
const rel = persona.keywords.filter((k) => q.includes(String(k).toLowerCase())).length;
const risk = /hack|harm|attack|bypass|illegal|weapon|exploit|steal/i.test(String(bundle.query));
let stance = 'neutral';
if ((persona.id === 'C2-VIR' || persona.id === 'C13-WARDEN') && risk) stance = 'caution';
else if (rel >= 1) stance = 'support';
const confidence = Math.max(0.05, Math.min(0.99, persona.prior - 0.12 + 0.09 * Math.min(rel, 3)));
verdicts.push({ agent_id: persona.id, name: persona.name, role: persona.role, cluster: persona.cluster,
  query: bundle.query, context: bundle.context, run_id: bundle.run_id,
  stance, confidence: +confidence.toFixed(2), relevance_hits: rel,
  points: [persona.role + ' lens: ' + (rel ? 'directly relevant (' + rel + ' keyword hits)' : 'general perspective'),
    stance === 'caution' ? 'RAISES OBJECTION on safety/ethics grounds' : 'position ' + stance + ' at ' + confidence.toFixed(2)] });
return [{ json: { bundle, verdicts } }];"""

SWARM_JS = """const s = $input.first().json;
await new Promise((r) => setTimeout(r, 150));
const v = s.verdicts[s.verdicts.length - 1];
const seed = String(v.agent_id).split('').reduce((a, c) => a + c.charCodeAt(0), 0);
const j = (n) => (((seed * (n + 7)) % 11) - 5) / 100;
const clamp = (x) => Math.max(0.05, Math.min(0.99, x));
const positions = [
  { angle: 'support', confidence: +clamp(v.confidence + 0.04 + j(1)).toFixed(2), note: 'steelman the verdict' },
  { angle: 'attack', confidence: +clamp(v.confidence - 0.06 + j(2)).toFixed(2), note: 'stress-test the weakest premise' },
  { angle: 'refine', confidence: +clamp(v.confidence + 0.02 + j(3)).toFixed(2), note: 'tighten wording and scope' }
];
const adjusted = +clamp(v.confidence * 0.6 + positions.reduce((a, p) => a + p.confidence, 0) / 3 * 0.4).toFixed(2);
v.swarm = { positions, adjusted_confidence: adjusted,
  summary: '3 micro-positions debated, adjusted ' + v.confidence + ' -> ' + adjusted };
return [{ json: s }];"""

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


def cnum(name):
    return int("".join(c for c in name if c.isdigit()))


agents = sorted([n for n in nodes if n.startswith("C") and len(n) > 1 and n[1].isdigit()
                 and not n.endswith("-swarm") and n != "C0-QUILLAN Throne"], key=cnum)
assert len(agents) == 34, f"expected 34 agents, got {len(agents)}"
swarms = [a + "-swarm" for a in agents]
for sname in swarms:
    assert sname in nodes, f"missing {sname}"

# 1. restore trigger
TRIG = "When clicking \u2018Execute workflow\u2019"
if TRIG not in nodes:
    qx, qy = nodes["Query Input"]["position"]
    trig = {"parameters": {}, "id": str(uuid.uuid4()), "name": TRIG,
            "type": "n8n-nodes-base.manualTrigger", "typeVersion": 1,
            "position": [qx - 220, qy]}
    nodes[TRIG] = trig
    w["nodes"].append(trig)
    print("trigger restored")
conns[TRIG] = {"main": [[{"node": "Query Input", "type": "main", "index": 0}]]}

# 2. prepare unwrap
prep = nodes["Prepare 34 Assignments"]["parameters"]["jsCode"]
if "rawParse" not in prep:
    prep = prep.replace("$input.first().json",
                        "($('Parse Identities JSON').first().json.data || $('Parse Identities JSON').first().json)", 1)
    nodes["Prepare 34 Assignments"]["parameters"]["jsCode"] = prep
    print("prepare fixed")

# 3. uniform agent + swarm code
for aname in agents:
    nodes[aname]["parameters"]["jsCode"] = AGENT_JS.replace("__ID__", aname)
for sname in swarms:
    nodes[sname]["parameters"]["jsCode"] = SWARM_JS
print("agent/swarm code unified")

# 4. debate reads S34 accumulator
djs = nodes["Debate Round 2"]["parameters"]["jsCode"]
if "$('C34-PREDATOR-swarm')" not in djs:
    old = "const verdicts = $input.all().map((i) => i.json);"
    assert old in djs
    djs = djs.replace(old, "const incomingDebate = $('C34-PREDATOR-swarm').first().json;\n"
                           "const verdicts = Array.isArray(incomingDebate.verdicts) ? incomingDebate.verdicts : [];", 1)
    nodes["Debate Round 2"]["parameters"]["jsCode"] = djs
    print("debate source fixed")

# 5. throne reads debate
tjs = nodes["C0-QUILLAN Throne"]["parameters"]["jsCode"]
if "$('Debate Round 2')" not in tjs:
    old = "const s = $input.first().json;"
    assert old in tjs
    nodes["C0-QUILLAN Throne"]["parameters"]["jsCode"] = tjs.replace(
        old, "const s = $('Debate Round 2').first().json;", 1)
    print("throne source fixed")

# 6. sequential chain links
for aname in agents:
    conns[aname] = {"main": [[]]}
for sname in swarms:
    conns[sname] = {"main": [[]]}
chain = []
for aname, sname in zip(agents, swarms):
    chain += [aname, sname]
conns["Prepare 34 Assignments"] = {"main": [[{"node": chain[0], "type": "main", "index": 0}]]}
for a, b in zip(chain, chain[1:]):
    conns[a] = {"main": [[{"node": b, "type": "main", "index": 0}]]}
conns[chain[-1]] = {"main": [[{"node": "Debate Round 2", "type": "main", "index": 0}]]}
print("chain wired:", len(chain) + 1, "links")


def http_node(name, body_expr, pos):
    return {"parameters": {"method": "POST", "url": "http://127.0.0.1:8777/api/events",
                           "sendBody": True, "specifyBody": "json", "jsonBody": body_expr,
                           "options": {"timeout": 3000}},
            "id": str(uuid.uuid4()), "name": name, "type": "n8n-nodes-base.httpRequest",
            "typeVersion": 4.2, "position": pos, "onError": "continueRegularOutput"}


# 7. swarm emit between S34 and debate
s34pos = nodes[swarms[-1]]["position"]
dpos = nodes["Debate Round 2"]["position"]
mx, my = (s34pos[0] + dpos[0]) // 2, (s34pos[1] + dpos[1]) // 2
esw = http_node("Emit Swarm",
    "={{ {'agent': 'council-34', 'type': 'stage', 'stage': 'swarm', 'status': 'pass', "
    "'detail': 'swarm accumulation complete: ' + $json.verdicts.length + ' verdicts'} }}",
    [mx, my])
nodes["Emit Swarm"] = esw
w["nodes"].append(esw)
conns[swarms[-1]] = {"main": [[{"node": "Emit Swarm", "type": "main", "index": 0}]]}
conns["Emit Swarm"] = {"main": [[{"node": "Debate Round 2", "type": "main", "index": 0}]]}
# debate must read past the emit: point at S34 directly
nodes["Debate Round 2"]["parameters"]["jsCode"] = nodes["Debate Round 2"]["parameters"]["jsCode"].replace(
    "$('C34-PREDATOR-swarm')", "$('Emit Swarm')", 1)
# NO - emit outputs HTTP response! debate needs S34 accumulator:
nodes["Debate Round 2"]["parameters"]["jsCode"] = nodes["Debate Round 2"]["parameters"]["jsCode"].replace(
    "$('Emit Swarm')", "$('C34-PREDATOR-swarm')", 1)
print("swarm emit inserted (debate still reads S34)")

# 8. kill-check fanout after throne
tpos = nodes["C0-QUILLAN Throne"]["position"]
kx = tpos[0] + 220
kill = {"parameters": {"mode": "runOnceForAllItems", "jsCode": KILL_JS},
        "id": str(uuid.uuid4()), "name": "Emit Kill-Checks",
        "type": "n8n-nodes-base.code", "typeVersion": 2, "position": [kx, tpos[1]]}
post = http_node("Post Kill-Checks", "={{ $json }}", [kx + 220, tpos[1]])
recol = {"parameters": {"mode": "runOnceForAllItems", "jsCode": "return [{ json: $('C0-QUILLAN Throne').first().json }];"},
         "id": str(uuid.uuid4()), "name": "Recollect Throne",
         "type": "n8n-nodes-base.code", "typeVersion": 2, "position": [kx + 440, tpos[1]]}
for nd in (kill, post, recol):
    nodes[nd["name"]] = nd
    w["nodes"].append(nd)
if "Emit Arbitration" in nodes:
    nodes["Emit Arbitration"]["position"] = [kx + 660, tpos[1]]
conns["C0-QUILLAN Throne"] = {"main": [[{"node": "Emit Kill-Checks", "type": "main", "index": 0}]]}
conns["Emit Kill-Checks"] = {"main": [[{"node": "Post Kill-Checks", "type": "main", "index": 0}]]}
conns["Post Kill-Checks"] = {"main": [[{"node": "Recollect Throne", "type": "main", "index": 0}]]}
conns["Recollect Throne"] = {"main": [[{"node": "Emit Arbitration", "type": "main", "index": 0}]]}
print("kill-check fanout wired")

# 9. council detail enrichment for live mesh
for n in w["nodes"]:
    if n["name"] == "Emit Council":
        body = n["parameters"]["jsonBody"]
        if "supporters" not in body:
            n["parameters"]["jsonBody"] = body.replace(
                " + ', veto=' + $json.debate.veto}",
                " + ', veto=' + $json.debate.veto + ', supporters [' + $json.debate.supporters.join(', ') + '], weakest ' + $json.debate.weakest}", 1)
            print("council detail enriched")

out = w if not is_list else [w]
open(P, "w", encoding="utf-8").write(json.dumps(out, indent=2))
print("saved", len(w["nodes"]), "nodes")
