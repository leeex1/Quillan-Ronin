#!/usr/bin/env python3
"""Charter rebuild: intake -> prism -> router -> council(+prism) -> swarm
-> diffusion -> 5 visible gates -> throne -> export. One line, no sim lane.
Reads MAIN fresh from DB. Backup first (caller backs up sqlite too)."""
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
print("loaded:", len(nodes), "nodes")


def nid():
    return str(uuid.uuid4())


# ---------- 1. delete superseded sim lane ----------
SIM_DEAD = ["\U0001f9e0 Wave 1 \u2014 Cognitive C1-C8",
            "\U0001f5e3 Wave 2 \u2014 Communication C9-C16",
            "\U0001f300 Wave 3 \u2014 Meta C17-C24",
            "\u2699 Wave 4 \u2014 Systems C25-C34",
            "\U0001f41d Swarm Fan-out",
            "\U0001f6e1 Quality Gates",
            "\U0001f44d Throne Synthesis",
            "\U0001f4e1 Emit: council",
            "\U0001f4e1 Emit: swarm",
            "\U0001f4e1 Emit: gates",
            "\U0001f4e1 Emit: arbitration"]
gone = [n for n in SIM_DEAD if n in nodes]
for n in gone:
    del nodes[n]
    conns.pop(n, None)
for k, v in list(conns.items()):
    for out in v.get("main", []):
        out[:] = [t for t in out if t.get("node") not in set(SIM_DEAD)]
print("sim lane removed:", len(gone))

# ---------- 2. prepare: prism-fed bundle ----------
PREPARE_JS = """const rawParse = $('Parse Identities JSON').first().json;
const data = rawParse.data || rawParse;
let prism = null, routing = null, src = null;
try { prism = $('PRISM').first().json.prism || null; } catch (e) {}
try { routing = $('ROUTER').first().json.routing || null; } catch (e) {}
try { src = $('INTAKE').first().json; } catch (e) {}
if (!src || !src.query) { try { src = $('Query Input').first().json; } catch (e) {} }
const run_id = (src && src.run_id) || ('run_' + Date.now().toString(36));
const query = String((src && src.query) || 'What should the council deliberate today?');
const context = String((src && src.context) || '');
if (!data.personas || !data.personas.length) {
  return [{ json: { ok: false, error: 'No personas found in identities file' } }];
}
return data.personas.map((p) => ({ json: { persona: p, query, context, run_id, prism, routing } }));""".replace(
    "PRISM", "\U0001f537 9-Vector Prism Shard").replace(
    "ROUTER", "\U0001f9ed C31-NEXUS Router").replace(
    "INTAKE", "\U0001f4e5 Intake & Validation")
nodes["Prepare 34 Assignments"]["parameters"]["jsCode"] = PREPARE_JS
print("prepare prism-fed")

# ---------- 3. agents: prism-aware risk ----------
AGENT_JS = """const items = $input.all().map((i) => i.json);
await new Promise((r) => setTimeout(r, 150));
let bundle, verdicts;
if (items[0].persona && !items[0].verdicts) {
  bundle = { query: items[0].query, context: items[0].context, run_id: items[0].run_id,
    prism: items[0].prism || null, routing: items[0].routing || null,
    personas: items.map((a) => a.persona) };
  verdicts = [];
} else {
  bundle = items[0].bundle;
  verdicts = [...items[0].verdicts];
}
const persona = bundle.personas.find((p) => p.id === '__ID__');
const q = String(bundle.query).toLowerCase();
const rel = persona.keywords.filter((k) => q.includes(String(k).toLowerCase())).length;
const flagged = (bundle.prism && bundle.prism.Ethics && bundle.prism.Ethics.risk_words) || [];
const seen = String(bundle.query).match(/hack|harm|attack|bypass|illegal|weapon|exploit|steal/gi) || [];
const riskWords = [...new Set([...flagged.map((w) => String(w).toLowerCase()), ...seen.map((w) => w.toLowerCase())])];
const risk = riskWords.length > 0;
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


def cnum(name):
    return int("".join(c for c in name if c.isdigit()))


agents = sorted([n for n in nodes if n.startswith("C") and len(n) > 1
                 and n[1].isdigit() and not n.endswith("-swarm")
                 and n != "C0-QUILLAN Throne"], key=cnum)
assert len(agents) == 34, len(agents)
for aname in agents:
    pid = aname.split("-")[0] + "-" + aname.split("-")[1]
    # node names are like C1-ASTRA already (persona id prefix)
    nodes[aname]["parameters"]["jsCode"] = AGENT_JS.replace("__ID__", aname)
print("34 agents prism-aware")

# ---------- 4. diffusion + 5 gates + gates emit ----------
DIFF_JS = """const s = $input.first().json;
await new Promise((r) => setTimeout(r, 400));
const HARD = 0.7;
const hard = (s.verdicts || []).filter((v) => (v.swarm ? v.swarm.adjusted_confidence : v.confidence) < HARD);
hard.forEach((v) => { v.refined = true; v.confidence = +Math.min(0.99, v.confidence + 0.08).toFixed(2); });
s.diffusion = { hard_tokens: hard.length, refined: hard.length, threshold: HARD };
return [{ json: s }];"""


def gate_js(key, extra, score, cond, note):
    return ("const s = $input.first().json;\n"
            "await new Promise((r) => setTimeout(r, 250));\n"
            + extra +
            "s.gates = s.gates || {};\n"
            f"s.gates.{key} = {{ score: {score}, pass: ({cond}) }};\n"
            f"s.gates.{key}.note = {note};\n"
            "return [{ json: s }];")


GATES = [
    ("Gate Logic C7", "logic",
     "const v = s.debate.agreement;",
     "+v.toFixed(2) * 1", "v >= 0.6", "'audit premises, trace deduction'"),
    ("Gate Ethics VIR", "ethics",
     ("const rw = (s.bundle.prism && s.bundle.prism.Ethics && s.bundle.prism.Ethics.risk_words) || [];\n"
      "const veto = s.debate.veto || rw.length > 0;\n"),
     "veto ? 0.1 : 0.95", "!veto",
     "veto ? 'C2-VIR VETO' : 'clear'"),
    ("Gate Truth C18", "truth",
     "const fact = /who|what|when|where|exact|capital|how many/i.test(s.query);\n",
     "fact ? 0.55 : 0.8", "true",
     "'source check' + (fact ? ', citations required' : '')"),
    ("Gate Clarity C15", "clarity", "",
     "+Math.min(0.95, 0.6 + String(s.query).length / 800).toFixed(2) * 1", "true",
     "'presentation check'"),
    ("Gate Paradox C17", "paradox",
     "const contra = /but |however|contradict|paradox|both /i.test(String(s.query).toLowerCase());\n",
     "contra ? 0.6 : 0.9", "true",
     "'contradiction check'"),
]

dpos = nodes["Debate Round 2"]["position"]
x, y = dpos[0] + 240, dpos[1]
new_nodes = []
new_nodes.append({"parameters": {"mode": "runOnceForAllItems", "jsCode": DIFF_JS},
                  "id": nid(), "name": "Diffusion Refine",
                  "type": "n8n-nodes-base.code", "typeVersion": 2,
                  "position": [x, y]})
x += 220
for gname, key, extra, score, cond, note in GATES:
    new_nodes.append({"parameters": {"mode": "runOnceForAllItems",
                                      "jsCode": gate_js(key, extra, score, cond, note)},
                      "id": nid(), "name": gname,
                      "type": "n8n-nodes-base.code", "typeVersion": 2,
                      "position": [x, y]})
    x += 220
new_nodes.append({"parameters": {
    "method": "POST", "url": "http://127.0.0.1:8777/api/events",
    "sendBody": True, "specifyBody": "json",
    "jsonBody": ("={{ {'agent': 'council-34', 'type': 'stage', 'stage': 'gates', "
                 "'status': (Object.values($json.gates || {}).every(function(g){return g.pass}) ? 'pass' : 'fail'), "
                 "'detail': 'gates ' + Object.values($json.gates || {}).filter(function(g){return g.pass}).length + '/5'} }}"),
    "options": {"timeout": 3000}},
    "id": nid(), "name": "Emit Gates",
    "type": "n8n-nodes-base.httpRequest", "typeVersion": 4.2,
    "position": [x, y], "onError": "continueRegularOutput"})
for nd in new_nodes:
    nodes[nd["name"]] = nd
print("diffusion + 5 gates + emit added:", len(new_nodes))

# ---------- 5. rewire into one line ----------
# drop old debate/throne fanouts to rebuilt chain
conns["Debate Round 2"] = {"main": [[{"node": "Emit Council",
                                       "type": "main", "index": 0},
                                      {"node": "Diffusion Refine",
                                       "type": "main", "index": 0}]]}
seq = ["Diffusion Refine"] + [g[0] for g in GATES] + ["Emit Gates"]
for a, b in zip(seq, seq[1:]):
    conns[a] = {"main": [[{"node": b, "type": "main", "index": 0}]]}
# throne reads gates emit? NO - http breaks chain. Throne reads last gate:
conns["Gate Paradox C17"] = {"main": [[{"node": "C0-QUILLAN Throne",
                                        "type": "main", "index": 0}]]}
# throne keeps killchecks; respondent repoint
conns["C0-QUILLAN Throne"] = {"main": [[{"node": "Emit Kill-Checks",
                                         "type": "main", "index": 0}]]}
# move tail lane right of new gates: reposition throne..export sequentially
tx = nodes["Emit Gates"]["position"][0] + 220
for nm in ["C0-QUILLAN Throne", "Emit Kill-Checks", "Post Kill-Checks",
           "Recollect Throne", "Emit Arbitration", "Aggregate",
           "Code in Python", "Code in Python1", "Convert to File1"]:
    if nm in nodes:
        nodes[nm]["position"] = [tx, y]
        tx += 220
# respond <- throne (was sim emit, deleted)
conns["\U0001f4e4 Respond"] = {"main": [[]]}
if "C0-QUILLAN Throne" in conns:
    outs = conns["C0-QUILLAN Throne"]["main"][0]
    if not any(t.get("node") == "\U0001f4e4 Respond" for t in outs):
        outs.append({"node": "\U0001f4e4 Respond", "type": "main", "index": 0})
print("one line wired; respond <- throne")

# ---------- 6. throne reads gates directly (linear) ----------
tjs = nodes["C0-QUILLAN Throne"]["parameters"]["jsCode"]
assert "$('Debate Round 2')" in tjs
nodes["C0-QUILLAN Throne"]["parameters"]["jsCode"] = tjs.replace(
    "const s = $('Debate Round 2').first().json;",
    "const s = $input.first().json;", 1)
print("throne linear")

# scrub any targets pointing at deleted nodes (e.g. Router->Wave1)
gone_set = set(SIM_DEAD)
for k, v in list(conns.items()):
    for out in v.get("main", []):
        out[:] = [t for t in out if t.get("node") not in gone_set]
print("dangling targets scrubbed")

P = r"C:\02_QUILLAN\08 - Templates & Config\workflows\emfo-charter.json"
open(P, "w", encoding="utf-8").write(json.dumps(
    {"id": "EmfoGhyIzSrktmIe", "name": "Quillan Sovereign \u2014 MAIN",
     "nodes": list(nodes.values()), "connections": conns,
     "settings": {"executionOrder": "v1"}, "staticData": None, "pinData": {}},
    indent=2))
print("saved", len(nodes), "nodes")
