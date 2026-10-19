#!/usr/bin/env python3
"""Tier-faithful rebuild (reads MAIN fresh from DB):
TIER 1: trigger -> query/read/parse -> PullAssign (persona pull weights)
        -> Prepare -> ...
TIER 2: 34 council agents (pull-weighted confidence)
TIER 3: 34 swarm nodes (unchanged logic)
GATES -> E-ICE governor -> throne audit -> export.
Backup: relies on sqlite .bak files + emfo-main.json copies.
"""
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
print("loaded:", len(nodes))


def nid():
    return str(uuid.uuid4())


PULL_JS = """const raw = $input.first().json;
const data = raw.data || raw;
let prism = null, routing = null, q = null, qi = null;
try { prism = $('PRISM').first().json.prism || null; } catch (e) {}
try { routing = $('ROUTER').first().json.routing || null; } catch (e) {}
try { q = $('INTAKE').first().json; } catch (e) {}
try { qi = $('Query Input').first().json; } catch (e) {}
const src = (q && q.query) ? q : qi;
const run_id = (src && src.run_id) || ('run_' + Date.now().toString(36));
const query = String((src && src.query) || 'What should the council deliberate today?');
const context = String((src && src.context) || '');
const ql = query.toLowerCase();
const pulls = {};
(data.personas || []).forEach((p) => {
  const rel = (p.keywords || []).filter((k) => ql.includes(String(k).toLowerCase())).length;
  pulls[p.id] = +Math.max(0.2, Math.min(0.99, p.prior * (0.6 + 0.2 * Math.min(rel, 2)))).toFixed(2);
});
return [{ json: { personas: data.personas || [], pulls, query, context, run_id, prism, routing } }];""".replace(
    "PRISM", "\U0001f537 9-Vector Prism Shard").replace(
    "ROUTER", "\U0001f9ed C31-NEXUS Router").replace(
    "INTAKE", "\U0001f4e5 Intake & Validation")

PREPARE_JS = """const inp = $input.first().json;
if (!inp.personas || !inp.personas.length) {
  return [{ json: { ok: false, error: 'No personas from Pull Assign' } }];
}
return inp.personas.map((p) => ({ json: { persona: p, query: inp.query, context: inp.context,
  run_id: inp.run_id, prism: inp.prism || null, routing: inp.routing || null,
  pulls: inp.pulls || {} } }));"""

AGENT_JS = """const items = $input.all().map((i) => i.json);
await new Promise((r) => setTimeout(r, 150));
let bundle, verdicts;
if (items[0].persona && !items[0].verdicts) {
  bundle = { query: items[0].query, context: items[0].context, run_id: items[0].run_id,
    prism: items[0].prism || null, routing: items[0].routing || null,
    pulls: items[0].pulls || {},
    personas: items.map((a) => a.persona) };
  verdicts = [];
} else {
  bundle = items[0].bundle;
  verdicts = [...items[0].verdicts];
}
const persona = bundle.personas.find((p) => p.id === '__ID__');
const pull = (bundle.pulls && bundle.pulls[persona.id]) || persona.prior;
const q = String(bundle.query).toLowerCase();
const rel = persona.keywords.filter((k) => q.includes(String(k).toLowerCase())).length;
const flagged = (bundle.prism && bundle.prism.Ethics && bundle.prism.Ethics.risk_words) || [];
const seen = String(bundle.query).match(/hack|harm|attack|bypass|illegal|weapon|exploit|steal/gi) || [];
const riskWords = [...new Set([...flagged.map((w) => String(w).toLowerCase()), ...seen.map((w) => w.toLowerCase())])];
const risk = riskWords.length > 0;
let stance = 'neutral';
if ((persona.id === 'C2-VIR' || persona.id === 'C13-WARDEN') && risk) stance = 'caution';
else if (rel >= 1) stance = 'support';
const base = Math.max(0.05, Math.min(0.99, persona.prior - 0.12 + 0.09 * Math.min(rel, 3)));
const confidence = +Math.max(0.05, Math.min(0.99, base * (0.7 + 0.3 * pull))).toFixed(2);
verdicts.push({ agent_id: persona.id, name: persona.name, role: persona.role, cluster: persona.cluster,
  query: bundle.query, context: bundle.context, run_id: bundle.run_id, pull,
  stance, confidence, relevance_hits: rel,
  points: [persona.role + ' lens (pull ' + pull + '): ' + (rel ? 'directly relevant (' + rel + ' keyword hits)' : 'general perspective'),
    stance === 'caution' ? 'RAISES OBJECTION on safety/ethics grounds' : 'position ' + stance + ' at ' + confidence] });
return [{ json: { bundle, verdicts } }];"""

EICE_JS = """const s = $input.first().json;
await new Promise((r) => setTimeout(r, 250));
const n = (s.verdicts || []).length;
const coherence = s.debate ? s.debate.agreement : 0.5;
const entropy = Math.max(0.5, 2 - coherence);
const depth = Math.max(1, n);
const e_ice = {
  depth, coherence: +coherence.toFixed(2), entropy: +entropy.toFixed(2),
  load: +Math.min(0.99, (depth * coherence / entropy / 34)).toFixed(2),
  bound: 'thermodynamic throttle: stable'
};
s.e_ice = e_ice;
return [{ json: s }];"""


def cnum(name):
    return int("".join(c for c in name if c.isdigit()))


agents = sorted([n for n in nodes if n.startswith("C") and len(n) > 1
                 and n[1].isdigit() and not n.endswith("-swarm")
                 and n != "C0-QUILLAN Throne"], key=cnum)
assert len(agents) == 34, len(agents)

# 1. PullAssign node (tier-1 pull gate, after Parse)
qx = nodes["Parse Identities JSON"]["position"]
pull = {"parameters": {"mode": "runOnceForAllItems", "jsCode": PULL_JS},
        "id": nid(), "name": "Pull Assign (Throne)",
        "type": "n8n-nodes-base.code", "typeVersion": 2,
        "position": [qx[0] + 220, qx[1]]}
nodes[pull["name"]] = pull
conns["Parse Identities JSON"] = {"main": [[{"node": "Emit Intake",
                                             "type": "main", "index": 0},
                                            {"node": "Pull Assign (Throne)",
                                             "type": "main", "index": 0}]]}
conns["Pull Assign (Throne)"] = {"main": [[{"node": "Prepare 34 Assignments",
                                            "type": "main", "index": 0}]]}
# emit stays a leaf hub event; prepare has exactly one parent now
if "Emit Intake" in conns:
    conns["Emit Intake"] = {"main": [[]]}
print("emit-intake parked as leaf; prepare single-parent")
# prepare now reads pull-assign directly (linear, no $())
nodes["Prepare 34 Assignments"]["parameters"]["jsCode"] = PREPARE_JS
print("pull gate wired; prepare linear")

# 2. agents pull-weighted
for aname in agents:
    nodes[aname]["parameters"]["jsCode"] = AGENT_JS.replace("__ID__", aname)
print("34 agents pull-weighted")

# 3. debate: pull-weighted consensus
djs = nodes["Debate Round 2"]["parameters"]["jsCode"]
old_mean = "const agreement = +mean(verdicts.map(conf)).toFixed(2);"
assert old_mean in djs
new_mean = ("const pw = verdicts.map((v) => (typeof v.pull === 'number' ? v.pull : 0.8));\n"
            "const agreement = +((verdicts.reduce((a, v, i) => a + conf(v) * pw[i], 0) / Math.max(1e-6, pw.reduce((a, b) => a + b, 0))).toFixed(2));")
nodes["Debate Round 2"]["parameters"]["jsCode"] = djs.replace(old_mean, new_mean, 1)
print("debate pull-weighted")

# 4. E-ICE governor between paradox gate and throne
gpos = nodes["Gate Paradox C17"]["position"]
eice = {"parameters": {"mode": "runOnceForAllItems", "jsCode": EICE_JS},
        "id": nid(), "name": "E-ICE Governor",
        "type": "n8n-nodes-base.code", "typeVersion": 2,
        "position": [gpos[0] + 220, gpos[1]]}
nodes[eice["name"]] = eice
conns["Gate Paradox C17"] = {"main": [[{"node": "E-ICE Governor",
                                        "type": "main", "index": 0}]]}
conns["E-ICE Governor"] = {"main": [[{"node": "C0-QUILLAN Throne",
                                      "type": "main", "index": 0}]]}
# shift throne tail right to make room
tx = gpos[0] + 440
for nm in ["C0-QUILLAN Throne", "Emit Kill-Checks", "Post Kill-Checks",
           "Recollect Throne", "Emit Arbitration", "Aggregate",
           "Code in Python", "Code in Python1", "Convert to File1"]:
    if nm in nodes:
        p = nodes[nm]["position"]
        nodes[nm]["position"] = [tx, p[1]]
        tx += 220
print("e-ice wired; tail shifted")

# 5. zone sticky tier language
TIERS = {
    "Zone: intake": "TIER 1 \u2014 THRONE INTAKE\nTrigger -> query -> identities -> PULL ASSIGN (persona pull weights).",
    "Zone: council": "TIER 2 \u2014 COUNCIL DELIBERATION\n34 experts, pull-weighted stances. Follow the snake.",
    "Zone: throne": "EXIT GATES + THRONE AUDIT\nDiffusion -> 5 gates -> E-ICE governor -> C0 final arbitration -> verdict file.",
    "Zone: webhook": "WEBHOOK SIM LANE\nPipeline demo + feeds prism/router into the council.",
}
for k, v in TIERS.items():
    if k in nodes:
        nodes[k]["parameters"]["content"] = v
print("zones relabeled")

# 6. single-parent audit on the touched chain
core = ["Parse Identities JSON", "Pull Assign (Throne)",
        "Prepare 34 Assignments", "Debate Round 2", "Diffusion Refine",
        "Gate Logic C7", "Gate Ethics VIR", "Gate Truth C18",
        "Gate Clarity C15", "Gate Paradox C17", "E-ICE Governor",
        "C0-QUILLAN Throne"]
parents = {}
for k, v in conns.items():
    for o in v.get("main", []):
        for t in o:
            parents.setdefault(t.get("node"), []).append(k)
for c in core:
    ps = [p for p in parents.get(c, [])]
    flag = "" if (c == "Parse Identities JSON" or len(ps) == 1) else "  <-- CHECK"
    print(f"parents({c}) = {ps}{flag}")

open(r"C:\02_QUILLAN\08 - Templates & Config\workflows\emfo-main.json",
     "w", encoding="utf-8").write(json.dumps(
    {"id": "EmfoGhyIzSrktmIe", "name": "Quillan Sovereign \u2014 MAIN",
     "nodes": list(nodes.values()), "connections": conns,
     "settings": {"executionOrder": "v1"}, "staticData": None, "pinData": {}},
    indent=2))
print("saved", len(nodes), "nodes")
