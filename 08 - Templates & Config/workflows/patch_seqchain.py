#!/usr/bin/env python3
"""Sequential council chain: Prepare -> C1 -> S1 -> C2 -> S2 ... -> C34 -> S34 -> Debate.

Why: n8n fires multi-source nodes once PER source branch, and $('...') refs to
nodes outside the direct input path race execution order. A single chain carries
{bundle, verdicts[]} through every node via $input only - deterministic.
Backup first.
"""
import json
import shutil
import sys

sys.stdout.reconfigure(encoding="utf-8")
P = r"C:\02_QUILLAN\08 - Templates & Config\workflows\starter odea workflow.json"
shutil.copy(P, P + ".bak-seqchain")
print("backup written")

wf = json.load(open(P, encoding="utf-8"))
nodes = {n["name"]: n for n in wf["nodes"]}
conns = wf["connections"]

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


def cnum(name):
    return int("".join(c for c in name if c.isdigit()))


agents = sorted([n for n in nodes if n.startswith("C") and n[1].isdigit()
                 and not n.endswith("-swarm") and n != "C0-QUILLAN Throne"],
                key=cnum)
assert len(agents) == 34, f"expected 34 agents, got {len(agents)}"
swarms = [a + "-swarm" for a in agents]
for sname in swarms:
    assert sname in nodes, f"missing {sname}"

for aname in agents:
    pid = aname  # node name IS the persona id (C1-ASTRA etc.)
    nodes[aname]["parameters"]["jsCode"] = AGENT_JS.replace("__ID__", pid)
for sname in swarms:
    nodes[sname]["parameters"]["jsCode"] = SWARM_JS
print("code rewritten: 34 agents + 34 swarms")

# rewire into one chain; lay it left-to-right for watchability
order = []
for aname, sname in zip(agents, swarms):
    order += [aname, sname]
x = 1560
for name in order:
    nodes[name]["position"] = [x, 300]
    x += 180
nodes["Debate Round 2"]["position"] = [x + 40, 300]
x += 260
for name in ("Emit Council", "C0-QUILLAN Throne", "Emit Arbitration"):
    if name in nodes:
        nodes[name]["position"] = [x, 300]
        x += 220

for aname in agents:
    conns[aname] = {"main": [[]]}
for sname in swarms:
    conns[sname] = {"main": [[]]}
conns["Prepare 34 Assignments"] = {"main": [[{"node": agents[0], "type": "main", "index": 0}]]}
prev = agents[0]
for sname, aname in zip(swarms, agents):
    pass
# chain: C1->S1->C2->S2...->C34->S34->Debate
seq = [agents[0]]
for aname, sname in zip(agents, swarms):
    pass
chain = []
for aname, sname in zip(agents, swarms):
    chain += [aname, sname]
for a, b in zip(chain, chain[1:]):
    conns[a] = {"main": [[{"node": b, "type": "main", "index": 0}]]}
conns[chain[-1]] = {"main": [[{"node": "Debate Round 2", "type": "main", "index": 0}]]}
print("chain wired:", len(chain) + 1, "links")

json.dump(wf, open(P, "w", encoding="utf-8"), indent=2)
print("saved")
