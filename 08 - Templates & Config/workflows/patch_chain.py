#!/usr/bin/env python3
"""Convert 34 parallel swarm->Debate links into a single accumulator chain.

Why: n8n runs a multi-source node once PER source branch (34 Debate runs of
1 verdict). A sequential chain S1->S2->...->S34->Debate carries one item with
a growing verdicts[] array, so Debate fires once with all 34.
Backup first. Idempotent: safe to re-run (rebuilds chain links from scratch).
"""
import json
import shutil
import sys

sys.stdout.reconfigure(encoding="utf-8")
P = r"C:\02_QUILLAN\08 - Templates & Config\workflows\starter odea workflow.json"
shutil.copy(P, P + ".bak-chain")
print("backup written")

wf = json.load(open(P, encoding="utf-8"))
nodes = {n["name"]: n for n in wf["nodes"]}
conns = wf["connections"]

SWARM_JS = """const incoming = $input.first().json;
const v = $('__AGENT__').first().json;
await new Promise((r) => setTimeout(r, 200));
const seed = String(v.agent_id).split('').reduce((a, c) => a + c.charCodeAt(0), 0);
const j = (n) => (((seed * (n + 7)) % 11) - 5) / 100;
const clamp = (x) => Math.max(0.05, Math.min(0.99, x));
const positions = [
  { angle: 'support', confidence: +clamp(v.confidence + 0.04 + j(1)).toFixed(2), note: 'steelman the verdict' },
  { angle: 'attack', confidence: +clamp(v.confidence - 0.06 + j(2)).toFixed(2), note: 'stress-test the weakest premise' },
  { angle: 'refine', confidence: +clamp(v.confidence + 0.02 + j(3)).toFixed(2), note: 'tighten wording and scope' }
];
const adjusted = +clamp(v.confidence * 0.6 + positions.reduce((a, p) => a + p.confidence, 0) / 3 * 0.4).toFixed(2);
const mine = { agent_id: v.agent_id, name: v.name, role: v.role, cluster: v.cluster,
  query: v.query, context: v.context, run_id: v.run_id, stance: v.stance,
  confidence: v.confidence, relevance_hits: v.relevance_hits, points: v.points,
  swarm: { positions, adjusted_confidence: adjusted } };
const arr = Array.isArray(incoming.verdicts) ? [...incoming.verdicts] : [];
if (!arr.some((x) => x.agent_id === mine.agent_id)) arr.push(mine);
return [{ json: { verdicts: arr, query: mine.query, context: mine.context, run_id: mine.run_id } }];"""

# order swarms by council number
def cnum(swarm_name):
    agent = swarm_name[:-len("-swarm")]
    return int("".join(c for c in agent if c.isdigit()))

swarms = sorted([n for n in nodes if n.endswith("-swarm")], key=cnum)
assert len(swarms) == 34, f"expected 34 swarms, got {len(swarms)}"
agents = [s[:-len("-swarm")] for s in swarms]

# 1. rewrite swarm code to accumulator style
for sname, aname in zip(swarms, agents):
    assert aname in nodes, f"missing agent {aname}"
    nodes[sname]["parameters"]["jsCode"] = SWARM_JS.replace("__AGENT__", aname)
print("swarm code rewritten: 34")

# 2. Debate reads the single accumulated item
djs = nodes["Debate Round 2"]["parameters"]["jsCode"]
old_head = "const verdicts = $input.all().map((i) => i.json);"
assert old_head in djs, "debate head not found"
nodes["Debate Round 2"]["parameters"]["jsCode"] = djs.replace(
    old_head,
    "const incomingDebate = $input.first().json;\n"
    "const verdicts = Array.isArray(incomingDebate.verdicts) ? incomingDebate.verdicts : [];",
    1)
print("debate input fixed")

# 3. rewire: keep C1->S1 kick, chain S1->S2->...->S34->Debate, drop the rest
for aname in agents:
    conns[aname] = {"main": [[]]}
conns[agents[0]] = {"main": [[{"node": swarms[0], "type": "main", "index": 0}]]}
for sname in swarms:
    conns[sname] = {"main": [[]]}
for a, b in zip(swarms, swarms[1:]):
    conns[a] = {"main": [[{"node": b, "type": "main", "index": 0}]]}
conns[swarms[-1]] = {"main": [[{"node": "Debate Round 2", "type": "main", "index": 0}]]}
print("chain wired: kick + 33 links + debate feed")

json.dump(wf, open(P, "w", encoding="utf-8"), indent=2)
print("saved")
