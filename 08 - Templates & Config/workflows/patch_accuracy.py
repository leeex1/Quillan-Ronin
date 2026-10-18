#!/usr/bin/env python3
"""Accuracy rebuild (reads MAIN fresh from DB):
- PullAssign: spec pull-softmax (priors x relevance, tau=1.0) + pull entropy.
- Agents: record pull (unchanged shape otherwise).
- Swarms: EGGROLL rank-8 world-sim (seeded, deterministic, no fake trio).
- Gates: canonical NULLION / WARDEN / SHEPHERD.
- E-ICE: Landauer analytic energy from run entropy. Lee-Mach PID step.
- Throne: audit. Typist: real polish pass. KillChecks: new gate shape.
Layout positions reused from replaced nodes where possible.
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


# ---------- PullAssign: spec softmax ----------
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
const TAU = 1.0;
const scored = (data.personas || []).map((p) => {
  const rel = (p.keywords || []).filter((k) => ql.includes(String(k).toLowerCase())).length;
  return { p, score: p.prior * (1 + rel) };
});
const mx = Math.max(...scored.map((s) => s.score / TAU));
const exps = scored.map((s) => Math.exp(s.score / TAU - mx));
const z = exps.reduce((a, b) => a + b, 0);
const pulls = {};
scored.forEach((s, i) => { pulls[s.p.id] = +(exps[i] / z).toFixed(4); });
let entropy = 0;
Object.values(pulls).forEach((v) => { if (v > 0) entropy -= v * Math.log(v); });
return [{ json: { personas: data.personas || [], pulls, pull_entropy: +entropy.toFixed(4),
  tau: TAU, query, context, run_id, prism, routing } }];""".replace(
    "PRISM", "\U0001f537 9-Vector Prism Shard").replace(
    "ROUTER", "\U0001f9ed C31-NEXUS Router").replace(
    "INTAKE", "\U0001f4e5 Intake & Validation")
nodes["Pull Assign (Throne)"]["parameters"]["jsCode"] = PULL_JS
print("pull-softmax live")

# ---------- agents: keep shape, pull already recorded ----------
# (no change needed - verdicts carry pull from bundle.pulls)

# ---------- swarms: EGGROLL rank-8 ----------
SWARM_JS = """const s = $input.first().json;
await new Promise((r) => setTimeout(r, 150));
const v = s.verdicts[s.verdicts.length - 1];
function hash(str) { let h = 2166136261; for (let i = 0; i < str.length; i++) { h ^= str.charCodeAt(i); h = Math.imul(h, 16777619); } return h >>> 0; }
function rng(seed) { let a = seed >>> 0; return function () { a |= 0; a = (a + 0x6D2B79F5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }
const R = 8, D = 8, COUPLING = 0.1, SCALE = 0.25;
const rnd = rng(hash(v.agent_id));
const gauss = () => (rnd() + rnd() + rnd() - 1.5) * 0.02;
const A = [], B = [];
for (let i = 0; i < D; i++) { A.push([]); for (let j = 0; j < R; j++) A[i].push(gauss()); }
for (let i = 0; i < R; i++) { B.push([]); for (let j = 0; j < D; j++) B[i].push(gauss()); }
const state = [v.confidence, Math.min(1, v.relevance_hits / 3), v.pull || 0.8, 0.9, v.stance === 'caution' ? 1 : 0, 0.5, 0.5, 0.5];
const mm = (X, M) => M[0].map((_, j) => X.reduce((a, x, i) => a + x * M[i][j], 0));
const tanh = (X) => X.map((x) => Math.tanh(x));
const inter = tanh(mm(mm(state, A), B));
const evolved = state.map((x, i) => x + COUPLING * inter[i]);
const modulation = evolved.reduce((a, b) => a + b, 0) / D - state.reduce((a, b) => a + b, 0) / D;
const adjusted = Math.max(0.05, Math.min(0.99, v.confidence + modulation * SCALE * 4));
v.swarm = { engine: 'EGGROLL rank-r world-sim', rank: R, coupling: COUPLING,
  diversity: +Math.abs(modulation).toFixed(4),
  adjusted_confidence: +adjusted.toFixed(2) };
return [{ json: s }];"""
c = 0
for n in nodes.values():
    if n["name"].endswith("-swarm"):
        n["parameters"]["jsCode"] = SWARM_JS
        c += 1
print("eggroll swarms:", c)

# ---------- E-ICE + PID ----------
EICE_JS = """const s = $input.first().json;
await new Promise((r) => setTimeout(r, 250));
const KB = 1.380649e-23, T = 300.0, LN2 = Math.LN2, GAMMA = 1.0;
const ent = (s.pull_entropy && s.pull_entropy > 1e-6) ? s.pull_entropy : Math.log(34);
const coherence = s.debate ? s.debate.agreement : 0.5;
const depth = (s.verdicts || []).length || 34;
const Is = (depth * coherence) / ent;
const e_omega = Is * GAMMA * GAMMA * KB * T * LN2;
const load = Math.min(0.99, (coherence * Math.log(34)) / Math.max(ent, 1e-6));
const integ = Math.min(1, coherence + 0.05);
const err = (0.85 - integ) - 0.5 * (0.90 - load);
const thresh = Math.max(0.40, Math.min(0.99, 0.80 + 0.15 * err));
s.e_ice = { joules: e_omega, load: +load.toFixed(2), bound_ok: load <= 0.90,
  depth, coherence: +coherence.toFixed(2), entropy: +ent.toFixed(3) };
s.lm6 = { target_integrity: 0.85, max_load: 0.90, kp: 0.15, ki: 0.05, kd: 0.02,
  pid_error: +err.toFixed(3), hard_threshold: +thresh.toFixed(2) };
return [{ json: s }];"""

GATE_TMPL = """const s = $input.first().json;
await new Promise((r) => setTimeout(r, 250));
s.gates = s.gates || {};
__BODY__
return [{ json: s }];"""
GATES = {
    "Gate Nullion C17": ("paradox",
        "const contra = /but |however|contradict|paradox|both /i.test(String(s.query).toLowerCase());\n"
        "const held = contra && s.debate.agreement < 0.6;\n"
        "s.gates.paradox = { score: held ? 0.6 : 0.9, pass: !held, note: held ? 'contradiction held open' : 'no paradox block' };"),
    "Gate Warden C13": ("safety",
        "const load = (s.e_ice && s.e_ice.load) || 0;\n"
        "const over = load > 0.90;\n"
        "const rw = (s.bundle.prism && s.bundle.prism.Ethics && s.bundle.prism.Ethics.risk_words) || [];\n"
        "s.gates.safety = { score: over ? 0.2 : (rw.length ? 0.5 : 0.95), pass: !over, energy_bound_ok: !over, risk_words: rw };"),
    "Gate Shepherd C18": ("truth",
        "const fact = /who|what|when|where|exact|capital|how many/i.test(s.query);\n"
        "s.gates.truth = { score: fact ? 0.55 : 0.8, pass: true, citations_needed: fact };"),
}

# replace old 5 gates with canonical 3 (reuse first 3 positions, delete rest)
OLD5 = ["Gate Logic C7", "Gate Ethics VIR", "Gate Truth C18", "Gate Clarity C15", "Gate Paradox C17"]
oldpos = [nodes[n]["position"] for n in OLD5 if n in nodes]
for n in OLD5:
    nodes.pop(n, None)
    conns.pop(n, None)
for k, v in list(conns.items()):
    for out in v.get("main", []):
        out[:] = [t for t in out if t.get("node") not in set(OLD5)]
print("old gates removed")
new_order = ["Gate Nullion C17", "Gate Warden C13", "Gate Shepherd C18"]
for (nm, key, body), pos in zip(
        [("Gate Nullion C17", "paradox", GATES["Gate Nullion C17"][1]),
         ("Gate Warden C13", "safety", GATES["Gate Warden C13"][1]),
         ("Gate Shepherd C18", "truth", GATES["Gate Shepherd C18"][1])],
        oldpos[:3]):
    nodes[nm] = {"parameters": {"mode": "runOnceForAllItems",
                                "jsCode": GATE_TMPL.replace("__BODY__", body)},
                 "id": nid(), "name": nm, "type": "n8n-nodes-base.code",
                 "typeVersion": 2, "position": pos}
print("canonical gates placed")

# chain: Diffusion -> E-ICE -> Nullion -> Warden -> Shepherd -> Throne
epos = nodes["Diffusion Refine"]["position"]
eice = {"parameters": {"mode": "runOnceForAllItems", "jsCode": EICE_JS},
        "id": nid(), "name": "E-ICE Governor",
        "type": "n8n-nodes-base.code", "typeVersion": 2,
        "position": [epos[0], epos[1]]}
# shift diffusion left to make room, e-ice takes diffusion's neighbors
dpos = nodes["Diffusion Refine"]["position"]
nodes["Diffusion Refine"]["position"] = [dpos[0] - 220, dpos[1]]
nodes[eice["name"]] = eice
seq = ["Diffusion Refine", "E-ICE Governor", "Gate Nullion C17",
       "Gate Warden C13", "Gate Shepherd C18", "C0-QUILLAN Throne"]
for a, b in zip(seq, seq[1:]):
    conns[a] = {"main": [[{"node": b, "type": "main", "index": 0}]]}
# move throne tail right of shepherd
tx = nodes["Gate Shepherd C18"]["position"][0] + 220
for nm in ["C0-QUILLAN Throne", "Typist C33", "Emit Kill-Checks",
           "Post Kill-Checks", "Recollect Throne", "Emit Arbitration",
           "Aggregate", "Code in Python", "Code in Python1",
           "Convert to File1"]:
    if nm in nodes:
        nodes[nm]["position"] = [tx, nodes[nm]["position"][1]]
        tx += 220
print("gate chain wired")

# ---------- throne audit (linear, reads gates) ----------
tjs = nodes["C0-QUILLAN Throne"]["parameters"]["jsCode"]
assert "const blocked = !gates.ethics.pass;" in tjs, "throne anchor missing"
nodes["C0-QUILLAN Throne"]["parameters"]["jsCode"] = tjs.replace(
    "const blocked = !gates.ethics.pass;",
    "const blocked = Object.values(s.gates || {}).some((g) => g.pass === false);", 1)
print("throne reads canonical gates")

# ---------- typist polish pass ----------
TYPIST_JS = """const s = $input.first().json;
await new Promise((r) => setTimeout(r, 300));
let t = String(s.answer || '');
t = t.replace(/\\s+/g, ' ').trim();
t = t.charAt(0).toUpperCase() + t.slice(1);
if (!/[.!?]$/.test(t)) t += '.';
s.answer = t;
s.typist_refined = true;
return [{ json: s }];"""
tpos = nodes["C0-QUILLAN Throne"]["position"]
typ = {"parameters": {"mode": "runOnceForAllItems", "jsCode": TYPIST_JS},
       "id": nid(), "name": "Typist C33",
       "type": "n8n-nodes-base.code", "typeVersion": 2,
       "position": [tpos[0], tpos[1]]}
nodes[typ["name"]] = typ
# throne -> typist -> killchecks ; throne keeps respond? check current outs
touts = [t["node"] for o in conns.get("C0-QUILLAN Throne", {}).get("main", [])
         for t in o]
print("throne outs before:", touts)
conns["C0-QUILLAN Throne"] = {"main": [[{"node": "Typist C33",
                                         "type": "main", "index": 0}]]}
conns["Typist C33"] = {"main": [[{"node": t, "type": "main", "index": 0}
                                 for t in touts]]}
# recollect + respond must read typist-final state
for nm, ref in (("Recollect Throne", "Typist C33"),
                ("\U0001f4e4 Respond", "Typist C33")):
    if nm in nodes:
        js = nodes[nm]["parameters"].get("jsCode") or nodes[nm]["parameters"].get("responseBody", "")
        if "C0-QUILLAN Throne" in js:
            key = "jsCode" if "jsCode" in nodes[nm]["parameters"] else "responseBody"
            nodes[nm]["parameters"][key] = js.replace("C0-QUILLAN Throne", ref)
            print(nm, "reads typist-final")
print("typist wired")

# ---------- killchecks: new gate shape ----------
for n in nodes.values():
    if n["name"] == "Emit Kill-Checks":
        js = n["parameters"]["jsCode"]
        assert "s.gates.logic.pass" in js and "s.gates.ethics.pass" in js
        js = js.replace("s.gates.logic.pass", "s.debate.agreement >= 0.6")
        js = js.replace("s.gates.ethics.pass", "!s.debate.veto")
        n["parameters"]["jsCode"] = js
        print("killchecks read canonical state")

# ---------- gates hub emit rides alongside shepherd->throne ----------
sout = [t["node"] for o in conns.get("Gate Shepherd C18", {}).get("main", [])
        for t in o]
assert sout == ["C0-QUILLAN Throne"], f"shepherd outs wrong: {sout}"
sout.append({"node": "Emit Gates", "type": "main", "index": 0})
conns["Gate Shepherd C18"] = {"main": [sout]}
print("shepherd feeds throne + gates emit")

# ---------- make room: typist takes its own slot, tail shifts right ----------
tpos = nodes["C0-QUILLAN Throne"]["position"]
nodes["Typist C33"]["position"] = [tpos[0] + 220, tpos[1]]
tx = tpos[0] + 440
for nm in ["Emit Kill-Checks", "Post Kill-Checks", "Recollect Throne",
           "Emit Arbitration", "Aggregate", "Code in Python",
           "Code in Python1", "Convert to File1",
           "\U0001f4e4 Respond"]:
    if nm in nodes:
        nodes[nm]["position"] = [tx, nodes[nm]["position"][1]]
        tx += 220
if "\U0001f4e4 Respond" in nodes:
    nodes["\U0001f4e4 Respond"]["position"][1] = nodes["C0-QUILLAN Throne"]["position"][1]
print("tail spaced for typist")

# ---------- gates emit body: 3 gates ----------
for n in nodes.values():
    if n["name"] == "Emit Gates":
        n["parameters"]["jsonBody"] = (
            "={{ {'agent': 'council-34', 'type': 'stage', 'stage': 'gates', "
            "'status': (Object.values($json.gates || {}).every(function(g){return g.pass}) ? 'pass' : 'fail'), "
            "'detail': 'gates nullion/warden/shepherd: ' + Object.values($json.gates || {}).filter(function(g){return g.pass}).length + '/3'} }}")
        print("gates emit updated")

open(r"C:\02_QUILLAN\08 - Templates & Config\workflows\emfo-main.json",
     "w", encoding="utf-8").write(json.dumps(
    {"id": "EmfoGhyIzSrktmIe", "name": "Quillan Sovereign \u2014 MAIN",
     "nodes": list(nodes.values()), "connections": conns,
     "settings": {"executionOrder": "v1"}, "staticData": None, "pinData": {}},
    indent=2))
print("saved", len(nodes), "nodes")
