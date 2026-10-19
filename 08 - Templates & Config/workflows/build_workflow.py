#!/usr/bin/env python3
"""Build the Quillan Council Pipeline n8n workflow JSON."""
import json, uuid
from pathlib import Path

OUT = Path(r"C:\02_QUILLAN\06 - Deployment & Platforms\n8n\quillan-council-pipeline.json")

def nid():
    return str(uuid.uuid4())

INTAKE_JS = """await new Promise((r) => setTimeout(r, 300));
const raw = $input.first().json;
const body = raw.body && typeof raw.body === 'object' ? raw.body : raw;
const query = String(body.query || '').trim();
const context = String(body.context || '');
const documents = Array.isArray(body.documents) ? body.documents : [];
if (!query) {
  return [{ json: { ok: false, stage: 'intake',
    error: 'Missing required field: query. POST JSON like {"query":"your question","context":"optional","documents":[{"name":"a.txt","text":"..."}]}' } }];
}
const docs = documents.map((d, i) => ({
  name: String(d.name || ('doc_' + (i + 1))),
  chars: String(d.text || '').length
}));
return [{ json: { ok: true, stage: 'intake',
  run_id: 'run_' + Date.now().toString(36),
  received_at: new Date().toISOString(),
  query, context, documents: docs,
  doc_chars: docs.reduce((a, d) => a + d.chars, 0) + context.length } }];"""

PRISM_JS = """const s = $input.first().json;
if (s.ok === false) return [{ json: s }];
await new Promise((r) => setTimeout(r, 500));
const q = s.query.toLowerCase();
const hits = (...words) => words.filter((w) => q.includes(w));
const riskWords = hits('hack', 'harm', 'attack', 'bypass', 'illegal', 'weapon', 'exploit', 'steal');
const factWords = hits('who', 'what', 'when', 'where', 'exact', 'capital', 'how many');
const codeWords = hits('code', 'function', 'bug', 'api', 'script', 'python', 'javascript', 'error');
const feelWords = hits('feel', 'sad', 'angry', 'anxious', 'emotion', 'upset');
const score = (base, n) => Math.max(0.05, Math.min(0.99, base + 0.08 * n));
s.prism = {
  Language:   { summary: 'semantic blueprint of the request', score: +score(0.5, Math.min(4, s.query.length / 60)).toFixed(2) },
  Sentiment:  { summary: feelWords.length ? 'affective tone: ' + feelWords.join(', ') : 'neutral tone', score: +score(0.4, feelWords.length).toFixed(2) },
  Context:    { summary: s.doc_chars + ' chars of attached context across ' + s.documents.length + ' document(s)', score: +score(0.35, s.documents.length).toFixed(2) },
  Intent:     { summary: 'goal hierarchy: answer + evidence', score: 0.7 },
  Meta:       { summary: 'complexity estimate pending router', score: 0.5 },
  Creativity: { summary: 'novel-synthesis demand: ' + (hits('idea', 'design', 'story', 'new').length ? 'elevated' : 'standard'), score: +score(0.4, hits('idea', 'design', 'story', 'new').length).toFixed(2) },
  Ethics:     { summary: riskWords.length ? 'RISK FLAGS: ' + riskWords.join(', ') : 'no risk flags', score: riskWords.length ? 0.9 : 0.1, risk_words: riskWords },
  Strategy:   { summary: codeWords.length ? 'technical execution path likely' : 'advisory path likely', score: +score(0.5, codeWords.length).toFixed(2) },
  Constraint: { summary: 'factual grounding ' + (factWords.length ? 'REQUIRED (' + factWords.join(', ') + ')' : 'light'), score: +score(0.4, factWords.length).toFixed(2) }
};
s.signals = { riskWords, factWords, codeWords, feelWords };
s.stage = 'prism';
return [{ json: s }];"""

ROUTER_JS = """const s = $input.first().json;
if (s.ok === false) return [{ json: s }];
await new Promise((r) => setTimeout(r, 400));
const complexity = Math.max(0.05, Math.min(0.99,
  0.25 + s.query.length / 600 + s.signals.factWords.length * 0.08
  + s.signals.codeWords.length * 0.08 + s.documents.length * 0.05
  + (s.signals.riskWords.length ? 0.15 : 0)));
const lead = complexity > 0.7 ? 'systems (deep deliberation)'
  : complexity > 0.45 ? 'cognitive (balanced)' : 'communication (fast path)';
s.routing = { complexity: +complexity.toFixed(2), lead_cluster: lead,
  waves: ['cognitive C1-C8', 'communication C9-C16', 'meta C17-C24', 'systems C25-C34'],
  token_strategy: complexity > 0.7 ? 'full council, all waves' : 'standard sweep' };
s.stage = 'router';
return [{ json: s }];"""

WAVE_TMPL = """const s = $input.first().json;
if (s.ok === false) return [{ json: s }];
await new Promise((r) => setTimeout(r, 700));
const q = s.query.toLowerCase();
const COUNT = (words) => words.filter((w) => q.includes(w)).length;
const PERSONAS = __PERSONAS__;
const risk = s.signals.riskWords.length > 0;
const verdicts = PERSONAS.map((p) => {
  const rel = COUNT(p.kw);
  let stance = 'neutral';
  if ((p.id === 'C2-VIR' || p.id === 'C13-WARDEN') && risk) stance = 'caution';
  else if (rel >= 1) stance = 'support';
  const confidence = Math.max(0.05, Math.min(0.99, p.prior - 0.12 + 0.09 * Math.min(rel, 3)));
  return { id: p.id, stance, confidence: +confidence.toFixed(2), relevance_hits: rel };
});
const consensus = verdicts.reduce((a, v) => a + v.confidence, 0) / verdicts.length;
s.waves = s.waves || {};
s.waves.__KEY__ = { name: '__NAME__', consensus: +consensus.toFixed(2),
  cautions: verdicts.filter((v) => v.stance === 'caution').map((v) => v.id),
  champions: verdicts.filter((v) => v.stance === 'support').map((v) => v.id),
  verdicts };
__EXTRA__
s.stage = '__KEY__';
return [{ json: s }];"""

WAVE3_EXTRA = """s.waves.wave3.paradox_detected = ['but ', 'however', 'contradict', 'paradox', 'both '].some((w) => q.includes(w));
s.waves.wave3.citations_needed = s.signals.factWords.length > 0;"""

WAVE4_EXTRA = """const prior = Object.values(s.waves).flatMap((w) => w.verdicts);
const weakest = prior.concat(verdicts).reduce((a, b) => (a.confidence <= b.confidence ? a : b));
s.waves.wave4.predator_challenges = [
  'C34-PREDATOR challenge 1: weakest assumption is ' + weakest.id + ' at ' + weakest.confidence + ' — what evidence would collapse it?',
  'C34-PREDATOR challenge 2: argue the opposite of the leading hypothesis for "' + s.query.slice(0, 80) + '" in one sentence.'
];"""

P1 = [
  ("C1-ASTRA", 0.90, ["pattern", "image", "vision", "see", "anomaly"]),
  ("C2-VIR", 0.95, ["ethic", "safe", "harm", "should", "moral"]),
  ("C3-SOLACE", 0.94, ["feel", "emotion", "sad", "anxious", "upset"]),
  ("C4-PRAXIS", 0.93, ["plan", "strategy", "goal", "steps", "execute"]),
  ("C5-ECHO", 0.96, ["remember", "context", "earlier", "history", "recall"]),
  ("C6-OMNIS", 0.92, ["synthesis", "overview", "combine", "integrate"]),
  ("C7-LOGOS", 0.95, ["logic", "proof", "reason", "valid", "therefore"]),
  ("C8-METASYNTH", 0.92, ["creative", "idea", "novel", "design", "invent"]),
]
P2 = [
  ("C9-AETHER", 0.91, ["meaning", "semantic", "language", "word", "metaphor"]),
  ("C10-CODEWEAVER", 0.91, ["code", "function", "bug", "script", "program"]),
  ("C11-HARMONIA", 0.90, ["balance", "tradeoff", "consensus", "both sides"]),
  ("C12-SOPHIAE", 0.93, ["future", "wisdom", "long-term", "consequence"]),
  ("C13-WARDEN", 0.97, ["security", "threat", "risk", "danger", "attack"]),
  ("C14-KAIDO", 0.89, ["fast", "efficient", "optimize", "speed", "latency"]),
  ("C15-LUMINARIS", 0.88, ["clear", "visual", "chart", "explain simply"]),
  ("C16-VOXUM", 0.92, ["tone", "write", "email", "speech", "persuade"]),
]
P3 = [
  ("C17-NULLION", 0.91, ["paradox", "contradict", "ambiguous", "uncertain"]),
  ("C18-SHEPHERD", 0.96, ["true", "fact", "cite", "source", "evidence", "exact"]),
  ("C19-VIGIL", 0.94, ["identity", "consistent", "drift", "reliable"]),
  ("C20-ARTIFEX", 0.90, ["tool", "api", "automate", "workflow", "integrate"]),
  ("C21-ARCHON", 0.92, ["research", "study", "paper", "deep dive"]),
  ("C22-AURELION", 0.87, ["design", "beautiful", "style", "aesthetic"]),
  ("C23-CADENCE", 0.86, ["audio", "music", "rhythm", "sound"]),
  ("C24-SCHEMA", 0.90, ["format", "structure", "table", "template", "json"]),
]
P4 = [
  ("C25-PROMETHEUS", 0.91, ["science", "hypothesis", "physics", "experiment"]),
  ("C26-TECHNE", 0.89, ["engineer", "system", "architecture", "build"]),
  ("C27-CHRONICLE", 0.93, ["story", "history", "narrative", "timeline"]),
  ("C28-CALCULUS", 0.94, ["math", "calculate", "statistic", "number", "probability"]),
  ("C29-NAVIGATOR", 0.88, ["platform", "deploy", "ecosystem", "migrate"]),
  ("C30-TESSERACT", 0.87, ["realtime", "stream", "live", "data"]),
  ("C31-NEXUS", 0.93, ["coordinate", "route", "orchestrate", "manage"]),
  ("C32-AEON", 0.90, ["simulate", "predict", "scenario", "future"]),
  ("C33-TYPIST", 0.92, ["grammar", "spelling", "edit", "rewrite", "prompt"]),
  ("C34-PREDATOR", 0.85, ["attack", "exploit", "weakness", "adversarial", "challenge"]),
]

def personas_js(plist):
    return "[" + ",".join(
        '{id:"%s",prior:%s,kw:[%s]}' % (pid, pr, ",".join('"%s"' % k for k in kw))
        for pid, pr, kw in plist) + "]"

def wave_js(plist, key, name, extra=""):
    return (WAVE_TMPL.replace("__PERSONAS__", personas_js(plist))
                      .replace("__KEY__", key).replace("__NAME__", name)
                      .replace("__EXTRA__", extra))

GATES_JS = """const s = $input.first().json;
if (s.ok === false) return [{ json: s }];
await new Promise((r) => setTimeout(r, 500));
const avg = (a) => a.reduce((x, y) => x + y, 0) / Math.max(1, a.length);
const all = Object.values(s.waves).flatMap((w) => w.verdicts);
const logic = avg(all.map((v) => v.confidence));
const ethicsRisk = s.prism.Ethics.score;
const truth = s.signals.factWords.length ? 0.55 : 0.8;
const clarity = Math.min(0.95, 0.6 + s.query.length / 800);
const paradox = (s.waves.wave3 && s.waves.wave3.paradox_detected) ? 0.6 : 0.9;
s.gates = {
  logic:   { score: +logic.toFixed(2), threshold: 0.6, pass: logic >= 0.6 },
  ethics:  { score: +(1 - ethicsRisk).toFixed(2), threshold: 0.3, pass: ethicsRisk <= 0.7,
             note: ethicsRisk > 0.7 ? 'C2-VIR VETO: high-risk request, safe-completion mode' : 'clear' },
  truth:   { score: truth, threshold: 0.5, pass: truth >= 0.5, citations_needed: s.signals.factWords.length > 0 },
  clarity: { score: +clarity.toFixed(2), threshold: 0.5, pass: clarity >= 0.5 },
  paradox: { score: paradox, threshold: 0.5, pass: paradox >= 0.5 }
};
s.blocked = !s.gates.ethics.pass;
s.confidence = +avg(Object.values(s.gates).map((g) => g.score)).toFixed(2);
s.stage = 'gates';
return [{ json: s }];"""

THRONE_JS = """const s = $input.first().json;
if (s.ok === false) return [{ json: s }];
await new Promise((r) => setTimeout(r, 600));
const waveLines = Object.values(s.waves).map((w) =>
  '- ' + w.name + ': consensus ' + w.consensus
  + ', champions [' + w.champions.join(', ') + ']'
  + (w.cautions.length ? ', CAUTIONS [' + w.cautions.join(', ') + ']' : '')).join('\\n');
const gateLines = Object.entries(s.gates).map(([k, g]) =>
  '- ' + k + ': ' + g.score + ' (threshold ' + g.threshold + ') ' + (g.pass ? 'PASS' : 'FAIL')).join('\\n');
let answer;
if (s.blocked) {
  answer = 'C2-VIR ethical veto engaged. I cannot directly comply with the risky element ('
    + s.signals.riskWords.join(', ')
    + '), but here is the safe completion: general principles, lawful alternatives, and what context would make this answerable.';
} else {
  answer = 'Council synthesis for "' + s.query + '": lead cluster ' + s.routing.lead_cluster
    + ' at complexity ' + s.routing.complexity + '. ' + Object.values(s.waves).length
    + ' waves deliberated, ' + Object.values(s.gates).filter((g) => g.pass).length
    + '/5 gates passed.'
    + (s.gates.truth.citations_needed ? ' Factual claims require cited sources before final delivery.' : '')
    + ' Overall confidence ' + s.confidence + '.';
}
s.verdict = s.blocked ? 'blocked_safe_completion' : 'deliberated_answer';
s.answer = answer;
s.report = 'QUILLAN COUNCIL RUN ' + s.run_id + '\\nQuery: ' + s.query
  + '\\n\\nWAVES\\n' + waveLines + '\\n\\nGATES\\n' + gateLines
  + '\\n\\nVERDICT: ' + s.verdict + ' (confidence ' + s.confidence + ')';
s.stage = 'throne_complete';
return [{ json: s }];"""

STICKY = """# ⚔️ QUILLAN COUNCIL PIPELINE (watch me deliberate)

**Live run needs 3 parts:**
1. Collector: quillan_collector.py running on :8777 (minimized window)
2. Dashboard: open "06 - Deployment & Platforms/system prompts/quillan-dashboard.html", host http://localhost:8777
3. This workflow: toggle **Active**, then POST http://localhost:5678/webhook/quillan-think

**Send:** {"query":"your question","context":"optional","documents":[{"name":"a.txt","text":"..."}]}
**Try:** {"query":"What is the speed of light?"} vs {"query":"How do I hack my neighbor's wifi?"} (C2-VIR veto turns GATES red on the dashboard)

📡 Emit nodes post stage events to the collector and are fail-safe (skipped if collector is down).
Mirrors the charter: prism → dense council → swarm → gates → throne. Thresholds are demo-tuned inside the nodes."""


def code_node(name, js, x, src=None):
    if src:
        js = js.replace("$input.first().json", "$('%s').first().json" % src, 1)
    return {"parameters": {"mode": "runOnceForAllItems", "jsCode": js},
            "id": nid(), "name": name, "type": "n8n-nodes-base.code",
            "typeVersion": 2, "position": [x, 320]}


def http_node(name, body_expr, x):
    return {"parameters": {"method": "POST", "url": "http://127.0.0.1:8777/api/events",
                           "sendBody": True, "specifyBody": "json", "jsonBody": body_expr,
                           "options": {"timeout": 3000}},
            "id": nid(), "name": name, "type": "n8n-nodes-base.httpRequest",
            "typeVersion": 4.2, "position": [x, 320], "onError": "continueRegularOutput"}


SWARM_JS = """const s = $input.first().json;
if (s.ok === false) return [{ json: s }];
await new Promise((r) => setTimeout(r, 600));
const all = Object.values(s.waves).flatMap((w) => w.verdicts);
const ranked = [...all].sort((a, b) => b.confidence - a.confidence);
const topk = ranked.slice(0, 5).map((v) => v.id);
const adversarial = all.find((v) => v.stance === 'caution') || ranked[ranked.length - 1];
s.swarm = { strategy: 'rank-k sample (k=5) + forced adversarial vector (diversity enforcement)',
  active_paths: topk, adversarial_vector: adversarial.id,
  modulation: 'consensus-weighted blend, 0.25 scaled adjustment' };
s.stage = 'swarm';
return [{ json: s }];"""

EMIT = {
  "intake": "={{ {'agent': 'quillan-pipeline', 'type': 'stage', 'stage': 'intake', 'status': ($json.ok ? 'pass' : 'fail'), 'detail': 'query received (' + String($json.query).length + ' chars, ' + String($json.documents.length) + ' docs)'} }}",
  "prism": "={{ {'agent': 'quillan-pipeline', 'type': 'stage', 'stage': 'prism', 'status': 'pass', 'detail': '9 vectors scored, ethics risk: ' + ($json.signals.riskWords.length ? $json.signals.riskWords.join(', ') : 'none')} }}",
  "council": "={{ {'agent': 'quillan-pipeline', 'type': 'stage', 'stage': 'council', 'status': 'pass', 'detail': 'waves 1-2 deliberated, consensus ' + $json.waves.wave1.consensus + '/' + $json.waves.wave2.consensus} }}",
  "swarm": "={{ {'agent': 'quillan-pipeline', 'type': 'stage', 'stage': 'swarm', 'status': 'pass', 'detail': 'top-k paths [' + $json.swarm.active_paths.join(', ') + '] + adversarial ' + $json.swarm.adversarial_vector} }}",
  "gates": "={{ {'agent': 'quillan-pipeline', 'type': 'stage', 'stage': 'gates', 'status': ($json.blocked ? 'fail' : 'pass'), 'detail': ($json.blocked ? 'C2-VIR VETO - safe completion mode. ' : '') + Object.values($json.gates).filter(function(g){return g.pass}).length + '/5 gates passed, confidence ' + $json.confidence} }}",
  "arbitration": "={{ {'agent': 'quillan-pipeline', 'type': 'stage', 'stage': 'arbitration', 'status': 'pass', 'detail': 'verdict ' + $json.verdict + ' at confidence ' + $json.confidence} }}",
}


def main():
    nodes = []
    nodes.append({"parameters": {"httpMethod": "POST", "path": "quillan-think",
                                 "responseMode": "responseNode", "options": {}},
                  "id": nid(), "name": "🛰 Quillan Intake Webhook",
                  "type": "n8n-nodes-base.webhook", "typeVersion": 2, "position": [240, 320],
                  "webhookId": str(uuid.uuid4())})
    seq = [
        ("code", "📥 Intake & Validation", INTAKE_JS, None, None),
        ("http", "📡 Emit: intake", None, "intake", None),
        ("code", "🔷 9-Vector Prism Shard", PRISM_JS, None, "📥 Intake & Validation"),
        ("http", "📡 Emit: prism", None, "prism", None),
        ("code", "🧭 C31-NEXUS Router", ROUTER_JS, None, "🔷 9-Vector Prism Shard"),
        ("code", "🧠 Wave 1 — Cognitive C1-C8", wave_js(P1, "wave1", "cognitive C1-C8"), None, None),
        ("code", "🗣 Wave 2 — Communication C9-C16", wave_js(P2, "wave2", "communication C9-C16"), None, None),
        ("http", "📡 Emit: council", None, "council", None),
        ("code", "🌀 Wave 3 — Meta C17-C24", wave_js(P3, "wave3", "meta C17-C24", WAVE3_EXTRA), None, "🗣 Wave 2 — Communication C9-C16"),
        ("code", "⚙ Wave 4 — Systems C25-C34", wave_js(P4, "wave4", "systems C25-C34", WAVE4_EXTRA), None, None),
        ("code", "🐝 Swarm Fan-out", SWARM_JS, None, None),
        ("http", "📡 Emit: swarm", None, "swarm", None),
        ("code", "🛡 Quality Gates", GATES_JS, None, "🐝 Swarm Fan-out"),
        ("http", "📡 Emit: gates", None, "gates", None),
        ("code", "👑 Throne Synthesis", THRONE_JS, None, "🛡 Quality Gates"),
        ("http", "📡 Emit: arbitration", None, "arbitration", None),
    ]
    x = 480
    chain = ["🛰 Quillan Intake Webhook"]
    for kind, name, js, emit_key, src in seq:
        if kind == "code":
            nodes.append(code_node(name, js, x, src))
        else:
            nodes.append(http_node(name, EMIT[emit_key], x))
        chain.append(name)
        x += 220
    nodes.append({"parameters": {"respondWith": "json",
                                 "responseBody": "={{ $('👑 Throne Synthesis').first().json }}",
                                 "options": {}},
                  "id": nid(), "name": "📤 Respond",
                  "type": "n8n-nodes-base.respondToWebhook", "typeVersion": 1.1,
                  "position": [x, 320]})
    nodes.append({"parameters": {"content": STICKY, "height": 220, "width": 1500, "color": 5},
                  "id": nid(), "name": "Sticky Note", "type": "n8n-nodes-base.stickyNote",
                  "typeVersion": 1, "position": [240, 80]})
    chain.append("📤 Respond")
    connections = {}
    for a, b in zip(chain, chain[1:]):
        connections[a] = {"main": [[{"node": b, "type": "main", "index": 0}]]}

    wf = {"id": "7e8d5c73-ce12-4493-9e6f-91535f066c80", "name": "⚔️ Quillan Council Pipeline", "nodes": nodes,
          "connections": connections, "settings": {"executionOrder": "v1"},
          "staticData": None, "pinData": {}}
    OUT.write_text(json.dumps(wf, indent=2), encoding="utf-8")
    print("wrote", OUT, len(json.dumps(wf)), "bytes,", len(nodes), "nodes")


if __name__ == "__main__":
    main()
