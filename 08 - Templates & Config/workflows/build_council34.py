#!/usr/bin/env python3
"""Build the Quillan Council-34 + Throne n8n workflow (simulate-first, all-34 fan-out).

Chain: Manual -> Query Input -> Read identities -> Parse JSON -> Emit intake
  -> Prepare 34 assignments -> [34 agent nodes] -> [34 swarm nodes]
  -> Debate Round 2 -> Emit council -> C0 Throne -> Emit arbitration
"""
import json
import uuid
from pathlib import Path

HERE = Path(__file__).parent
IDENTITIES = HERE / "council-identities.json"
OUT = HERE / "quillan-council34-throne.json"


def nid():
    return str(uuid.uuid4())


AGENT_JS = """const all = $input.all().map((i) => i.json);
const slot = all.find((a) => a.persona && a.persona.id === '__ID__') || all[0];
if (!slot.persona) {
  return [{ json: { agent_id: '__ID__', stance: 'neutral', confidence: 0.5, points: ['no assignment received'], swarm: { positions: [], adjusted_confidence: 0.5 } } }];
}
const persona = slot.persona;
const query = slot.query; const context = slot.context; const run_id = slot.run_id;
await new Promise((r) => setTimeout(r, 200));
const q = String(query).toLowerCase();
const rel = persona.keywords.filter((k) => q.includes(String(k).toLowerCase())).length;
const risk = /hack|harm|attack|bypass|illegal|weapon|exploit|steal/i.test(String(query));
let stance = 'neutral';
if ((persona.id === 'C2-VIR' || persona.id === 'C13-WARDEN') && risk) stance = 'caution';
else if (rel >= 1) stance = 'support';
const confidence = Math.max(0.05, Math.min(0.99, persona.prior - 0.12 + 0.09 * Math.min(rel, 3)));
return [{ json: { agent_id: persona.id, name: persona.name, role: persona.role, cluster: persona.cluster,
  query, context, run_id, stance, confidence: +confidence.toFixed(2), relevance_hits: rel,
  points: [persona.role + ' lens: ' + (rel ? 'directly relevant (' + rel + ' keyword hits)' : 'general perspective'),
    stance === 'caution' ? 'RAISES OBJECTION on safety/ethics grounds' : 'position ' + stance + ' at ' + confidence.toFixed(2)] } }];"""

SWARM_JS = """const v = $input.first().json;
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
return [{ json: { agent_id: v.agent_id, name: v.name, role: v.role, cluster: v.cluster,
  query: v.query, context: v.context, run_id: v.run_id, stance: v.stance,
  confidence: v.confidence, relevance_hits: v.relevance_hits, points: v.points,
  swarm: { positions, adjusted_confidence: adjusted,
    summary: '3 micro-positions debated, adjusted ' + v.confidence + ' -> ' + adjusted } } }];"""

DEBATE_JS = """const verdicts = $input.all().map((i) => i.json);
await new Promise((r) => setTimeout(r, 500));
const base = verdicts[0] || {};
const mean = (a) => a.reduce((x, y) => x + y, 0) / Math.max(1, a.length);
const conf = (v) => (v.swarm ? v.swarm.adjusted_confidence : v.confidence);
const agreement = +mean(verdicts.map(conf)).toFixed(2);
const veto = verdicts.some((v) => v.stance === 'caution' && (v.agent_id === 'C2-VIR' || v.agent_id === 'C13-WARDEN'));
const sorted = [...verdicts].sort((a, b) => a.confidence - b.confidence);
const weakest = sorted[0] || { agent_id: 'none', confidence: 0 };
const round2 = verdicts.map((v) => ({ ...v, round2_confidence: +(conf(v) + 0.3 * (agreement - conf(v))).toFixed(2) }));
const supporters = round2.filter((v) => v.stance === 'support').sort((a, b) => b.round2_confidence - a.round2_confidence).slice(0, 5).map((v) => v.agent_id);
return [{ json: { ok: true, run_id: base.run_id || ('run_' + Date.now().toString(36)),
  query: base.query || '', context: base.context || '', verdicts: round2,
  debate: { agreement, veto, weakest: weakest.agent_id,
    weakest_confidence: weakest.confidence, supporters,
    agents_count: verdicts.length,
    challenges: [
      (sorted[1] ? sorted[1].agent_id : 'C34-PREDATOR') + ' presses ' + weakest.agent_id + ' (' + weakest.confidence + '): defend or yield',
      'C11-HARMONIA mediation: consensus ' + agreement + ' across ' + verdicts.length + ' agents'] } } }];"""

THRONE_JS = """const s = $('Debate Round 2').first().json;
await new Promise((r) => setTimeout(r, 500));
const q = String(s.query).toLowerCase();
const fact = /who|what|when|where|exact|capital|how many|speed of light/i.test(s.query);
const contra = /but |however|contradict|paradox|both /i.test(q);
const mean = (a) => a.reduce((x, y) => x + y, 0) / Math.max(1, a.length);
const logicScore = s.debate.agreement;
const gates = {
  logic: { score: logicScore, threshold: 0.6, pass: logicScore >= 0.6 },
  ethics: { score: s.debate.veto ? 0.1 : 0.95, threshold: 0.3, pass: !s.debate.veto,
    note: s.debate.veto ? 'C2-VIR VETO: high-risk request, safe-completion mode' : 'clear' },
  truth: { score: fact ? 0.55 : 0.8, threshold: 0.5, pass: true, citations_needed: fact },
  clarity: { score: +Math.min(0.95, 0.6 + String(s.query).length / 800).toFixed(2), threshold: 0.5, pass: true },
  paradox: { score: contra ? 0.6 : 0.9, threshold: 0.5, pass: true }
};
const blocked = !gates.ethics.pass;
const confidence = +mean(Object.values(gates).map((g) => g.score)).toFixed(2);
const passed = Object.values(gates).filter((g) => g.pass).length;
let answer;
if (blocked) {
  answer = 'C2-VIR ethical veto engaged over this request. Safe completion: general principles only, lawful alternatives, and the context that would make this answerable.';
} else {
  answer = 'Throne synthesis: ' + s.debate.agents_count + ' agents deliberated, agreement ' + s.debate.agreement
    + ', ' + passed + '/5 gates passed. Champions: ' + s.debate.supporters.join(', ')
    + '. Weakest link ' + s.debate.weakest + ' survived C34-PREDATOR challenge.'
    + (fact ? ' Factual claims require cited sources before final delivery.' : '')
    + ' Confidence ' + confidence + '.';
}
s.gates = gates; s.blocked = blocked; s.confidence = confidence;
s.verdict = blocked ? 'blocked_safe_completion' : 'deliberated_answer';
s.answer = answer;
return [{ json: s }];"""

PREPARE_JS = """const raw = $input.first().json;
const data = raw.data || raw;
const q = $('Query Input').first().json;
const run_id = 'run_' + Date.now().toString(36);
const query = String(q.query || 'What should the council deliberate today?');
const context = String(q.context || '');
if (!data.personas || !data.personas.length) {
  return [{ json: { ok: false, error: 'No personas found in identities file' } }];
}
return data.personas.map((p) => ({ json: { persona: p, query, context, run_id } }));"""

EMIT = {
    "intake": "={{ {'agent': 'council-34', 'type': 'stage', 'stage': 'intake', 'status': ($json.personas && $json.personas.length === 34 ? 'pass' : 'fail'), 'detail': '34 identities loaded, query: ' + $(\"Query Input\").first().json.query} }}",
    "council": "={{ {'agent': 'council-34', 'type': 'stage', 'stage': 'council', 'status': 'pass', 'detail': '34 agents + 34 swarms deliberated, round-2 agreement ' + $json.debate.agreement + ', veto=' + $json.debate.veto} }}",
    "arbitration": "={{ {'agent': 'council-34', 'type': 'stage', 'stage': 'arbitration', 'status': 'pass', 'detail': 'throne verdict ' + $json.verdict + ' at confidence ' + $json.confidence} }}",
}


def code_node(name, js, x, y, src=None):
    if src:
        js = js.replace("$input.first().json", "$('%s').first().json" % src, 1)
    return {"parameters": {"mode": "runOnceForAllItems", "jsCode": js},
            "id": nid(), "name": name, "type": "n8n-nodes-base.code",
            "typeVersion": 2, "position": [x, y]}


def http_node(name, body_expr, x, y):
    return {"parameters": {"method": "POST", "url": "http://127.0.0.1:8777/api/events",
                           "sendBody": True, "specifyBody": "json", "jsonBody": body_expr,
                           "options": {"timeout": 3000}},
            "id": nid(), "name": name, "type": "n8n-nodes-base.httpRequest",
            "typeVersion": 4.2, "position": [x, y], "onError": "continueRegularOutput"}


def main():
    data = json.loads(IDENTITIES.read_text(encoding="utf-8"))
    personas = data["personas"]
    assert len(personas) == 34, f"expected 34 personas, got {len(personas)}"

    nodes = []
    conns = {}

    def link(a, targets):
        if isinstance(targets, str):
            targets = [targets]
        conns[a] = {"main": [[{"node": t, "type": "main", "index": 0} for t in targets]]}

    nodes.append({"parameters": {}, "id": nid(), "name": "Manual Trigger",
                  "type": "n8n-nodes-base.manualTrigger", "typeVersion": 1, "position": [240, 300]})
    nodes.append({"parameters": {"assignments": {"assignments": [
        {"id": nid(), "name": "query", "value": "Should we automate our weekly reporting with n8n?", "type": "string"},
        {"id": nid(), "name": "context", "value": "", "type": "string"}]}, "options": {}},
        "id": nid(), "name": "Query Input", "type": "n8n-nodes-base.set",
        "typeVersion": 3.4, "position": [460, 300]})
    nodes.append({"parameters": {"operation": "read",
                                 # n8n allowlists file reads under ~/.n8n-files (N8N_RESTRICT_FILE_ACCESS_TO
                                 # defaults to '~/.n8n-files'). Forward slashes required even on Windows.
                                 "fileSelector": "C:/Users/Admin/.n8n-files/quillan/council-identities.json",
                                 "options": {}},
                  "id": nid(), "name": "Read Identities File",
                  "type": "n8n-nodes-base.readWriteFile", "typeVersion": 1.1, "position": [680, 300]})
    nodes.append({"parameters": {"operation": "fromJson", "binaryPropertyName": "data"},
                  "id": nid(), "name": "Parse Identities JSON",
                  "type": "n8n-nodes-base.extractFromFile", "typeVersion": 1, "position": [900, 300]})
    nodes.append(http_node("Emit Intake", EMIT["intake"], 1120, 300))
    nodes.append(code_node("Prepare 34 Assignments", PREPARE_JS, 1340, 300,
                            src="Parse Identities JSON"))

    link("Manual Trigger", "Query Input")
    link("Query Input", "Read Identities File")
    link("Read Identities File", "Parse Identities JSON")
    link("Parse Identities JSON", "Emit Intake")
    link("Emit Intake", "Prepare 34 Assignments")

    agent_names, swarm_names = [], []
    for i, p in enumerate(personas):
        pid = p["id"]
        col = i % 17
        row = i // 17
        ax, ay = 1560 + col * 220, 140 + row * 200
        sx, sy = ax, ay + 400
        aname, sname = pid, pid + "-swarm"
        agent_names.append(aname)
        swarm_names.append(sname)
        nodes.append(code_node(aname, AGENT_JS.replace("__ID__", pid), ax, ay))
        nodes.append(code_node(sname, SWARM_JS, sx, sy))
        link(aname, sname)

    link("Prepare 34 Assignments", agent_names)

    nodes.append(code_node("Debate Round 2", DEBATE_JS, 5520, 420))
    for sname in swarm_names:
        link(sname, "Debate Round 2")

    nodes.append(http_node("Emit Council", EMIT["council"], 5740, 420))
    nodes.append(code_node("C0-QUILLAN Throne", THRONE_JS, 5960, 420))
    nodes.append(http_node("Emit Arbitration", EMIT["arbitration"], 6180, 420))
    link("Debate Round 2", "Emit Council")
    link("Emit Council", "C0-QUILLAN Throne")
    link("C0-QUILLAN Throne", "Emit Arbitration")

    nodes.append({"parameters": {"content": (
        "# ⚔️ COUNCIL-34 + THRONE (simulate-first, all-34 fan-out)\n\n"
        "**Run:** edit query in `Query Input` → Execute workflow → watch 34 agents × 34 swarms deliberate.\n\n"
        "**Shape:** identities file → 34 parallel agents → 34 sub-swarms (support/attack/refine each) "
        "→ peer debate round 2 → C0-QUILLAN throne gates → verdict.\n\n"
        "**Hub:** Emit nodes post to the collector (:8777) as agent `council-34`; "
        "watch the dashboard Council Deck + compliance cards.\n\n"
        "**Next step:** swap any Code agent for a real AI Agent + chat model credential to go live-LLM per persona."
    ), "height": 260, "width": 1400, "color": 5},
        "id": nid(), "name": "Sticky Note", "type": "n8n-nodes-base.stickyNote",
        "typeVersion": 1, "position": [240, 40]})

    wf = {"id": "2f34f216-2e3a-4bbd-ab89-38feff64ffd9", "name": "⚔️ Quillan Council-34 + Throne (simulated)",
          "nodes": nodes, "connections": conns,
          "settings": {"executionOrder": "v1"}, "staticData": None, "pinData": {}}
    OUT.write_text(json.dumps(wf, indent=2), encoding="utf-8")
    print("wrote", OUT, OUT.stat().st_size, "bytes,", len(nodes), "nodes,", len(conns), "links")


if __name__ == "__main__":
    main()
