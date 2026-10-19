#!/usr/bin/env python3
"""Relay MAIN canvas for readability. POSITIONS + zone stickies ONLY.
No params, connections, or logic touched.
Zones:
  y=300 main lane: trigger..prepare, then council snake (2 serpentine rows)
  y=750 tail lane: debate..throne..export
  y=1100 webhook sim lane (legacy demo, still runs)
  y=1500 parking lot: shells, stub file node, parked converter
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
print("loaded:", len(nodes), "nodes")


def place(name, x, y):
    if name in nodes:
        nodes[name]["position"] = [x, y]
    else:
        print("WARN missing:", name)


def sticky(name, content, x, y, w=340, h=90, color=4):
    if name in nodes:
        nodes[name]["parameters"]["content"] = content
        nodes[name]["position"] = [x, y]
        return
    nodes[name] = {
        "parameters": {"content": content, "height": h, "width": w,
                        "color": color},
        "id": str(uuid.uuid4()), "name": name,
        "type": "n8n-nodes-base.stickyNote", "typeVersion": 1,
        "position": [x, y]}

# ---- main lane y=300 ----
x = 240
for nm in ["When clicking \u2018Execute workflow\u2019", "Query Input",
           "Read Identities File", "Parse Identities JSON", "Emit Intake",
           "Prepare 34 Assignments"]:
    place(nm, x, 300)
    x += 220

# ---- council snake: 34 agents+swarms interleaved, serpentine ----
agents = [f"C{i}-{n}" for i, n in
          enumerate(["ASTRA", "VIR", "SOLACE", "PRAXIS", "ECHO", "OMNIS",
                     "LOGOS", "METASYNTH", "AETHER", "CODEWEAVER", "HARMONIA",
                     "SOPHIAE", "WARDEN", "KAIDO", "LUMINARIS", "VOXUM",
                     "NULLION", "SHEPHERD", "VIGIL", "ARTIFEX", "ARCHON",
                     "AURELION", "CADENCE", "SCHEMA", "PROMETHEUS", "TECHNE",
                     "CHRONICLE", "CALCULUS", "NAVIGATOR", "TESSERACT",
                     "NEXUS", "AEON", "TYPIST", "PREDATOR"], start=1)]
chain = []
for a in agents:
    chain += [a, a + "-swarm"]
assert all(c in nodes for c in chain), "chain node missing"
STEP = 200
X0 = 1560
for i, nm in enumerate(chain[:34]):
    place(nm, X0 + i * STEP, 200)
X1 = X0 + 33 * STEP  # 8160
for i, nm in enumerate(chain[34:]):
    place(nm, X1 - i * STEP, 500)

# ---- tail lane y=750 ----
x = 1560
for nm in ["Debate Round 2", "Emit Council", "C0-QUILLAN Throne",
           "Emit Kill-Checks", "Post Kill-Checks", "Recollect Throne",
           "Emit Arbitration", "Aggregate", "Code in Python",
           "Code in Python1", "Convert to File1"]:
    place(nm, x, 750)
    x += 220

# ---- webhook sim lane y=1100 ----
x = 240
for nm in ["\U0001f6f0 Quillan Intake Webhook", "\U0001f4e5 Intake & Validation",
           "\U0001f4e1 Emit: intake", "\U0001f537 9-Vector Prism Shard",
           "\U0001f4e1 Emit: prism", "\U0001f9ed C31-NEXUS Router",
           "\U0001f9e0 Wave 1 \u2014 Cognitive C1-C8",
           "\U0001f5e3 Wave 2 \u2014 Communication C9-C16",
           "\U0001f4e1 Emit: council",
           "\U0001f300 Wave 3 \u2014 Meta C17-C24",
           "\u2699 Wave 4 \u2014 Systems C25-C34",
           "\U0001f41d Swarm Fan-out", "\U0001f4e1 Emit: swarm",
           "\U0001f6e1 Quality Gates", "\U0001f4e1 Emit: gates",
           "\U0001f44d Throne Synthesis", "\U0001f4e1 Emit: arbitration",
           "\U0001f4e4 Respond"]:
    place(nm, x, 1100)
    x += 220

# ---- parking lot y=1500 ----
shells = (["AI Agent"] + [f"AI Agent{i}" for i in range(1, 34)])
ok_shells = [s for s in shells if s in nodes]
print("parking shells:", len(ok_shells))
for i, nm in enumerate(ok_shells):
    place(nm, 240 + (i % 9) * 190, 1500 + (i // 9) * 200)
place("Read/Write Files from Disk", 240 + 9 * 190, 1500)
place("Convert to File", 240 + 9 * 190, 1700)

# ---- zone stickies ----
sticky("Zone: intake", "1 - INTAKE\nTrigger -> query -> identities file. One run starts here.",
       240, 140, color=5)
sticky("Zone: council", "2 - COUNCIL DELIBERATION\nFollow the snake: 34 agents x 34 swarms, left to right, top row then bottom.",
       1560, 60, w=520, color=5)
sticky("Zone: throne", "3 - THRONE + EXPORT\nDebate -> gates -> verdict -> verdict file.",
       1560, 610, color=5)
sticky("Zone: webhook", "4 - WEBHOOK SIM (legacy demo)\nSelf-contained pipeline demo. Also feeds the council via Router.",
       240, 960, color=3)
sticky("Zone: parking", "5 - PARKING LOT\n35 live-agent shells (need LLM brains) + spare parts. Unwired on purpose.",
       240, 1360, w=520, color=2)
# refresh the export sticky (it went live)
if "Sticky TODO export" in nodes:
    nodes["Sticky TODO export"]["parameters"]["content"] = (
        "EXPORT LEG: LIVE\nThrone verdict -> verdict_file text. Runs every cycle.")
    nodes["Sticky TODO export"]["position"] = [3400, 610]

P = r"C:\02_QUILLAN\08 - Templates & Config\workflows\emfo-layout.json"
open(P, "w", encoding="utf-8").write(json.dumps(
    {"id": "EmfoGhyIzSrktmIe", "name": "Quillan Sovereign \u2014 MAIN",
     "nodes": list(nodes.values()), "connections": conns,
     "settings": {"executionOrder": "v1"}, "staticData": None, "pinData": {}},
    indent=2))
print("saved layout file,", len(nodes), "nodes")
