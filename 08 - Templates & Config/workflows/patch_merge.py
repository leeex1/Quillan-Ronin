#!/usr/bin/env python3
"""Merge the two islands: webhook pipeline Router -> council Prepare.
Prepare prefers webhook Intake state, falls back to manual Query Input.
Backup first.
"""
import json
import shutil
import sys

sys.stdout.reconfigure(encoding="utf-8")
P = r"C:\02_QUILLAN\08 - Templates & Config\workflows\emfo-main.json"
shutil.copy(P, P + ".bak-merge")
print("backup written")

wf = json.load(open(P, encoding="utf-8"))
w = wf[0] if isinstance(wf, list) else wf
nodes = {n["name"]: n for n in w["nodes"]}
conns = w["connections"]

ROUTER = "\U0001f9ed C31-NEXUS Router"
INTAKE = "\U0001f4e5 Intake & Validation"
assert ROUTER in nodes and INTAKE in nodes, "pipeline nodes missing"

# 1. dual-source prepare
prep = nodes["Prepare 34 Assignments"]["parameters"]["jsCode"]
old_head = prep.split("\n", 4)
print("prepare head now:", old_head[0][:80])
new_head = """const rawParse = $('Parse Identities JSON').first().json;
const data = rawParse.data || rawParse;
let q = null, qi = null;
try { q = $('INTAKEPH').first().json; } catch (e) {}
try { qi = $('Query Input').first().json; } catch (e) {}
const src = (q && q.query) ? q : qi;
const run_id = (src && src.run_id) || ('run_' + Date.now().toString(36));
const query = String((src && src.query) || 'What should the council deliberate today?');
const context = String((src && src.context) || '');""".replace("INTAKEPH", INTAKE)
lines = prep.split("\n")
# drop old header lines (rawParse/data/q/qi/run_id/query/context) up to the personas guard
cut = 0
for i, ln in enumerate(lines):
    if "if (!data.personas" in ln:
        cut = i
        break
assert cut > 0, "personas guard not found"
rest = "\n".join(lines[cut:])
nodes["Prepare 34 Assignments"]["parameters"]["jsCode"] = new_head + "\n" + rest
print("prepare dual-sourced")

# 2. router feeds prepare (keeps existing router->wave link too)
outs = conns.get(ROUTER, {}).get("main", [[]])[0]
print("router currently feeds:", [t["node"] for t in outs])
if not any(t["node"] == "Prepare 34 Assignments" for t in outs):
    outs.append({"node": "Prepare 34 Assignments", "type": "main", "index": 0})
    conns[ROUTER] = {"main": [outs]}
    print("router -> prepare linked")

out = w if not isinstance(wf, list) else [w]
open(P, "w", encoding="utf-8").write(json.dumps(out, indent=2))
print("saved", len(w["nodes"]), "nodes")
