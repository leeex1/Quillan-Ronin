#!/usr/bin/env python3
"""Invert sub-node links: model (source) -> agents (destinations).

n8n stores ai_* links with the SUB-NODE as source (data flows model->agent).
Our agent-keyed links validate as empty. One source, 35 targets.
"""
import json
import sys

sys.stdout.reconfigure(encoding="utf-8")
P = r"C:\02_QUILLAN\08 - Templates & Config\workflows\emfo-main.json"
wf = json.load(open(P, encoding="utf-8"))
w = wf[0] if isinstance(wf, list) else wf
conns = w["connections"]
MODEL = "OpenRouter Chat Model"

agents = [k for k in conns if conns[k].get("ai_languageModel")]
print("agent-keyed ai links found:", len(agents))
for k in agents:
    del conns[k]["ai_languageModel"]

shell_order = ["AI Agent"] + [f"AI Agent{i}" for i in range(1, 34)] + ["C0-QUILLAN \u00b7LIVE"]
missing = [s for s in shell_order if s not in [n["name"] for n in w["nodes"]]]
print("missing shells:", missing if missing else "none")
conns[MODEL] = {"ai_languageModel": [[
    {"node": s, "type": "ai_languageModel", "index": 0} for s in shell_order]]}
print("model-sourced link with", len(shell_order), "targets")
open(P, "w", encoding="utf-8").write(
    json.dumps(w if not isinstance(wf, list) else [w], indent=2))
print("saved")
