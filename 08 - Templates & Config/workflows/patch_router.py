#!/usr/bin/env python3
"""Router feeds the shared Read node: webhook runs get identities too."""
import json
import sys

sys.stdout.reconfigure(encoding="utf-8")
P = r"C:\02_QUILLAN\08 - Templates & Config\workflows\emfo-main.json"
wf = json.load(open(P, encoding="utf-8"))
w = wf[0] if isinstance(wf, list) else wf
conns = w["connections"]
ROUTER = "\U0001f9ed C31-NEXUS Router"
outs = conns.get(ROUTER, {}).get("main", [[[]]])[0]
print("router feeds:", [t["node"] for t in outs])
if not any(t["node"] == "Read Identities File" for t in outs):
    outs.append({"node": "Read Identities File", "type": "main", "index": 0})
    conns[ROUTER] = {"main": [outs]}
    print("router -> read linked")
open(P, "w", encoding="utf-8").write(
    json.dumps(w if not isinstance(wf, list) else [w], indent=2))
print("saved")
