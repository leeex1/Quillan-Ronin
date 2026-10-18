#!/usr/bin/env python3
"""Drop the direct Router->Prepare link (Read path now serves both runs)."""
import json
import sys

sys.stdout.reconfigure(encoding="utf-8")
P = r"C:\02_QUILLAN\08 - Templates & Config\workflows\emfo-main.json"
wf = json.load(open(P, encoding="utf-8"))
w = wf[0] if isinstance(wf, list) else wf
conns = w["connections"]
ROUTER = "\U0001f9ed C31-NEXUS Router"
outs = conns.get(ROUTER, {}).get("main", [[[]]])[0]
conns[ROUTER] = {"main": [[t for t in outs
                           if t.get("node") != "Prepare 34 Assignments"]]}
print("router feeds:",
      [t["node"] for t in conns[ROUTER]["main"][0]])
open(P, "w", encoding="utf-8").write(
    json.dumps(w if not isinstance(wf, list) else [w], indent=2))
print("saved")
