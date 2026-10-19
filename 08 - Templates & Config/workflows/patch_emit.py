#!/usr/bin/env python3
"""Enrich council emit details with persona data for the live mesh."""
import json
import sys

sys.stdout.reconfigure(encoding="utf-8")
P = r"C:\02_QUILLAN\08 - Templates & Config\workflows\starter odea workflow.json"
wf = json.load(open(P, encoding="utf-8"))
for n in wf["nodes"]:
    if n["name"] == "Emit Council":
        body = n["parameters"]["jsonBody"]
        old = "' + $json.debate.agreement + ', veto=' + $json.debate.veto}"
        assert old in body, "council body not found"
        new = ("' + $json.debate.agreement + ', veto=' + $json.debate.veto "
               "+ ', supporters [' + $json.debate.supporters.join(', ') + '], weakest ' + $json.debate.weakest}")
        n["parameters"]["jsonBody"] = body.replace(old, new, 1)
        print("council emit enriched")
json.dump(wf, open(P, "w", encoding="utf-8"), indent=2)
print("saved")
