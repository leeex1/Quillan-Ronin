#!/usr/bin/env python3
"""Debate must read S34 accumulator, not the HTTP emit response in front of it."""
import json
import sys

sys.stdout.reconfigure(encoding="utf-8")
P = r"C:\02_QUILLAN\08 - Templates & Config\workflows\starter odea workflow.json"
wf = json.load(open(P, encoding="utf-8"))
for n in wf["nodes"]:
    if n["name"] == "Debate Round 2":
        js = n["parameters"]["jsCode"]
        old = "const incomingDebate = $input.first().json;"
        assert old in js, "debate head not found"
        n["parameters"]["jsCode"] = js.replace(
            old, "const incomingDebate = $('C34-PREDATOR-swarm').first().json;", 1)
        print("debate source fixed")
json.dump(wf, open(P, "w", encoding="utf-8"), indent=2)
print("saved")
