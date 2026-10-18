#!/usr/bin/env python3
"""Add the Extract-wrapper unwrap to the canvas Prepare node."""
import json
import sys

sys.stdout.reconfigure(encoding="utf-8")
P = r"C:\02_QUILLAN\08 - Templates & Config\workflows\starter odea workflow.json"
wf = json.load(open(P, encoding="utf-8"))
for n in wf["nodes"]:
    if n["name"] == "Prepare 34 Assignments":
        js = n["parameters"]["jsCode"]
        old = "const data = $('Parse Identities JSON').first().json;"
        assert old in js, "prepare head not found"
        n["parameters"]["jsCode"] = js.replace(
            old,
            "const rawParse = $('Parse Identities JSON').first().json;\n"
            "const data = rawParse.data || rawParse;", 1)
        print("unwrap added")
json.dump(wf, open(P, "w", encoding="utf-8"), indent=2)
print("saved")
