#!/usr/bin/env python3
"""Fix intake status check: Parse wraps payload as {data: {...}}."""
import json
import sys

sys.stdout.reconfigure(encoding="utf-8")
P = r"C:\02_QUILLAN\08 - Templates & Config\workflows\starter odea workflow.json"
wf = json.load(open(P, encoding="utf-8"))
for n in wf["nodes"]:
    if n["name"] == "Emit Intake":
        body = n["parameters"]["jsonBody"]
        old = "($json.personas && $json.personas.length === 34 ? 'pass' : 'fail')"
        assert old in body, "intake status not found"
        n["parameters"]["jsonBody"] = body.replace(
            old,
            "($json.data && $json.data.personas && $json.data.personas.length === 34 ? 'pass' : 'fail')",
            1)
        print("intake status fixed")
json.dump(wf, open(P, "w", encoding="utf-8"), indent=2)
print("saved")
