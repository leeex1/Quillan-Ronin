#!/usr/bin/env python3
"""Convert the two Python pass-through nodes to JavaScript (no Py runner here)."""
import json
import sys

sys.stdout.reconfigure(encoding="utf-8")
P = r"C:\02_QUILLAN\08 - Templates & Config\workflows\emfo-main.json"
wf = json.load(open(P, encoding="utf-8"))
w = wf[0] if isinstance(wf, list) else wf
JS = ("const items = $input.all();\n"
      "items.forEach((item) => { item.json.my_new_field = 1; });\n"
      "return items;")
for n in w["nodes"]:
    if n["name"] in ("Code in Python", "Code in Python1"):
        n["parameters"] = {"mode": "runOnceForAllItems", "jsCode": JS}
        print("converted:", n["name"])
open(P, "w", encoding="utf-8").write(
    json.dumps(w if not isinstance(wf, list) else [w], indent=2))
print("saved")
