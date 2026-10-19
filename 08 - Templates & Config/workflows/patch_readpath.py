#!/usr/bin/env python3
"""Point the canvas Read node at the allowlisted identities copy."""
import json
import shutil
import sys

sys.stdout.reconfigure(encoding="utf-8")
P = r"C:\02_QUILLAN\08 - Templates & Config\workflows\starter odea workflow.json"
wf = json.load(open(P, encoding="utf-8"))
for n in wf["nodes"]:
    if n["name"] == "Read Identities File":
        n["parameters"]["operation"] = "read"
        n["parameters"]["fileSelector"] = (
            "C:/Users/Admin/.n8n-files/quillan/council-identities.json")
        n["parameters"]["options"] = {}
        print("read path fixed")
json.dump(wf, open(P, "w", encoding="utf-8"), indent=2)
print("saved")
