#!/usr/bin/env python3
"""Connect the middle: park the dead trigger->file stub, feed export from throne."""
import json
import shutil
import sys

sys.stdout.reconfigure(encoding="utf-8")
P = r"C:\02_QUILLAN\08 - Templates & Config\workflows\emfo-main.json"
wf = json.load(open(P, encoding="utf-8"))
w = wf[0] if isinstance(wf, list) else wf
shutil.copy(P, P + ".bak-middle")
print("backup written")
nodes = {n["name"]: n for n in w["nodes"]}
conns = w["connections"]

for n in w["nodes"]:
    if n["name"] in ("Aggregate", "Code in Python", "Convert to File",
                     "Code in Python1", "Convert to File1"):
        print(n["name"], "|", n["type"],
              "| params:", json.dumps(n.get("parameters", {}))[:220])

# 1. park the middle stub trigger (it fires a broken file read on full runs)
parked = 0
for k in list(conns):
    if k == "Manual Trigger":
        conns[k] = {"main": [[]]}
        parked += 1
print("middle trigger parked:", parked)

# 2. throne feeds the export leg (keeps existing throne outputs too)
t = conns.get("C0-QUILLAN Throne", {}).get("main", [[]])[0]
print("throne currently feeds:", [x["node"] for x in t])
if not any(x["node"] == "Aggregate" for x in t):
    t.append({"node": "Aggregate", "type": "main", "index": 0})
    conns["C0-QUILLAN Throne"] = {"main": [t]}
    print("throne -> aggregate linked")

open(P, "w", encoding="utf-8").write(
    json.dumps(w if not isinstance(wf, list) else [w], indent=2))
print("saved")
