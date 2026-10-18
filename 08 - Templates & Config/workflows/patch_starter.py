#!/usr/bin/env python3
"""Surgical fixes for the merged starter workflow. Backup first, then patch."""
import json
import shutil
import sys

sys.stdout.reconfigure(encoding="utf-8")
P = r"C:\02_QUILLAN\08 - Templates & Config\workflows\starter odea workflow.json"
shutil.copy(P, P + ".bak-prethronefix")
print("backup written")

wf = json.load(open(P, encoding="utf-8"))

fixed = 0
for n in wf["nodes"]:
    if n["name"] == "C0-QUILLAN Throne":
        js = n["parameters"]["jsCode"]
        old = "$input.first().json"
        assert old in js, "throne source pattern not found"
        n["parameters"]["jsCode"] = js.replace(
            old, "$('Debate Round 2').first().json", 1)
        fixed += 1
print("throne patched:", fixed)

conns = wf["connections"]
print("--- Aggregate input from? ---")
for k, v in conns.items():
    for out in v.get("main", []):
        for t in out:
            if t["node"] == "Aggregate":
                print("source:", k)
print("--- dead AI Agent outgoing targets ---")
tg = set()
for k, v in conns.items():
    if k.startswith("AI Agent"):
        for out in v.get("main", []):
            for t in out:
                tg.add(t["node"])
print(sorted(tg))

json.dump(wf, open(P, "w", encoding="utf-8"), indent=2)
print("saved")
