#!/usr/bin/env python3
"""Validate emfo-main.json: dangling refs + JS syntax."""
import json
import subprocess
import sys
import tempfile
import os

sys.stdout.reconfigure(encoding="utf-8")
wf = json.load(open(
    r"C:\02_QUILLAN\08 - Templates & Config\workflows\emfo-main.json",
    encoding="utf-8"))
w = wf[0] if isinstance(wf, list) else wf
names = set(n["name"] for n in w["nodes"])
print("nodes:", len(names))
bad = []
for k, v in w["connections"].items():
    if not isinstance(v, dict):
        bad.append((k, "non-dict conns value"))
        continue
    for o in v.get("main", []):
        if isinstance(o, dict):
            o = [o]
        for t in o:
            if isinstance(t, dict) and t.get("node") not in names:
                bad.append((k, t.get("node")))
print("dangling:", bad if bad else "none")
njs = jb = 0
for n in w["nodes"]:
    js = n.get("parameters", {}).get("jsCode")
    if not js:
        continue
    njs += 1
    with tempfile.NamedTemporaryFile("w", suffix=".mjs", delete=False,
                                      encoding="utf-8") as f:
        f.write(";(async () => {\n" + js + "\n})();")
        p = f.name
    r = subprocess.run(["node", "--check", p], capture_output=True, text=True)
    os.unlink(p)
    if r.returncode != 0:
        jb += 1
        print("FAIL", n["name"], r.stderr[:200])
print("JS:", njs, "BAD:", jb)
