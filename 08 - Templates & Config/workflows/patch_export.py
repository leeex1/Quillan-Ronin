#!/usr/bin/env python3
"""Make the export tail valid: shape verdict text, point toText at it."""
import json
import sys

sys.stdout.reconfigure(encoding="utf-8")
P = r"C:\02_QUILLAN\08 - Templates & Config\workflows\emfo-main.json"
wf = json.load(open(P, encoding="utf-8"))
w = wf[0] if isinstance(wf, list) else wf
nodes = {n["name"]: n for n in w["nodes"]}

SHAPE_JS = """const items = $input.all().map((i) => i.json);
const first = items[0] || {};
const th = first.verdict ? first : (first.data && first.data[0]) || first;
const lines = [
  'QUILLAN COUNCIL VERDICT',
  'verdict: ' + (th.verdict || 'unknown'),
  'confidence: ' + (th.confidence || '?'),
  'query: ' + (th.query || ''),
  '',
  String(th.answer || th.report || JSON.stringify(th).slice(0, 2000))
];
return [{ json: { verdict_file: lines.join('\\n'), run_id: th.run_id || '' } }];"""

for nm in ("Code in Python", "Code in Python1"):
    if nm in nodes:
        nodes[nm]["parameters"] = {
            "mode": "runOnceForAllItems", "jsCode": SHAPE_JS}
        print("shaped:", nm)

if "Convert to File1" in nodes:
    p = nodes["Convert to File1"]["parameters"]
    p["sourceProperty"] = "verdict_file"
    print("toText source set: verdict_file")

# Convert to File (toJson) mid-chain would wrap json into binary and break
# the text flow; point the chain straight: Code1 -> File1 already is direct.
# Ensure File (toJson) output feeds Code1 fine (binary passthrough keeps json).
# Straighten the tail: Code -> Code1 -> File1 (park the mid toJson converter,
# it wraps items in binary and hides the verdict text; node kept, unwired).
conns = w["connections"]
conns["Code in Python"] = {"main": [[{"node": "Code in Python1",
                                      "type": "main", "index": 0}]]}
if "Convert to File" in conns:
    conns["Convert to File"] = {"main": [[]]}
    print("mid toJson converter parked (kept on canvas)")
open(P, "w", encoding="utf-8").write(
    json.dumps(w if not isinstance(wf, list) else [w], indent=2))
print("saved")
