"""Audit benchmark_100_per_section_results.json for harness defects (prompt echo, leakage)."""
import json

PATH = r"C:\02_QUILLAN\exports\benchmark_100_per_section_results.json"
ECHO_MARKERS = ("Example 1: Input:", "Available tool:", "User request:", "A precise, helpful AI assistant", "Write concise Python")

with open(PATH, encoding="utf-8") as f:
    data = json.load(f)

for model, m in data.items():
    print("=" * 70)
    print(model)
    by_cat = {}
    for log in m["detailed_logs"]:
        c = by_cat.setdefault(log["category"], {"n": 0, "echo": 0, "empty": 0, "echo_pass": 0})
        c["n"] += 1
        r = log["response"]
        echoed = any(k in r for k in ECHO_MARKERS)
        if echoed:
            c["echo"] += 1
            if log["passed"]:
                c["echo_pass"] += 1
        if not r.strip():
            c["empty"] += 1
    for cat, c in by_cat.items():
        print(f"  {cat:20s} n={c['n']} prompt_echo={c['echo']} (of which PASS={c['echo_pass']}) empty={c['empty']}")

# Show one raw sample per category for model A
m = data["Model A Flagship (Polymath v11)"]
seen = set()
for log in m["detailed_logs"]:
    if log["category"] in seen:
        continue
    seen.add(log["category"])
    print("-" * 70)
    print(log["category"], log["id"], "PASS" if log["passed"] else "FAIL")
    print(repr(log["response"][:400]))

