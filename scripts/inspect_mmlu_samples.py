import json

with open(r"C:\02_QUILLAN\exports\benchmark_100_per_section_results.json", "r", encoding="utf-8") as f:
    data = json.load(f)

for mname in ["Model A Flagship (Polymath v11)", "Model B Mini (0.5B Sovereign)", "Model C Dragon (1.0B Hybrid)"]:
    m = data[mname]
    mmlu_logs = [l for l in m["detailed_logs"] if l["category"] == "MMLU_Academic"]
    print(f"=== {mname} MMLU FIRST 5 SAMPLES ===")
    for i in range(5):
        resp = mmlu_logs[i]["response"].replace("\n", " ")[:80]
        passed = mmlu_logs[i]["passed"]
        print(f"  [{i}] PASS={passed} | Resp: {resp}")

