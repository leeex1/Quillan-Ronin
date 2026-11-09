import json

with open(r"C:\02_QUILLAN\exports\benchmark_100_per_section_results.json", "r", encoding="utf-8") as f:
    data = json.load(f)

for model, mdata in data.items():
    print("=" * 80)
    print(f"{model}: {mdata.get('overall_accuracy')}% (95% CI: {mdata.get('ci_95')}) [{mdata.get('total_passed')}/{mdata.get('total_probes')}] in {mdata.get('elapsed_seconds')}s")
    print("=" * 80)
    for cat, cdata in mdata.get('category_summary', {}).items():
        print(f"  * {cat:22s}: {cdata.get('passed'):3d}/{cdata.get('total'):3d} ({cdata.get('accuracy'):5.1f}%, +/-{cdata.get('margin_of_error'):.1f}%) CI: {cdata.get('ci_95')}")

