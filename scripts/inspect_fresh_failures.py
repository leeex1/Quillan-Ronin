import json

data = json.load(open('C:/02_QUILLAN/exports/fresh_held_out_benchmark_results.json', encoding='utf-8'))
for model, mdata in data.items():
    print('=' * 70)
    print(f"{model} - Total: {mdata.get('total_passed')}/{mdata.get('total_probes')} ({mdata.get('overall_accuracy')}%)")
    print('=' * 70)
    fails = [l for l in mdata.get('detailed_logs', []) if not l.get('passed')]
    print(f"Total Failures: {len(fails)}")
    for f in fails:
        resp = str(f.get('response', '')).replace('\n', ' ')[:90]
        print(f"  [{f.get('id')}] ({f.get('category')}) -> {resp}")
