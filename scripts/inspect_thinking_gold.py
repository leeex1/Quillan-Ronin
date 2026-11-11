import json, glob

paths = glob.glob(r'C:\02_QUILLAN\**\sovereign_thinking_gold.jsonl', recursive=True)
if paths:
    with open(paths[0], 'r', encoding='utf-8') as f:
        lines = [json.loads(line) for line in f if line.strip()]
    print('Total samples:', len(lines))
    print('Keys of line 0:', lines[0].keys())
    for k, v in lines[0].items():
        if isinstance(v, str):
            print(f'{k}: {repr(v[:120])}')
        else:
            print(f'{k}: {type(v)}')
