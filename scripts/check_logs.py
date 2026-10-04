import json
from pathlib import Path
for name in ['oni_run_console.log','oni_train_log.jsonl']:
    p = Path(r"C:\02_QUILLAN\training_logs") / name
    if p.exists():
        lines = p.read_text(encoding='utf-8', errors='replace').splitlines()
        print(f"{name}: {len(lines)} lines")
        if lines:
            print(f"  last: {lines[-1][:180]}")
        if name.endswith('.jsonl') and len(lines) > 1:
            print(f"  second last: {lines[-2][:180]}")
