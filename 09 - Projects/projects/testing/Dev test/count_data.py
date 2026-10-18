import json
from pathlib import Path

data_dir = Path(r"C:\02_QUILLAN\training_data")
for f in sorted(data_dir.glob("*.jsonl")):
    count = sum(1 for line in open(f, encoding="utf-8") if line.strip())
    print(f"  {f.name}: {count} samples")
