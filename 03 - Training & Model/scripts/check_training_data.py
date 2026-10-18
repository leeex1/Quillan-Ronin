import json, os, sys
from pathlib import Path

BASE = Path(r"C:\Users\Admin\Quillan-Ronin")
MODEL_DIR = BASE / "Quillan-v4.2-model" / "quillan_custom"
DATA_DIR = BASE / "training_data"

# Check what training data we have
print("=== TRAINING DATA INVENTORY ===")
jsonl_files = list(DATA_DIR.glob("*.jsonl"))
pt_files = list(DATA_DIR.glob("*.pt"))

for f in sorted(jsonl_files, key=lambda x: x.stat().st_size, reverse=True):
    sz = f.stat().st_size / 1024 / 1024
    # Count lines
    with open(f, 'r', encoding='utf-8', errors='ignore') as fh:
        lines = sum(1 for _ in fh)
    print(f"  {f.name}: {sz:.1f} MB, {lines:,} samples")

print(f"\n=== MODEL CHECKPOINT ===")
print(f"  Path: {MODEL_DIR}")
print(f"  SafeTensors: {(MODEL_DIR / 'model.safetensors').exists()}")
print(f"  Config: {(MODEL_DIR / 'config.json').exists()}")
print(f"  Tokenizer: {(MODEL_DIR / 'tokenizer.json').exists()}")

# Check tokenizer vocab
with open(MODEL_DIR / "tokenizer.json", 'r', encoding='utf-8') as f:
    tok = json.load(f)
added = tok.get('added_tokens', [])
base_vocab = len(tok.get('model', {}).get('vocab', {}))
print(f"  Base vocab: {base_vocab}")
print(f"  Added tokens: {len(added)}")
print(f"  Total vocab: {base_vocab + len(added)}")
