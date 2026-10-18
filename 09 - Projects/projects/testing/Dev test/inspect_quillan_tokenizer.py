import os
import sys
import torch
import json
from pathlib import Path

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

REPO_ROOT = Path(r"C:\02_QUILLAN")

print("==================================================================")
print("   👑 INSPECTING QUILLAN CUSTOM TOKENIZER & PRE-TOKENIZED DATA")
print("==================================================================")

tokenizer_json = REPO_ROOT / "training_data" / "tokenizer.json"
if tokenizer_json.exists():
    with open(tokenizer_json, "r", encoding="utf-8") as f:
        tok_data = json.load(f)
    vocab = tok_data.get("model", {}).get("vocab", {})
    print(f"[+] Custom Tokenizer Found: {tokenizer_json}")
    print(f"    Total Vocab Size: {len(vocab)}")
    sample_items = list(vocab.items())[:10]
    print(f"    Sample Vocab Entries: {sample_items}")

pt_files = [
    REPO_ROOT / "training_data" / "Quillan_Clean_Reasoning_Gold_Dataset.jsonl",
    REPO_ROOT / "training_data" / "Quillan_Hyper_Tune_Gold_Dataset.jsonl",
    REPO_ROOT / "training_data" / "quillan_12mb_training_dataset.pt"
]

for pt_path in pt_files:
    if pt_path.exists():
        print(f"\n[*] Inspecting: {pt_path.name}")
        if pt_path.suffix == ".pt":
            data = torch.load(pt_path, weights_only=False)
            if isinstance(data, torch.Tensor):
                print(f"    Tensor Shape: {data.shape}, Dtype: {data.dtype}")
            elif isinstance(data, dict):
                print(f"    Dict Keys: {list(data.keys())}")
            elif isinstance(data, list):
                print(f"    List Length: {len(data)}, Sample Type: {type(data[0])}")
        else:
            with open(pt_path, "r", encoding="utf-8") as f:
                first_lines = [f.readline().strip() for _ in range(2)]
            print(f"    Sample JSONL Line 1: {first_lines[0][:120]}...")

print("\n[+] Inspection Complete!")
