import os
import sys
import torch
from pathlib import Path

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

REPO_ROOT = Path(r"C:\02_QUILLAN")

print("==================================================================")
print("   👑 INSPECTING PRE-TOKENIZED DATASET TENSORS")
print("==================================================================")

pt_files = [
    REPO_ROOT / "training_data" / "quillan_12mb_training_dataset.pt",
    REPO_ROOT / "training_data" / "full_train.pt",
    REPO_ROOT / "training_data" / "quillan_corpus_CLEAN_V7.pt"
]

for pt_path in pt_files:
    if pt_path.exists():
        print(f"\n[*] File: {pt_path.name}")
        try:
            data = torch.load(pt_path, weights_only=False)
            if isinstance(data, torch.Tensor):
                print(f"    Type: Tensor, Shape: {data.shape}, Min Token ID: {data.min().item()}, Max Token ID: {data.max().item()}")
            elif isinstance(data, dict):
                print(f"    Type: Dict, Keys: {list(data.keys())}")
                for k, v in list(data.items())[:3]:
                    if isinstance(v, torch.Tensor):
                        print(f"      Key '{k}': Tensor Shape {v.shape}, Min: {v.min().item()}, Max: {v.max().item()}")
            elif isinstance(data, list):
                print(f"    Type: List, Length: {len(data)}, Element Type: {type(data[0])}")
        except Exception as e:
            print(f"    [Error loading]: {e}")

print("\n[+] Inspection Complete!")
