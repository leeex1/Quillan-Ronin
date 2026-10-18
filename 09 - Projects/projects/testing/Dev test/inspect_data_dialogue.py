import os
import sys
import torch
import json
import tiktoken

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

REPO_ROOT = r"C:\02_QUILLAN"
DATA_DIR = os.path.join(REPO_ROOT, "training_data")

enc = tiktoken.get_encoding("gpt2")

print("=== CHECKING TOKEN ENCODINGS IN DATASETS ===")

pt_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".pt")]
for pf in pt_files[:3]:
    fpath = os.path.join(DATA_DIR, pf)
    data = torch.load(fpath, map_location="cpu", weights_only=False)
    if isinstance(data, torch.Tensor):
        sample_ids = data[:50].tolist()
        print(f"\n[PT File] {pf}:")
        print("Raw Token IDs:", sample_ids)
        print("Decoded Text:", enc.decode(sample_ids))
    elif isinstance(data, list) and len(data) > 0:
        sample = data[0]
        if isinstance(sample, dict):
            sample_ids = sample.get("input_ids", [])[:50]
        elif isinstance(sample, list):
            sample_ids = sample[:50]
        else:
            sample_ids = []
        if isinstance(sample_ids, torch.Tensor): sample_ids = sample_ids.tolist()
        print(f"\n[PT List File] {pf}:")
        print("Raw Token IDs:", sample_ids)
        if sample_ids:
            print("Decoded Text:", enc.decode(sample_ids))

jsonl_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".jsonl")]
for jf in jsonl_files[:3]:
    fpath = os.path.join(DATA_DIR, jf)
    print(f"\n[JSONL File] {jf}:")
    with open(fpath, "r", encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f):
            if i >= 2: break
            obj = json.loads(line.strip())
            if "text" in obj:
                txt = obj["text"][:100]
                print(f"  Line {i} raw text: {txt}")
                print(f"  Line {i} encoded: {enc.encode(txt)[:20]}")
            elif "input_ids" in obj:
                toks = obj["input_ids"][:50]
                print(f"  Line {i} input_ids: {toks}")
                print(f"  Line {i} decoded: {enc.decode(toks)}")
