import sys
import json
import random
from pathlib import Path
REPO = Path(r"C:\02_QUILLAN")
sys.path += [str(REPO / "03 - Training & Model"), str(REPO)]
import torch
from quillan_bpe_tokenizer import QuillanBPETokenizer
tok = QuillanBPETokenizer()

print("=== RAW instruct_train.jsonl rows ===")
with open(REPO / "training_data" / "hf_samurai" / "instruct_train.jsonl", encoding="utf-8") as f:
    lines = [l for l in f if l.strip()]
for i in random.Random(11).sample(range(len(lines)), 2):
    o = json.loads(lines[i])
    s = json.dumps(o)[:800]
    print(f"--- instruct row {i} ---")
    print(s)
    print()

print("=== DECODED slice rows (mini_slice_20k) ===")
d = torch.load(REPO / "training_data" / "mini_slice_20k.pt", map_location="cpu", weights_only=True)
x = d["input_ids"] if isinstance(d, dict) else d
print("shape:", tuple(x.shape))
for i in random.Random(7).sample(range(len(x)), 3):
    ids = [int(v) for v in x[i] if int(v) != 1][:150]
    print(f"--- slice row {i} ({len(ids)} real tokens) ---")
    print(repr(tok.decode(ids))[:700])
    print()
