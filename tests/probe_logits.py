import torch
from tokenizers import Tokenizer
import sys

sys.path.insert(0, r'C:\02_QUILLAN\03 - Training & Model\scripts')
sys.path.insert(0, r'C:\02_QUILLAN\02 - Knowledge Foundation\knowledge\legacy\00 - Meta\oni')
from quillan_v5_4_oni import QuillanOniConfig, QuillanRoninOni

tok_path = r'C:\02_QUILLAN\02 - Knowledge Foundation\knowledge\legacy\00 - Meta\oni\tokenizer.json'
tok = Tokenizer.from_file(tok_path)

ckpt_path = r'C:\02_QUILLAN\checkpoints\checkpoints_oni\sft_runs\quillan_mini_0.6b_tripartite.pt'
ckpt = torch.load(ckpt_path, map_location="cpu")

import inspect
valid_keys = set(inspect.signature(QuillanOniConfig.__init__).parameters.keys())
cfg_dict = {k: v for k, v in ckpt["config"].items() if k in valid_keys}
cfg = QuillanOniConfig(**cfg_dict)
model = QuillanRoninOni(cfg).cpu().eval()
model.load_state_dict(ckpt["model_state_dict"], strict=False)

prompt = "Quillan is a sovereign cognitive agent. Its mission is"
enc = tok.encode(prompt).ids
x = torch.tensor([enc], dtype=torch.long)

with torch.no_grad():
    out = model(x)
    logits = out[0] if isinstance(out, tuple) else out
    last_logits = logits[0, -1, :]
    probs = torch.softmax(last_logits, dim=-1)
    top_probs, top_ids = torch.topk(probs, 15)

print("Top 15 candidates for next token after prompt:")
for p, idx in zip(top_probs, top_ids):
    print(f"  Token {idx.item():5d} ('{tok.id_to_token(idx.item())}'): prob = {p.item():.4f}")
