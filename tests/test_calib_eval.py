import os, sys, time, torch
import torch.nn.functional as F
from tokenizers import Tokenizer

sys.path.insert(0, r'C:\02_QUILLAN\03 - Training & Model\scripts')
sys.path.insert(0, r'C:\02_QUILLAN\02 - Knowledge Foundation\knowledge\legacy\00 - Meta\oni')
from quillan_v5_4_oni import QuillanOniConfig, QuillanRoninOni

tok_path = r'C:\02_QUILLAN\02 - Knowledge Foundation\knowledge\legacy\00 - Meta\oni\tokenizer.json'
tok = Tokenizer.from_file(tok_path)

ckpt_path = r'C:\02_QUILLAN\checkpoints\checkpoints_oni\sft_runs\quillan_mini_0.6b_tripartite.pt'
ckpt = torch.load(ckpt_path, map_location="cpu")

import inspect
valid_keys = set(inspect.signature(QuillanOniConfig.__init__).parameters.keys())
filtered_cfg = {k: v for k, v in ckpt["config"].items() if k in valid_keys}
cfg = QuillanOniConfig(**filtered_cfg)

model = QuillanRoninOni(cfg).cpu().eval()
model.load_state_dict(ckpt["model_state_dict"], strict=False)

prompt = "Quillan is a sovereign cognitive agent. Its mission is to"
enc = tok.encode(prompt).ids
x = torch.tensor([enc], dtype=torch.long)

with torch.no_grad():
    out = model(x, deliberation=False)
    logits = out[0] if isinstance(out, tuple) else out
    last_logits = logits[0, -1, :].clone().float()

print("Logits shape:", last_logits.shape)
print("Logits min/max/mean/nan:", last_logits.min().item(), last_logits.max().item(), last_logits.mean().item(), torch.isnan(last_logits).any().item())

probs = F.softmax(last_logits, dim=-1)
print("Probs min/max/mean/nan:", probs.min().item(), probs.max().item(), probs.mean().item(), torch.isnan(probs).any().item())

top5 = torch.topk(probs, 5)
print("Top 5 tokens:")
for p, idx in zip(top5.values, top5.indices):
    print(f"  {p.item()*100:5.2f}% | ID {idx.item():5d} | '{tok.decode([idx.item()])}'")

print("\nGenerating 20 tokens with fast Top-K sampling...")
gen = list(enc)
t0 = time.time()
for _ in range(20):
    x = torch.tensor([gen[-128:]], dtype=torch.long)
    out = model(x, deliberation=False)
    logits = out[0] if isinstance(out, tuple) else out
    next_logits = logits[0, -1, :].clone().float()
    
    # Standard repetition penalty
    for prev_tok in set(gen[-32:]):
        if next_logits[prev_tok] > 0:
            next_logits[prev_tok] /= 1.25
        else:
            next_logits[prev_tok] *= 1.25
            
    v, idxs = torch.topk(next_logits / 0.7, 40)
    top_probs = F.softmax(v, dim=-1)
    sample_idx = torch.multinomial(top_probs, 1).item()
    next_tok = idxs[sample_idx].item()
    gen.append(next_tok)
    if next_tok == tok.token_to_id("<|im_end|>") or next_tok == tok.token_to_id("<|endoftext|>"):
        break

elapsed = time.time() - t0
print(f"Generation took {elapsed:.2f}s ({len(gen) - len(enc)} tokens, { (len(gen) - len(enc))/max(1e-4, elapsed):.2f} tok/s)")
print("Result:\n" + tok.decode(gen))

