import os
import sys
import torch
import torch.nn.functional as F
import tiktoken
from pathlib import Path

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

REPO_ROOT = Path(r"C:\02_QUILLAN")
sys.path.insert(0, str(REPO_ROOT))

from _dev.quillan_v8_saturated import QuillanRoninSovereign, QuillanArchConfig

print("==================================================================")
print("   👑 DIAGNOSING MODEL EMBEDDING & VOCABULARY ALIGNMENT")
print("==================================================================")

cfg = QuillanArchConfig(hidden_dim=1024, ffn_dim=2048, num_experts=34, text_only=True, eggroll_rank=256)
model = QuillanRoninSovereign(cfg).to("cpu")

ckpt_path = r"C:\02_QUILLAN\checkpoints\checkpoints_sft\quillan_final_best.pt"
print(f"[*] Loading checkpoint: {ckpt_path}")
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
sd = ckpt.get("model_state_dict", ckpt)

msd = model.state_dict()
for k, v in sd.items():
    if k in msd and v.shape == msd[k].shape:
        msd[k].copy_(v)
model.load_state_dict(msd)

emb_w = model.ingestion.txt_emb.weight
print(f"[EMBEDDING] Shape: {emb_w.shape}, Mean: {emb_w.mean().item():.6f}, Std: {emb_w.std().item():.6f}, Norm: {emb_w.norm().item():.4f}")

# Check top predicted tokens for simple prompt
enc = tiktoken.get_encoding("gpt2")
prompt = "The capital of France is"
tokens = enc.encode(prompt)
inp = torch.tensor([tokens], dtype=torch.long)

with torch.no_grad():
    logits = model(inp)[:, -1, :]
    top_v, top_i = torch.topk(logits, 10, dim=-1)
    
print(f"\nPrompt: '{prompt}'")
print("Top 10 predicted tokens:")
for val, idx in zip(top_v[0].tolist(), top_i[0].tolist()):
    word = enc.decode([idx])
    print(f"  ID {idx:5d} | Logit: {val:7.3f} | Word: {repr(word)}")
