#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import torch
from tokenizers import Tokenizer

sys.path.insert(0, r'C:\02_QUILLAN\03 - Training & Model\scripts')
from quillan_v5_4_oni import QuillanOniConfig, QuillanRoninOni

tok_path = r'C:\02_QUILLAN\02 - Knowledge Foundation\knowledge\legacy\00 - Meta\oni\tokenizer.json'
ckpt_path = r'C:\02_QUILLAN\checkpoints\checkpoints_oni\sft_runs\quillan_sft_latest.pt'

print(f"Loading Tokenizer from: {tok_path}")
tok = Tokenizer.from_file(tok_path)

print(f"Loading Checkpoint: {ckpt_path}")
ckpt = torch.load(ckpt_path, map_location="cpu")
print(f"Checkpoint Step: {ckpt.get('step')} | Best Val CE: {ckpt.get('best_val_ce', 'N/A'):.4f}")

cfg = QuillanOniConfig(**ckpt['config'])
model = QuillanRoninOni(cfg).cpu().eval()
model.load_state_dict(ckpt['model_state_dict'], strict=False)

prompts = [
    "Quillan is an autonomous cognitive AI. Its core purpose is",
    "def calculate_fibonacci(n):",
    "The 34-member Council operates through"
]

for prompt in prompts:
    print("\n" + "="*60)
    print(f"PROMPT: {prompt}")
    print("="*60)
    enc = tok.encode(prompt).ids
    generated = list(enc)
    
    with torch.no_grad():
        for _ in range(25):
            x = torch.tensor([generated[-128:]], dtype=torch.long, device="cpu")
            out = model(x)
            logits = out[0] if isinstance(out, tuple) else out
            next_tok = torch.argmax(logits[:, -1, :], dim=-1).item()
            generated.append(next_tok)
            if next_tok == tok.token_to_id("<|endoftext|>"):
                break
                
    result_text = tok.decode(generated)
    print(f"OUTPUT:\n{result_text}")

