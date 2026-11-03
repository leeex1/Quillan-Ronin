#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quillan-Ronin: Interactive Sovereign Generation & Council Activation Probe
"""
import sys
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

sys.path.insert(0, r'C:\02_QUILLAN\03 - Training & Model\scripts')
from quillan_v5_4_oni import QuillanOniConfig, QuillanRoninOni

tok_path = r'C:\02_QUILLAN\02 - Knowledge Foundation\knowledge\legacy\00 - Meta\oni\tokenizer.json'
ckpt_path = r'C:\02_QUILLAN\checkpoints\checkpoints_oni\sft_runs\quillan_sft_latest.pt'

print(f"Loading Tokenizer from: {tok_path}")
tok = Tokenizer.from_file(tok_path)

print(f"Loading Checkpoint: {ckpt_path}")
ckpt = torch.load(ckpt_path, map_location="cpu")
print(f"Step: {ckpt.get('step')} | Best Val CE: {ckpt.get('best_val_ce', 'N/A'):.4f}")

cfg = QuillanOniConfig(**ckpt['config'])
model = QuillanRoninOni(cfg).cpu().eval()
model.load_state_dict(ckpt['model_state_dict'], strict=False)

def sample_next_token(logits, temperature=0.7, top_k=40, top_p=0.9, rep_penalty=1.2, generated_tokens=None):
    logits = logits.squeeze(0).clone()
    
    # Apply repetition penalty
    if generated_tokens and rep_penalty != 1.0:
        for tid in set(generated_tokens):
            if logits[tid] > 0:
                logits[tid] /= rep_penalty
            else:
                logits[tid] *= rep_penalty
                
    if temperature <= 0.0:
        return torch.argmax(logits).item()
        
    logits = logits / temperature
    
    # Top-K filtering
    if top_k > 0:
        top_k = min(top_k, logits.size(-1))
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = -float('Inf')
        
    # Top-P (nucleus) filtering
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = -float('Inf')
        
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).item()

test_prompts = [
    # 1. Plain prefix
    "The 34-member Council is designed to",
    # 2. ChatML dialogue format
    "<|im_start|>user\nWhat is the mission of Quillan?\n<|im_end|>\n<|im_start|>assistant\n",
    # 3. Python code prompt
    "def calculate_fibonacci(n):\n    \"\"\"Return the nth Fibonacci number.\"\"\"\n"
]

for p in test_prompts:
    print("\n" + "="*70)
    print(f"INPUT PROMPT:\n{p}")
    print("="*70)
    
    enc = tok.encode(p).ids
    generated = list(enc)
    
    with torch.no_grad():
        for _ in range(40):
            x = torch.tensor([generated[-128:]], dtype=torch.long, device="cpu")
            out = model(x)
            logits = out[0] if isinstance(out, tuple) else out
            last_logits = logits[:, -1, :]
            
            next_token = sample_next_token(
                last_logits, 
                temperature=0.75, 
                top_k=40, 
                top_p=0.85, 
                rep_penalty=1.25, 
                generated_tokens=generated[-30:]
            )
            generated.append(next_token)
            
            if next_token in [tok.token_to_id("<|endoftext|>"), tok.token_to_id("<|im_end|>")]:
                break
                
    result_text = tok.decode(generated)
    print("GENERATED RESPONSE:")
    print(result_text)

