#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys, time, torch
import torch.nn.functional as F
from tokenizers import Tokenizer

sys.path.insert(0, r'C:\02_QUILLAN\03 - Training & Model\scripts')
from quillan_v5_4_oni import QuillanOniConfig, QuillanRoninOni

tok_path = r'C:\02_QUILLAN\02 - Knowledge Foundation\knowledge\legacy\00 - Meta\oni\tokenizer.json'
ckpt_path = r'C:\02_QUILLAN\checkpoints\checkpoints_oni\sft_runs\quillan_sft_latest.pt'

print(f"Loading Tokenizer from: {tok_path}")
tok = Tokenizer.from_file(tok_path)

print(f"Loading Checkpoint into CUDA (FP16): {ckpt_path}")
ckpt = torch.load(ckpt_path, map_location="cpu")
print(f"Step: {ckpt.get('step')} | Best Val CE: {ckpt.get('best_val_ce', 'N/A'):.4f}")

cfg = QuillanOniConfig(**ckpt['config'])
model = QuillanRoninOni(cfg).half().cuda().eval()
model.load_state_dict({k: v.half() if v.is_floating_point() else v for k, v in ckpt['model_state_dict'].items()}, strict=False)

vram = torch.cuda.memory_allocated() / (1024*1024)
print(f"Model loaded into VRAM: {vram:.1f} MB (Comfortably under 4 GB!)")

def sample_next_token(logits, temperature=0.75, top_k=40, top_p=0.85, rep_penalty=1.25, generated_tokens=None):
    logits = logits.squeeze(0).float()
    if generated_tokens and rep_penalty != 1.0:
        for tid in set(generated_tokens):
            if logits[tid] > 0:
                logits[tid] /= rep_penalty
            else:
                logits[tid] *= rep_penalty
    if temperature <= 0.0:
        return torch.argmax(logits).item()
    logits = logits / temperature
    if top_k > 0:
        top_k = min(top_k, logits.size(-1))
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = -float('Inf')
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
    "Quillan is an autonomous cognitive AI. Its core purpose is",
    "<|im_start|>user\nWhat is the mission of Quillan?\n<|im_end|>\n<|im_start|>assistant\n",
    "def calculate_fibonacci(n):\n    \"\"\"Return the nth Fibonacci number.\"\"\"\n"
]

for p in test_prompts:
    print("\n" + "="*70)
    print(f"INPUT PROMPT:\n{p}")
    print("="*70)
    enc = tok.encode(p).ids
    generated = list(enc)
    
    t0 = time.time()
    with torch.no_grad():
        for _ in range(40):
            x = torch.tensor([generated[-128:]], dtype=torch.long, device="cuda")
            out = model(x)
            logits = out[0] if isinstance(out, tuple) else out
            last_logits = logits[:, -1, :]
            
            next_token = sample_next_token(
                last_logits, 
                temperature=0.75, 
                top_k=40, 
                top_p=0.85, 
                rep_penalty=1.3, 
                generated_tokens=generated[-30:]
            )
            generated.append(next_token)
            if next_token in [tok.token_to_id("<|endoftext|>"), tok.token_to_id("<|im_end|>")]:
                break
    elapsed = time.time() - t0
    result_text = tok.decode(generated)
    print(f"GENERATED RESPONSE ({elapsed:.2f}s, {len(generated)-len(enc)} tokens, {elapsed/(len(generated)-len(enc)+1e-5)*1000:.1f}ms/tok):")
    print(result_text)

