#!/usr/bin/env python3
"""
Quillan-Ronin v5.3.1 — Conversational Dialogue Generation Audit Test
Tests the target-masked dialogue checkpoint (quillan_causal_aligned.pt) across multi-turn chat prompts.
"""

import os
import sys
import torch
import torch.nn.functional as F
from pathlib import Path

ROOT = Path(r"C:\Users\Admin\Quillan-Ronin")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / '_dev'))
sys.path.insert(0, r"C:\02_QUILLAN\09 - Projects\projects")  # FIX 2026-09-09: live _dev package home

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from _dev.quillan_v8_saturated import QuillanRoninSovereign, QuillanArchConfig
from _dev.quillan_bpe_tokenizer import QuillanBPETokenizer

def generate(model, tokenizer, prompt, max_new_tokens=80, temperature=0.7, top_p=0.9, device='cpu'):
    model.eval()
    input_ids = tokenizer.encode(prompt)
    generated = list(input_ids)
    eos_id = getattr(tokenizer, 'eos_id', 50256)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            inp = torch.tensor([generated[-512:]], dtype=torch.long, device=device)
            out = model(inp)
            logits = out["logits"][:, -1, :]

            if temperature > 0:
                logits = logits / temperature
                probs = F.softmax(logits, dim=-1)
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float('-inf')

                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).item()
            else:
                next_token = torch.argmax(logits, dim=-1).item()

            if next_token == eos_id:
                break
            generated.append(next_token)

    gen_text = tokenizer.decode(generated[len(input_ids):])
    return gen_text

def main():
    device = 'cuda' if (torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 7) else 'cpu'
    print("=" * 65)
    print("  QUILLAN CONVERSATIONAL DIALOGUE GENERATION TEST")
    print("=" * 65)

    cfg = QuillanArchConfig(device=device, text_only=True, eggroll_rank=16)
    model = QuillanRoninSovereign(cfg).to(device)

    tok_path = ROOT / '_dev' / 'quillan_bpe_tokenizer_hf' / 'tokenizer.json'
    tokenizer = QuillanBPETokenizer()
    tokenizer.load(str(tok_path))

    # Load Base
    base_path = ROOT / 'checkpoints' / 'checkpoints_v2' / 'quillan_full_base_final.pt'
    print(f"[1/2 LOAD BASE] {base_path.name}...")
    base_ckpt = torch.load(str(base_path), map_location=device, weights_only=False)
    base_sd = base_ckpt.get('model_state_dict', base_ckpt)
    model_sd = model.state_dict()
    for k, v in base_sd.items():
        if k in model_sd and v.shape == model_sd[k].shape:
            model_sd[k].copy_(v)

    # Load Aligned SFT
    sft_path = ROOT / 'checkpoints' / 'checkpoints_sft' / 'quillan_causal_aligned.pt'
    print(f"[2/2 LOAD CONVERSATIONAL SFT] {sft_path.name}...")
    sft_ckpt = torch.load(str(sft_path), map_location=device, weights_only=False)
    sft_sd = sft_ckpt.get('model_state_dict', sft_ckpt)
    for k, v in sft_sd.items():
        if k in model_sd and v.shape == model_sd[k].shape:
            model_sd[k].copy_(v)

    prompts = [
        ("CONVERSATIONAL IDENTITY", "<|system|>\nYou are Quillan, a sovereign AI assistant.\n<|user|>\nHello! Who are you?\n<|assistant|>\n"),
        ("PYTHON TUTOR", "<|system|>\nYou are a computer science tutor. Explain concepts clearly.\n<|user|>\nHow do I open and read a file in Python?\n<|assistant|>\n"),
        ("SCIENCE REASONING", "<|system|>\nYou are a science teacher. Explain simply.\n<|user|>\nWhy is the sky blue?\n<|assistant|>\n"),
    ]

    for title, prompt in prompts:
        print("\n" + "-" * 65)
        print(f"[{title}]")
        print(f"PROMPT   :\n{prompt.strip()}")
        out_text = generate(model, tokenizer, prompt, max_new_tokens=80, temperature=0.7, top_p=0.9, device=device)
        print(f"\nGENERATED:\n{out_text}")
        print("-" * 65, flush=True)

    print("\n[COMPLETE] Conversational dialogue generation test finished successfully.")

if __name__ == '__main__':
    main()
