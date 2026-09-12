#!/usr/bin/env python3
"""
Quillan-Ronin v5.3.1 — Streaming Few-Shot Dialogue Generation Test
"""

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

def generate_stream(model, tokenizer, prompt, max_new_tokens=60, temperature=0.7, top_p=0.9, device='cpu'):
    model.eval()
    input_ids = tokenizer.encode(prompt)
    generated = list(input_ids)
    eos_id = getattr(tokenizer, 'eos_id', 50256)

    print(prompt, end="", flush=True)

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
            token_str = tokenizer.decode([next_token])
            print(token_str, end="", flush=True)

    print("\n" + "-" * 65, flush=True)

def main():
    device = 'cuda' if (torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 7) else 'cpu'
    print("=" * 65)
    print("  QUILLAN FEW-SHOT DIALOGUE STREAMING TEST")
    print("=" * 65)

    cfg = QuillanArchConfig(device=device, text_only=True, eggroll_rank=16)
    model = QuillanRoninSovereign(cfg).to(device)

    tok_path = ROOT / '_dev' / 'quillan_bpe_tokenizer_hf' / 'tokenizer.json'
    tokenizer = QuillanBPETokenizer()
    tokenizer.load(str(tok_path))

    # Load Base & SFT
    base_path = ROOT / 'checkpoints' / 'checkpoints_v2' / 'quillan_full_base_final.pt'
    base_ckpt = torch.load(str(base_path), map_location=device, weights_only=False)
    base_sd = base_ckpt.get('model_state_dict', base_ckpt)
    model_sd = model.state_dict()
    for k, v in base_sd.items():
        if k in model_sd and v.shape == model_sd[k].shape:
            model_sd[k].copy_(v)

    sft_path = ROOT / 'checkpoints' / 'checkpoints_sft' / 'quillan_causal_aligned.pt'
    sft_ckpt = torch.load(str(sft_path), map_location=device, weights_only=False)
    sft_sd = sft_ckpt.get('model_state_dict', sft_ckpt)
    for k, v in sft_sd.items():
        if k in model_sd and v.shape == model_sd[k].shape:
            model_sd[k].copy_(v)

    few_shot_prompt = (
        "<|system|>\n"
        "You are Quillan, a sovereign AI assistant. You speak clearly, concisely, and fluently in English.\n\n"
        "<|user|>\n"
        "Hi! Who are you and what can you do?\n\n"
        "<|assistant|>\n"
        "<think>\n"
        "The user wants an introduction to my identity and skills. I will answer clearly in fluent English.\n"
        "</think>\n"
        "Hello! I am Quillan, a sovereign AI assistant. I can assist you with software engineering, complex mathematics, creative writing, and scientific analysis.\n\n"
        "<|user|>\n"
        "How do I open and read a text file in Python?\n\n"
        "<|assistant|>\n"
    )

    generate_stream(model, tokenizer, few_shot_prompt, max_new_tokens=60, temperature=0.7, top_p=0.9, device=device)
    print("\n[COMPLETE] Few-Shot Dialogue Test Finished.")

if __name__ == '__main__':
    main()
