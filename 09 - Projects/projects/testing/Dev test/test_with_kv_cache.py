#!/usr/bin/env python3
"""
Test autoregressive generation using proper KV Caching (past_key_values)
to see the true output of the model without re-executing agentic tools on every token.
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

def generate_kv(model, tokenizer, prompt, max_new_tokens=80, temperature=0.2, repetition_penalty=1.2, top_p=0.9, top_k=40, device='cpu'):
    model.eval()
    input_ids = tokenizer.encode(prompt)
    generated = list(input_ids)
    eos_id = getattr(tokenizer, 'eos_id', 50256)

    # Initial prompt pass
    inp = torch.tensor([input_ids], dtype=torch.long, device=device)
    with torch.no_grad():
        out = model(inp)
        logits = out["logits"][:, -1, :].clone()
        past_key_values = out.get("past_key_values")

        for step in range(max_new_tokens):
            # Apply repetition penalty
            if repetition_penalty != 1.0:
                for token_id in set(generated):
                    if logits[0, token_id] < 0:
                        logits[0, token_id] *= repetition_penalty
                    else:
                        logits[0, token_id] /= repetition_penalty

            if temperature > 0:
                logits = logits / temperature
                if top_k > 0:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = float('-inf')
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).item()
            else:
                next_token = torch.argmax(logits, dim=-1).item()

            if next_token == eos_id:
                break
            generated.append(next_token)

            # Autoregressive step with KV Cache
            next_inp = torch.tensor([[next_token]], dtype=torch.long, device=device)
            out = model(next_inp, past_key_values=past_key_values)
            logits = out["logits"][:, -1, :].clone()
            past_key_values = out.get("past_key_values")

    return tokenizer.decode(generated[len(input_ids):])

def main():
    device = 'cuda' if (torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 7) else 'cpu'
    cfg = QuillanArchConfig(device=device, text_only=True, eggroll_rank=16)
    model = QuillanRoninSovereign(cfg).to(device)

    tok_path = ROOT / '_dev' / 'quillan_bpe_tokenizer_hf' / 'tokenizer.json'
    tokenizer = QuillanBPETokenizer()
    tokenizer.load(str(tok_path))

    # Load Base
    base_path = ROOT / 'checkpoints' / 'checkpoints_v2' / 'quillan_full_base_final.pt'
    base_ckpt = torch.load(str(base_path), map_location=device, weights_only=False)
    base_sd = base_ckpt.get('model_state_dict', base_ckpt)
    model_sd = model.state_dict()
    for k, v in base_sd.items():
        if k in model_sd and v.shape == model_sd[k].shape:
            model_sd[k].copy_(v)

    # Load SFT Best
    sft_path = ROOT / 'checkpoints' / 'checkpoints_sft' / 'quillan_full_param_v2_best.pt'
    print(f"Loading {sft_path.name}...")
    sft_ckpt = torch.load(str(sft_path), map_location=device, weights_only=False)
    sft_sd = sft_ckpt.get('model_state_dict', sft_ckpt)
    for k, v in sft_sd.items():
        if k in model_sd and v.shape == model_sd[k].shape:
            model_sd[k].copy_(v)
    model.load_state_dict(model_sd)

    prompts = [
        "<|user|>\nHello Quillan! What architectural features do you use?\n<|assistant|>\n",
        "<|user|>\nWrite a Python function to read a file.\n<|assistant|>\n",
        "<|user|>\nWhat is photosyntheses?\n<|assistant|>\n<think>\nPhotosynthesis is the process by which plants convert sunlight into energy.\n</think>\n\n"
    ]

    for p in prompts:
        print("\n" + "=" * 60)
        print("PROMPT:", p.strip())
        print("-" * 60)
        res = generate_kv(model, tokenizer, p, max_new_tokens=70, temperature=0.3, repetition_penalty=1.25, device=device)
        print("RESPONSE:", res.strip())

if __name__ == '__main__':
    main()
