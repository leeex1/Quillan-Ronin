#!/usr/bin/env python3
"""
Quillan-Ronin v5.3.1 — Multi-Turn Live Interactive Chat Test Script
Simulates multiple back-and-forth conversational turns using interactive_chat's exact wrapper logic.
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

FEW_SHOT_SYSTEM_HEADER = (
    "<|system|>\n"
    "You are Quillan, an Omni-Fractal Sovereign AI Assistant (v5.3.1 Samurai). "
    "You speak clearly, concisely, and fluently in English. You answer questions directly, "
    "provide accurate code, and offer helpful explanations.\n\n"
    "<|user|>\n"
    "Hello! Who are you?\n\n"
    "<|assistant|>\n"
    "<think>\n"
    "The user is asking for an introduction to my identity. I will respond clearly and concisely.\n"
    "</think>\n"
    "Hello! I am Quillan, a sovereign AI assistant. I can assist you with software engineering, "
    "mathematical reasoning, scientific analysis, and creative writing. How can I help you today?\n\n"
)

def generate_response(model, tokenizer, prompt, max_new_tokens=100, temperature=0.2, top_p=0.85, top_k=40, device='cpu'):
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
                if top_k > 0:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = float('-inf')
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

    response_text = tokenizer.decode(generated[len(input_ids):])
    return response_text

def main():
    device = 'cuda' if (torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 7) else 'cpu'
    print("=" * 65)
    print("  QUILLAN MULTI-TURN CONVERSATIONAL CHAT SIMULATION TEST")
    print("=" * 65)

    cfg = QuillanArchConfig(device=device, text_only=True, eggroll_rank=16)
    model = QuillanRoninSovereign(cfg).to(device)

    tok_path = ROOT / '_dev' / 'quillan_bpe_tokenizer_hf' / 'tokenizer.json'
    tokenizer = QuillanBPETokenizer()
    tokenizer.load(str(tok_path))

    # Load Base & SFT Checkpoints
    base_path = ROOT / 'checkpoints' / 'checkpoints_v2' / 'quillan_full_base_final.pt'
    print(f"[1/2 LOAD BASE] {base_path.name}...")
    base_ckpt = torch.load(str(base_path), map_location=device, weights_only=False)
    base_sd = base_ckpt.get('model_state_dict', base_ckpt)
    model_sd = model.state_dict()
    for k, v in base_sd.items():
        if k in model_sd and v.shape == model_sd[k].shape:
            model_sd[k].copy_(v)

    sft_path = ROOT / 'checkpoints' / 'checkpoints_sft' / 'quillan_full_param_v2_best.pt'
    print(f"[2/2 LOAD CONVERSATIONAL SFT] {sft_path.name}...")
    sft_ckpt = torch.load(str(sft_path), map_location=device, weights_only=False)
    sft_sd = sft_ckpt.get('model_state_dict', sft_ckpt)
    for k, v in sft_sd.items():
        if k in model_sd and v.shape == model_sd[k].shape:
            model_sd[k].copy_(v)

    test_queries = [
        "Hi Quillan! What architectural features do you use?",
        "Write a quick Python function to read a text file safely.",
        "Explain photosyntheses in simple terms for a student."
    ]

    history = FEW_SHOT_SYSTEM_HEADER

    for idx, user_msg in enumerate(test_queries, 1):
        print(f"\n" + "-" * 65)
        print(f"[TURN {idx} USER]: {user_msg}")
        prompt = history + f"<|user|>\n{user_msg}\n\n<|assistant|>\n"
        response = generate_response(model, tokenizer, prompt, max_new_tokens=90, temperature=0.7, top_p=0.9, device=device)
        print(f"[TURN {idx} QUILLAN RESPONSE]:\n{response.strip()}")
        print("-" * 65, flush=True)

        history += f"<|user|>\n{user_msg}\n\n<|assistant|>\n{response}\n\n"
        if len(history) > 2000:
            history = FEW_SHOT_SYSTEM_HEADER + history[-1500:]

    print("\n[COMPLETE] Multi-turn conversational simulation test completed.")

if __name__ == '__main__':
    main()
