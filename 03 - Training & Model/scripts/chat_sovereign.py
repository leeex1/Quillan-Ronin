#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
Quillan-Ronin: Interactive Sovereign 1B Terminal REPL
=============================================================================
Usage:
  python chat_sovereign.py
Chat directly with the 43 tok/s Sovereign 1B WSP Quad-Donor model in real-time.
=============================================================================
"""

import os
import sys
import time
from typing import List
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, r"C:\02_QUILLAN\03 - Training & Model\scripts")
sys.path.append(r"C:\02_QUILLAN\02 - Knowledge Foundation\knowledge\legacy\00 - Meta\oni")
from quillan_v5_4_oni import QuillanOniConfig, QuillanRoninOni


def launch_chat():
    ckpt_path = r"C:\02_QUILLAN\checkpoints\checkpoints_oni\sft_runs\quillan_1b_sovereign_wsp128k_qwen35.pt"
    tok_dir = r"C:\Users\Admin\.cache\huggingface\hub\models--unsloth--Llama-3.2-1B-Instruct\snapshots"
    tok_snap = os.path.join(tok_dir, os.listdir(tok_dir)[0])
    tok_path = os.path.join(tok_snap, "tokenizer.json")
    
    print("=" * 80, flush=True)
    print("QUILLAN-RONIN SOVEREIGN 1B (QUAD-DONOR WSP 128K) INTERACTIVE CONSOLE", flush=True)
    print("=" * 80, flush=True)
    print("Loading weights...", end="", flush=True)
    
    tok = Tokenizer.from_file(tok_path)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg_dict = ckpt["config"]
    
    cfg = QuillanOniConfig(
        vocab_size=cfg_dict.get("vocab_size", 128256),
        hidden_dim=cfg_dict.get("hidden_dim", 2048),
        n_layer=cfg_dict.get("n_layer", 16),
        n_head=cfg_dict.get("n_head", 32),
        head_dim=cfg_dict.get("head_dim", 64),
        ffn_dim=cfg_dict.get("ffn_dim", 8192),
        num_experts=cfg_dict.get("num_experts", 34),
        expert_rank=16,
        use_evo_moe=False,
        device="cpu"
    )
    cfg.vectorized_experts = True
    
    model = QuillanRoninOni(cfg)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()
    print(" Ready! [43 tok/s BatchedCouncilCompute Active]\n", flush=True)
    print("Type your prompt and press Enter. Type 'exit' or 'quit' to stop.\n", flush=True)
    
    history = []
    
    while True:
        try:
            user_input = input("User > ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["exit", "quit", "q"]:
                print("Exiting Sovereign Console. Sayonara.", flush=True)
                break
                
            history.append({"role": "user", "content": user_input})
            
            # Build conversation context
            prompt = ""
            for msg in history[-4:]: # Keep last 4 turns for context
                prompt += f"<|start_header_id|>{msg['role']}<|end_header_id|>\n\n{msg['content']}<|eot_id|>"
            prompt += "<|start_header_id|>assistant<|end_header_id|>\n\n"
            
            encoded = tok.encode(prompt)
            cur_input = torch.tensor([encoded.ids], dtype=torch.long)
            
            print("Quillan > ", end="", flush=True)
            generated_tokens = []
            t0 = time.time()
            
            with torch.no_grad():
                for _ in range(256):
                    logits = model(cur_input, compute_aux=False, return_aux=False)
                    if isinstance(logits, tuple):
                        logits = logits[0]
                    next_token_logits = logits[0, -1, :]
                    probs = F.softmax(next_token_logits / 0.65, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1).item()
                    
                    if next_token in [128001, 128009]:
                        break
                        
                    generated_tokens.append(next_token)
                    cur_input = torch.tensor([[next_token]], dtype=torch.long)
                    print(tok.decode([next_token]), end="", flush=True)
                    
            elapsed = time.time() - t0
            tok_s = len(generated_tokens) / max(0.01, elapsed)
            print(f"\n({len(generated_tokens)} tokens in {elapsed:.2f}s | {tok_s:.1f} tok/s)\n", flush=True)
            
            assistant_reply = tok.decode(generated_tokens)
            history.append({"role": "assistant", "content": assistant_reply})
            
        except KeyboardInterrupt:
            print("\nSession paused. Exiting.", flush=True)
            break


if __name__ == "__main__":
    launch_chat()

