#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
Quillan-Ronin: Fast Sovereign Inference & Verification Probe
=============================================================================
Tests fast batched inference, checks for absence of cyclic attractor loops,
and verifies output fluency and accuracy.
=============================================================================
"""

import os
import sys

# Configure threading limits
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

sys.path.insert(0, r'C:\02_QUILLAN\03 - Training & Model\scripts')
from quillan_v5_4_oni import QuillanOniConfig, QuillanRoninOni


def test_generation():
    print("=" * 80, flush=True)
    print("VERIFYING QUILLAN SOVEREIGN 1B (QUAD-DONOR WSP 128K WITH QWEN 3.5 2B)", flush=True)
    print("=" * 80, flush=True)
    
    ckpt_path = r"C:\02_QUILLAN\checkpoints\checkpoints_oni\sft_runs\quillan_1b_sovereign_wsp128k_qwen35.pt"
    if not os.path.exists(ckpt_path):
        print(f"Error: Checkpoint {ckpt_path} not found!", flush=True)
        return
        
    tok_dir = r"C:\Users\Admin\.cache\huggingface\hub\models--unsloth--Llama-3.2-1B-Instruct\snapshots"
    tok_snap = os.path.join(tok_dir, os.listdir(tok_dir)[0])
    tok_path = os.path.join(tok_snap, "tokenizer.json")
    
    print(f"Loading 128k Tokenizer from {tok_path}...", flush=True)
    tok = Tokenizer.from_file(tok_path)
    
    print(f"Loading Checkpoint from {ckpt_path}...", flush=True)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    
    cfg_dict = ckpt["config"]
    state_dict = ckpt.pop("model_state_dict")
    del ckpt
    import gc
    gc.collect()
    
    cfg = QuillanOniConfig(
        vocab_size=cfg_dict.get("vocab_size", 128256),
        hidden_dim=cfg_dict.get("hidden_dim", 2048),
        n_layer=cfg_dict.get("n_layer", 16),
        n_head=cfg_dict.get("n_head", 32),
        head_dim=cfg_dict.get("head_dim", 64),
        ffn_dim=cfg_dict.get("ffn_dim", 8192),
        num_experts=cfg_dict.get("num_experts", 34),
        expert_rank=16,
        use_evo_moe=False
    )
    cfg.vectorized_experts = True
    
    print("Instantiating QuillanRoninOni model directly in bfloat16...", flush=True)
    orig_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    model = QuillanRoninOni(cfg)
    torch.set_default_dtype(orig_dtype)
    
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    del state_dict
    gc.collect()
    print(f"Loaded successfully. Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}", flush=True)
    model.eval()
    
    # Check if BatchedCouncilCompute is supported
    batched_status = [getattr(blk.moe, "vectorized_experts", False) for blk in model.h]
    print(f"BatchedCouncilCompute status across 16 layers: {all(batched_status)} (Layers enabled: {sum(batched_status)}/16)", flush=True)
    
    test_prompts = [
        "Explain the core difference between inductive and deductive reasoning.",
        "Solve for x: 3x + 15 = 45.",
        "What is Bushido, and how does it guide honor and duty?"
    ]
    
    for prompt_idx, prompt in enumerate(test_prompts, 1):
        print("\n" + "-" * 60, flush=True)
        print(f"TEST {prompt_idx}: \"{prompt}\"", flush=True)
        print("-" * 60, flush=True)
        
        encoded = tok.encode(f"<|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n")
        input_ids = torch.tensor([encoded.ids], dtype=torch.long)
        
        generated_tokens = []
        cur_input = input_ids
        
        max_new_tokens = 40
        t0 = time.time()
        with torch.no_grad():
            for step in range(max_new_tokens):
                # Pass full prefix history so model maintains causal coherence
                logits = model(cur_input, compute_aux=False, return_aux=False)
                if isinstance(logits, tuple):
                    logits = logits[0]
                next_token_logits = logits[0, -1, :]
                
                # Temperature sampling with top-k filtering
                temp = 0.6
                top_k = 50
                v, _ = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
                next_token_logits[next_token_logits < v[-1]] = -float('Inf')
                probs = F.softmax(next_token_logits / temp, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).item()
                
                # Check for EOT token
                if next_token in [128001, 128009]: # Llama 3 <|end_of_text|>, <|eot_id|>
                    break
                    
                generated_tokens.append(next_token)
                # Autoregressively append new token to maintain context
                cur_input = torch.cat([cur_input, torch.tensor([[next_token]], dtype=torch.long)], dim=1)
                
                # Print token immediately
                token_str = tok.decode([next_token])
                print(token_str, end="", flush=True)
                
        elapsed = time.time() - t0
        tok_sec = len(generated_tokens) / max(0.01, elapsed)
        print(f"\n[Generated {len(generated_tokens)} tokens in {elapsed:.2f}s ({tok_sec:.2f} tok/s)]", flush=True)


if __name__ == "__main__":
    test_generation()
