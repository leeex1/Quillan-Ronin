#!/usr/bin/env python3
"""
Quillan-Ronin v5.3.1 — Text Generation Script (2026)
===================================================
Loads the trained checkpoint and runs autoregressive text generation
with full support for the multi-diffusion step KV caching.
"""
import os
import sys
import gc
import time
import argparse
import warnings

# Disable CUDA and warnings
os.environ['CUDA_VISIBLE_DEVICES'] = ''
warnings.filterwarnings('ignore', category=UserWarning, module='torch')

import torch
import torch.nn.functional as F
from pathlib import Path

# Add root and _dev to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / '_dev'))
sys.path.insert(0, str(ROOT))

# ── Force UTF-8 on Windows ────────────────────────────────────────────────────
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from quillan_v8_saturated import QuillanRoninSovereign, QuillanArchConfig
from quillan_bpe_tokenizer import QuillanBPETokenizer

def load_checkpoint_clean(model, ckpt_path):
    print(f"[LOAD] Loading checkpoint: {ckpt_path.name}")
    ckpt = torch.load(ckpt_path, map_location='cpu')
    sd = ckpt.get('model_state_dict', ckpt)

    model_sd = model.state_dict()
    loaded_count = 0
    skipped_count = 0
    for k in list(sd.keys()):
        v = sd.pop(k)
        if k in model_sd:
            if v.shape == model_sd[k].shape:
                model_sd[k].copy_(v)
                loaded_count += 1
            else:
                skipped_count += 1
        else:
            skipped_count += 1
            
    print(f"[LOAD] Loaded {loaded_count} keys, skipped {skipped_count} mismatched/missing keys")
    del sd
    gc.collect()

def generate(model, tokenizer, prompt, max_new_tokens=100, temperature=0.7, top_k=50, top_p=0.9, repetition_penalty=1.2):
    model.eval()
    
    # Encode prompt
    tokens = tokenizer.encode(prompt)
    print(f"\nPrompt tokens: {tokens}")
    print(f"--- Prompt: {prompt} ---")
    print("--- Generation ---")
    
    input_tensor = torch.tensor([tokens], dtype=torch.long)
    past_key_values = None
    generated_tokens = []
    
    for step in range(max_new_tokens):
        with torch.no_grad():
            # Full sequence forward pass (matching training mode)
            out = model(input_tensor, past_key_values=None, use_cache=False, recursive_depth=1)
            logits = out["logits"][:, -1, :]  # [1, vocab_size]
            
            # Apply repetition & n-gram block penalty across context
            if repetition_penalty != 1.0:
                all_toks = input_tensor[0].tolist()
                for token in set(all_toks):
                    if logits[0, token] > 0:
                        logits[0, token] /= repetition_penalty
                    else:
                        logits[0, token] *= repetition_penalty
                # Extra penalty for immediate previous token
                last_tok = all_toks[-1]
                if logits[0, last_tok] > 0:
                    logits[0, last_tok] /= (repetition_penalty * 1.3)
                # 2-gram repeat ban: prevent generating token B if sequence (last_tok, B) already occurred
                for i in range(len(all_toks) - 1):
                    if all_toks[i] == last_tok:
                        next_t = all_toks[i+1]
                        if logits[0, next_t] > 0:
                            logits[0, next_t] /= (repetition_penalty * 1.5)

            # Apply sampling with Top-K and Top-P (Nucleus) filtering
            if temperature > 0.0:
                logits = logits / temperature
                if top_k > 0:
                    v, ix = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits_filter = torch.full_like(logits, float('-inf'))
                    logits_filter.scatter_(1, ix, v)
                    logits = logits_filter
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    logits = logits.masked_fill(indices_to_remove, float('-inf'))
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
                
            token_id = next_token.item()
            generated_tokens.append(token_id)
            input_tensor = torch.cat([input_tensor, next_token], dim=-1)
            
            # Decode and print token
            token_text = tokenizer.decode([token_id])
            print(token_text, end='', flush=True)
            
            if token_id == tokenizer.eos_token_id:
                print("\n[EOS reached]")
                break
    print("\n------------------")
    return tokenizer.decode(generated_tokens)

def main():
    parser = argparse.ArgumentParser(description="Generate text from Quillan-Ronin checkpoint")
    parser.add_argument("--checkpoint", type=str, default="checkpoints_v2/quillan_sovereign_latest.pt",
                        help="Path to checkpoint file (.pt)")
    parser.add_argument("--prompt", type=str, default="Explain the concept of Lee-Mach-6 cognitive binding.",
                        help="Prompt text to start generation")
    parser.add_argument("--tokens", type=int, default=100, help="Max new tokens to generate")
    parser.add_argument("--temp", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k sampling size")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p nucleus sampling size")
    parser.add_argument("--penalty", type=float, default=1.2, help="Repetition penalty")
    args = parser.parse_args()

    # Load tokenizer
    tok_path = ROOT / "quillan_bpe_tokenizer.pkl"
    tokenizer = QuillanBPETokenizer(path=tok_path)
    
    # Build model
    print("[INIT] Building model architecture...")
    cfg = QuillanArchConfig(device='cpu', text_only=True, eggroll_rank=16)
    model = QuillanRoninSovereign(cfg)
    
    # Load checkpoint
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        fallback_path = Path("checkpoints_v2/quillan_sovereign_final.pt")
        if fallback_path.exists():
            ckpt_path = fallback_path
        else:
            print(f"[ERROR] Checkpoint not found at {ckpt_path} or {fallback_path}")
            sys.exit(1)
            
    load_checkpoint_clean(model, ckpt_path)
    
    # Generate
    generate(model, tokenizer, args.prompt, max_new_tokens=args.tokens, temperature=args.temp, top_k=args.top_k, top_p=args.top_p, repetition_penalty=args.penalty)

if __name__ == "__main__":
    main()
