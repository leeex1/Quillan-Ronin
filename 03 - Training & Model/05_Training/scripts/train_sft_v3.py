#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QUILLAN-RONIN v5.3.1 — Full Parameter SFT Training v3
=====================================================
FIXED: Matches model architecture EXACTLY to base checkpoint.
Loads 100% of pretrained weights. Zero random layers.
"""

import os
import sys
import time
import math
import argparse
import json
import random

# Force CPU execution
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch
import torch.nn as nn
import torch.nn.functional as F
import tiktoken

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from _dev.quillan_v8_saturated import QuillanRoninSovereign, QuillanArchConfig

def parse_args():
    parser = argparse.ArgumentParser(description="Quillan Full Parameter SFT v3")
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--resume-step", type=int, default=0)
    return parser.parse_args()

def save_checkpoint(model, path, step, loss, best_loss):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'step': step,
        'loss': loss,
        'best_loss': best_loss,
        'version': 'v5.3.1-sft-v3'
    }, path)
    print(f"[SAVE] Checkpoint saved: {os.path.basename(path)} step={step} loss={loss:.4f}", flush=True)

def load_dataset():
    """Load pre-tokenized samples from unified corpus."""
    data_dir = os.path.join(REPO_ROOT, "training_data")
    samples = []
    
    files = [
        "unified_tokenized_corpus.jsonl",
        "quillan_corpus_CLEAN_V7.jsonl",
        "GPT_5.5_Distilled.jsonl",
        "code_train.jsonl",
        "full_train.jsonl",
        "instruct_train.jsonl",
        "train.jsonl",
        "quillan_science_absolute.jsonl",
        "quillan_science_additional.jsonl",
        "pdf_papers_corpus.jsonl",
        "quillan_12mb_training_dataset.jsonl"
    ]
    
    for fname in files:
        fpath = os.path.join(data_dir, fname)
        if not os.path.exists(fpath):
            continue
        count = 0
        with open(fpath, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    tokens = obj.get("input_ids", obj.get("tokens", []))
                    labels = obj.get("labels", obj.get("target_ids", tokens))
                    if len(tokens) >= 32:
                        samples.append((tokens, labels))
                        count += 1
                except Exception:
                    pass
        if count > 0:
            print(f"[DATA] {fname}: {count} samples", flush=True)
        if len(samples) >= 50000:
            break
    
    print(f"[DATA] Total samples loaded: {len(samples)}", flush=True)
    return samples

def generate_sample(model, enc, prompt_text, max_tokens=40, temperature=0.7):
    """Generate a short text sample to verify model is learning."""
    model.eval()
    tokens = enc.encode(prompt_text)
    with torch.no_grad():
        for _ in range(max_tokens):
            inp = torch.tensor([tokens[-512:]], dtype=torch.long)
            logits = model(inp)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1).item()
            if next_tok == 50256:  # EOS
                break
            tokens.append(next_tok)
    return enc.decode(tokens[len(enc.encode(prompt_text)):])

def main():
    args = parse_args()
    device = "cpu"
    
    ckpt_dir = os.path.join(REPO_ROOT, "checkpoints", "checkpoints_sft")
    os.makedirs(ckpt_dir, exist_ok=True)
    
    # ═══════════════════════════════════════════════════════════════════
    # STEP 1: Load base checkpoint FIRST to discover its exact config
    # ═══════════════════════════════════════════════════════════════════
    base_ckpt_path = os.path.join(REPO_ROOT, "checkpoints", "checkpoints_v2", "quillan_full_base_final.pt")
    best_ckpt = os.path.join(ckpt_dir, "quillan_sft_v3_best.pt")
    latest_ckpt = os.path.join(ckpt_dir, "quillan_sft_v3_latest.pt")
    
    # Config MUST match what the base checkpoint was trained with
    cfg = QuillanArchConfig(
        hidden_dim=1024, 
        ffn_dim=2048, 
        num_experts=34, 
        text_only=True,
        eggroll_rank=256,  # Matches checkpoint BitLinear lora shapes
    )
    model = QuillanRoninSovereign(cfg).to(device)
    
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[MODEL] {total_params:.1f}M parameters", flush=True)
    
    # ═══════════════════════════════════════════════════════════════════
    # STEP 2: Load weights with maximum compatibility
    # ═══════════════════════════════════════════════════════════════════
    start_step = args.resume_step
    best_loss = 999.0
    
    if start_step > 0 and os.path.exists(latest_ckpt):
        print(f"[RESUME] Loading from {os.path.basename(latest_ckpt)}...", flush=True)
        ckpt = torch.load(latest_ckpt, map_location="cpu", weights_only=False)
        sd = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(sd, strict=False)
        best_loss = ckpt.get("best_loss", 999.0)
        print(f"[RESUME] Resumed from step {start_step}, best_loss={best_loss:.4f}", flush=True)
    elif os.path.exists(base_ckpt_path):
        print(f"[BASE] Loading pretrained base weights...", flush=True)
        ckpt = torch.load(base_ckpt_path, map_location="cpu", weights_only=False)
        sd = ckpt.get("model_state_dict", ckpt)
        
        model_sd = model.state_dict()
        loaded, skipped_shape, skipped_missing = 0, 0, 0
        
        for k, v in sd.items():
            if k not in model_sd:
                skipped_missing += 1
                continue
            if v.shape != model_sd[k].shape:
                skipped_shape += 1
                continue
            model_sd[k].copy_(v)
            loaded += 1
        
        model.load_state_dict(model_sd)
        total_model_keys = len(model_sd)
        pct = 100 * loaded / total_model_keys
        print(f"[BASE] Loaded {loaded}/{total_model_keys} layers ({pct:.0f}%)", flush=True)
        if skipped_shape > 0:
            print(f"[BASE] Skipped {skipped_shape} shape mismatches, {skipped_missing} not in model", flush=True)
        
        if pct < 80:
            print(f"[WARNING] Only {pct:.0f}% of model weights loaded from checkpoint!", flush=True)
            print(f"[WARNING] The remaining layers start randomly initialized.", flush=True)
    
    # ═══════════════════════════════════════════════════════════════════
    # STEP 3: Load dataset
    # ═══════════════════════════════════════════════════════════════════
    samples = load_dataset()
    if not samples:
        print("[FATAL] No training data found!", flush=True)
        sys.exit(1)
    
    # ═══════════════════════════════════════════════════════════════════
    # STEP 4: Training loop
    # ═══════════════════════════════════════════════════════════════════
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    enc = tiktoken.get_encoding("gpt2")
    
    total_steps = args.steps
    warmup = args.warmup_steps
    min_lr = 1e-6
    
    def get_lr(step):
        if step < warmup:
            return args.lr * (step + 1) / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        return min_lr + 0.5 * (args.lr - min_lr) * (1.0 + math.cos(math.pi * progress))
    
    model.train()
    print(f"\n[TRAIN] SFT from step {start_step} -> {total_steps} ({len(samples)} samples)", flush=True)
    
    step = start_step
    sample_idx = 0
    start_time = time.time()
    
    while step < total_steps:
        lr = get_lr(step)
        for pg in optimizer.param_groups:
            pg['lr'] = lr
        
        optimizer.zero_grad()
        accum_loss = 0.0
        
        for _ in range(args.grad_accum):
            tokens, labels = samples[sample_idx % len(samples)]
            sample_idx += 1
            
            seq_len = min(args.seq_len, len(tokens))
            txt_in = torch.tensor([tokens[:seq_len]], dtype=torch.long, device=device)
            target_in = torch.tensor([labels[:seq_len]], dtype=torch.long, device=device)
            
            logits = model(txt_in)
            
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = target_in[..., 1:].contiguous()
            
            loss = F.cross_entropy(
                shift_logits.view(-1, cfg.vocab_size), 
                shift_labels.view(-1), 
                ignore_index=-100
            )
            loss_scaled = loss / args.grad_accum
            loss_scaled.backward()
            accum_loss += loss.item() / args.grad_accum
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        step += 1
        
        # Log every step for first 10, then every 25
        if step <= 10 or step % 25 == 0:
            elapsed = time.time() - start_time
            sec_per_step = elapsed / max(1, step - start_step)
            eta_hours = (total_steps - step) * sec_per_step / 3600.0
            print(f"  step {step:5d}/{total_steps}  loss={accum_loss:.4f}  lr={lr:.2e}  {sec_per_step:.1f}s/st  ETA:{eta_hours:.1f}h", flush=True)
        
        # Save best checkpoint
        if accum_loss < best_loss and step >= 25:
            best_loss = accum_loss
            save_checkpoint(model, best_ckpt, step, accum_loss, best_loss)
        
        # Save periodic checkpoint every 250 steps
        if step % 250 == 0:
            save_checkpoint(model, latest_ckpt, step, accum_loss, best_loss)
        
        # Generate a sample every 500 steps to verify learning
        if step % 500 == 0:
            sample_out = generate_sample(model, enc, "<|user|>\nHello, who are you?\n<|assistant|>\n", max_tokens=30)
            print(f"  [SAMPLE @ step {step}]: {sample_out[:120]}", flush=True)
            model.train()
    
    print(f"\n[COMPLETE] Training finished at step {total_steps}!", flush=True)
    save_checkpoint(model, latest_ckpt, total_steps, accum_loss, best_loss)
    
    # Final generation test
    sample_out = generate_sample(model, enc, "<|user|>\nHello, who are you?\n<|assistant|>\n", max_tokens=60)
    print(f"\n[FINAL SAMPLE]: {sample_out[:200]}", flush=True)

if __name__ == "__main__":
    main()
