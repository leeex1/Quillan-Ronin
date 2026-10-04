#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QUILLAN-RONIN v5.3.1 — Dialogue SFT Training (Run 2)
=====================================================
Loads best checkpoint from Run 1 and fine-tunes exclusively on
high-quality dialogue datasets to sharpen conversational ability.

Priority datasets:
  1. GPT_5.5_Distilled.jsonl     - 100% structured <|user|>/<|assistant|> pairs
  2. quillan_corpus_CLEAN_V7.jsonl - Quillan-specific knowledge corpus
  3. quillan_12mb_training_dataset.jsonl - Quillan persona/instruct data

Launch AFTER: train_sft_v3.py finishes (task-18000 completes).
"""

import os
import sys
import time
import math
import json
import random

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch
import torch.nn.functional as F
import tiktoken

# stdout utf-8
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

REPO_ROOT = r"C:\02_QUILLAN"
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from _dev.quillan_v8_saturated import QuillanRoninSovereign, QuillanArchConfig

# ─── CONFIG ─────────────────────────────────────────────────────────────────
TOTAL_STEPS   = 5000
LR            = 5e-6          # Lower LR - already-trained model
MIN_LR        = 5e-7
WARMUP_STEPS  = 100
GRAD_ACCUM    = 4
SEQ_LEN       = 512
SAVE_EVERY    = 250
SAMPLE_EVERY  = 500
LOG_EVERY     = 25

# Datasets in priority order - ALL dialogue-focused
DIALOGUE_DATASETS = [
    "GPT_5.5_Distilled.jsonl",           # 100% <|user|>/<|assistant|> pairs
    "quillan_corpus_CLEAN_V7.jsonl",      # Quillan knowledge corpus
    "quillan_12mb_training_dataset.jsonl",# Quillan persona/instruct
    "Quillan_Ronin_v5.3.1_Samurai_Training_Seed_Dataset.jsonl",  # Seed data
    "instruct_train.jsonl",               # General instruction following
    "full_train.jsonl",                   # Full instruct mix
]

CKPT_DIR     = os.path.join(REPO_ROOT, "checkpoints", "checkpoints_sft")
RUN1_BEST    = os.path.join(CKPT_DIR, "quillan_sft_v3_best.pt")
RUN2_BEST    = os.path.join(CKPT_DIR, "quillan_dialogue_best.pt")
RUN2_LATEST  = os.path.join(CKPT_DIR, "quillan_dialogue_latest.pt")

# ─── HELPERS ─────────────────────────────────────────────────────────────────

def save_checkpoint(model, path, step, loss, best_loss):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'step': step,
        'loss': loss,
        'best_loss': best_loss,
        'version': 'v5.3.1-dialogue-run2'
    }, path)
    print(f"[SAVE] {os.path.basename(path)}  step={step}  loss={loss:.4f}", flush=True)


def load_dialogue_dataset():
    """Load only dialogue-format samples, weighted toward GPT_5.5_Distilled."""
    data_dir = os.path.join(REPO_ROOT, "training_data")
    enc = tiktoken.get_encoding("gpt2")
    samples = []
    
    for fname in DIALOGUE_DATASETS:
        fpath = os.path.join(data_dir, fname)
        if not os.path.exists(fpath):
            print(f"[DATA] SKIP (not found): {fname}", flush=True)
            continue
        
        count = 0
        with open(fpath, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    
                    # Pre-tokenized sample
                    tokens = obj.get("input_ids", obj.get("tokens", []))
                    labels = obj.get("labels", obj.get("target_ids", tokens))
                    
                    # Raw text sample - tokenize on the fly
                    if not tokens:
                        text = obj.get("text", obj.get("content", ""))
                        if text and len(text) > 32:
                            tokens = enc.encode(text)[:SEQ_LEN]
                            labels = tokens
                    
                    if len(tokens) >= 32:
                        samples.append((tokens, labels))
                        count += 1
                except Exception:
                    pass
        
        if count > 0:
            print(f"[DATA] {fname}: {count} samples", flush=True)
        
        # Cap at 200K total
        if len(samples) >= 200000:
            break
    
    # Shuffle all samples so dialogue is interleaved
    random.shuffle(samples)
    print(f"[DATA] Total dialogue samples: {len(samples)}", flush=True)
    return samples


def generate_sample(model, enc, prompt, max_tokens=50, temp=0.7, top_p=0.9):
    model.eval()
    tokens = enc.encode(prompt)
    with torch.no_grad():
        for _ in range(max_tokens):
            inp = torch.tensor([tokens[-512:]], dtype=torch.long)
            logits = model(inp)[:, -1, :] / temp
            probs = F.softmax(logits, dim=-1)
            sorted_p, sorted_i = torch.sort(probs, descending=True)
            cum = torch.cumsum(sorted_p, dim=-1)
            mask = cum > top_p
            mask[..., 1:] = mask[..., :-1].clone()
            mask[..., 0] = 0
            logits_masked = logits.scatter(1, sorted_i, logits.gather(1, sorted_i).masked_fill(mask, float('-inf')))
            probs = F.softmax(logits_masked, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1).item()
            if next_tok == 50256:
                break
            tokens.append(next_tok)
    suffix = enc.decode(tokens[len(enc.encode(prompt)):])
    return suffix.replace('\ufffd', '?')


def get_lr(step):
    if step < WARMUP_STEPS:
        return LR * (step + 1) / WARMUP_STEPS
    progress = (step - WARMUP_STEPS) / max(1, TOTAL_STEPS - WARMUP_STEPS)
    return MIN_LR + 0.5 * (LR - MIN_LR) * (1.0 + math.cos(math.pi * progress))


# ─── MAIN ───────────────────────────────────────────────────────────────────

def main():
    enc = tiktoken.get_encoding("gpt2")
    
    # Build model (must match Run 1 config exactly)
    cfg = QuillanArchConfig(
        hidden_dim=1024,
        ffn_dim=2048,
        num_experts=34,
        text_only=True,
        eggroll_rank=256,
    )
    model = QuillanRoninSovereign(cfg)
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[MODEL] {total_params:.1f}M parameters", flush=True)
    
    # Load Run 1 best checkpoint
    if os.path.exists(RUN1_BEST):
        print(f"[BASE] Loading Run 1 best checkpoint: {RUN1_BEST}", flush=True)
        ckpt = torch.load(RUN1_BEST, map_location="cpu", weights_only=False)
        sd   = ckpt.get("model_state_dict", ckpt)
        msd  = model.state_dict()
        loaded = 0
        for k, v in sd.items():
            if k in msd and v.shape == msd[k].shape:
                msd[k].copy_(v)
                loaded += 1
        model.load_state_dict(msd)
        best_r1 = ckpt.get("best_loss", 999.0)
        print(f"[BASE] Loaded {loaded} layers. Run 1 best loss was {best_r1:.4f}", flush=True)
    else:
        print(f"[WARN] Run 1 checkpoint not found: {RUN1_BEST}", flush=True)
        print(f"[WARN] Starting from scratch. Make sure Run 1 is complete!", flush=True)
    
    # Load dialogue dataset
    samples = load_dialogue_dataset()
    if not samples:
        print("[FATAL] No dialogue data found!", flush=True)
        sys.exit(1)
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    
    model.train()
    best_loss = 999.0
    sample_idx = 0
    start_time = time.time()
    
    # Test prompts
    test_prompts = [
        "<|user|>\nHello! Who are you?\n<|assistant|>\n",
        "<|user|>\nWhat can you do?\n<|assistant|>\n",
        "<|user|>\nExplain quantum entanglement briefly.\n<|assistant|>\n",
    ]
    
    print(f"\n[TRAIN] Dialogue SFT  0 -> {TOTAL_STEPS} steps  ({len(samples)} samples)", flush=True)
    print(f"[TRAIN] LR={LR:.1e}  warmup={WARMUP_STEPS}  grad_accum={GRAD_ACCUM}", flush=True)
    
    for step in range(1, TOTAL_STEPS + 1):
        lr = get_lr(step - 1)
        for pg in optimizer.param_groups:
            pg['lr'] = lr
        
        optimizer.zero_grad()
        accum_loss = 0.0
        
        for _ in range(GRAD_ACCUM):
            tokens, labels = samples[sample_idx % len(samples)]
            sample_idx += 1
            
            seq_len  = min(SEQ_LEN, len(tokens))
            txt_in   = torch.tensor([tokens[:seq_len]], dtype=torch.long)
            tgt_in   = torch.tensor([labels[:seq_len]], dtype=torch.long)
            
            logits   = model(txt_in)
            loss     = F.cross_entropy(
                logits[..., :-1, :].contiguous().view(-1, cfg.vocab_size),
                tgt_in[..., 1:].contiguous().view(-1),
                ignore_index=-100
            )
            (loss / GRAD_ACCUM).backward()
            accum_loss += loss.item() / GRAD_ACCUM
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # Logging
        if step <= 10 or step % LOG_EVERY == 0:
            elapsed = time.time() - start_time
            sps = elapsed / step
            eta_h = (TOTAL_STEPS - step) * sps / 3600
            print(f"  step {step:5d}/{TOTAL_STEPS}  loss={accum_loss:.4f}  lr={lr:.2e}  {sps:.1f}s/st  ETA:{eta_h:.1f}h", flush=True)
        
        # Save best
        if accum_loss < best_loss and step >= 25:
            best_loss = accum_loss
            save_checkpoint(model, RUN2_BEST, step, accum_loss, best_loss)
        
        # Save periodic
        if step % SAVE_EVERY == 0:
            save_checkpoint(model, RUN2_LATEST, step, accum_loss, best_loss)
        
        # Text generation sample
        if step % SAMPLE_EVERY == 0:
            for prompt in test_prompts:
                out = generate_sample(model, enc, prompt, max_tokens=50)
                label = prompt.split("\n")[1]
                print(f"  [SAMPLE '{label}'] → {out[:100]}", flush=True)
            model.train()
    
    # Final save
    save_checkpoint(model, RUN2_LATEST, TOTAL_STEPS, accum_loss, best_loss)
    print(f"\n[COMPLETE] Dialogue SFT Run 2 finished! Best loss: {best_loss:.4f}", flush=True)
    
    # Final samples
    print("\n[FINAL SAMPLES]", flush=True)
    for prompt in test_prompts:
        out = generate_sample(model, enc, prompt, max_tokens=80)
        print(f"\n  Q: {prompt.split(chr(10))[1]}\n  A: {out[:200]}", flush=True)

if __name__ == "__main__":
    main()
