#!/usr/bin/env python3
"""
Quillan-Ronin v5.3.1 — Conversational Target-Masked SFT Script
Enforces strict target masking on System and User tokens (-100) so 100% of autograd
gradients focus on producing fluent, natural, and human-like Assistant dialogue responses.
"""

import os
import sys
import time
import glob
import json
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / '_dev'))

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from _dev.quillan_v8_saturated import QuillanRoninSovereign, QuillanArchConfig
from _dev.quillan_bpe_tokenizer import QuillanBPETokenizer

def load_dialogue_samples(jsonl_path, tokenizer, max_seq_len=512, max_samples=15000):
    samples = []
    print(f"[DATA] Loading conversational turns from {Path(jsonl_path).name}...", flush=True)
    with open(jsonl_path, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                messages = data.get('messages', [])
                if not messages:
                    continue

                full_token_ids = []
                full_target_ids = []

                for msg in messages:
                    role = msg.get('role', '')
                    content = msg.get('content', '')
                    if not content:
                        continue

                    prompt_text = f"<|{role}|>\n{content}\n"
                    t_ids = tokenizer.encode(prompt_text)

                    if role in ['system', 'user']:
                        # Mask prompt tokens in loss calculation
                        mask_ids = [-100] * len(t_ids)
                        full_token_ids.extend(t_ids)
                        full_target_ids.extend(mask_ids)
                    elif role == 'assistant':
                        # Train 100% autograd gradient on assistant response tokens
                        full_token_ids.extend(t_ids)
                        full_target_ids.extend(t_ids)

                if len(full_token_ids) > 1:
                    # Truncate to max_seq_len
                    input_ids = full_token_ids[:max_seq_len - 1]
                    target_ids = full_target_ids[1:max_seq_len]
                    input_ids = input_ids[:len(target_ids)]

                    # Ensure there is at least one assistant target token to train on
                    if any(t != -100 for t in target_ids):
                        samples.append((input_ids, target_ids))
                        if len(samples) >= max_samples:
                            break
            except Exception:
                continue

    print(f"[DATA] Successfully packed {len(samples)} target-masked conversational turns.", flush=True)
    return samples

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Quillan Fluent Conversational Dialogue SFT")
    parser.add_argument("--steps", type=int, default=2000, help="Number of dialogue SFT steps")
    parser.add_argument("--lr", type=float, default=1.5e-4, help="Learning rate")
    parser.add_argument("--seq-len", type=int, default=512, help="Sequence length")
    args = parser.parse_args()

    print("=" * 65)
    print("  QUILLAN-RONIN v5.3.1 — FLUENT DIALOGUE CONVERSATIONAL SFT")
    print(f"  Target: {args.steps} steps | LR: {args.lr} | Seq Len: {args.seq_len}")
    print("=" * 65)

    device = 'cuda' if (torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 7) else 'cpu'
    print(f"[HARDWARE] Compute device: {device}")

    cfg = QuillanArchConfig(device=device, text_only=True, eggroll_rank=16)
    model = QuillanRoninSovereign(cfg).to(device)

    tok_path = ROOT / '_dev' / 'quillan_bpe_tokenizer_hf' / 'tokenizer.json'
    tokenizer = QuillanBPETokenizer()
    tokenizer.load(str(tok_path))

    # 1. Load Base Checkpoint
    base_path = ROOT / 'checkpoints' / 'checkpoints_v2' / 'quillan_full_base_final.pt'
    print(f"\n[1/2 BASE LOAD] {base_path.name}...")
    base_ckpt = torch.load(str(base_path), map_location=device, weights_only=False)
    base_sd = base_ckpt.get('model_state_dict', base_ckpt)
    model_sd = model.state_dict()
    base_loaded = sum(1 for k, v in base_sd.items() if k in model_sd and v.shape == model_sd[k].shape)
    for k, v in base_sd.items():
        if k in model_sd and v.shape == model_sd[k].shape:
            model_sd[k].copy_(v)
    print(f"[1/2 BASE LOAD] Loaded {base_loaded} / 662 base keys.")

    # 2. Overlay SFT Checkpoint
    sft_path = ROOT / 'checkpoints' / 'checkpoints_sft' / 'quillan_causal_aligned.pt'
    print(f"[2/2 SFT LOAD] {sft_path.name}...")
    sft_ckpt = torch.load(str(sft_path), map_location=device, weights_only=False)
    sft_sd = sft_ckpt.get('model_state_dict', sft_ckpt)
    sft_loaded = sum(1 for k, v in sft_sd.items() if k in model_sd and v.shape == model_sd[k].shape)
    for k, v in sft_sd.items():
        if k in model_sd and v.shape == model_sd[k].shape:
            model_sd[k].copy_(v)
    print(f"[2/2 SFT LOAD] Overlaid {sft_loaded} SFT adapter keys.")

    # Load Conversational SFT Samples from instruct_train.jsonl
    jsonl_path = ROOT / 'training_data' / 'instruct_train.jsonl'
    samples = load_dialogue_samples(jsonl_path, tokenizer, max_seq_len=args.seq_len, max_samples=15000)
    if not samples:
        print("[ERROR] No target-masked dialogue samples loaded!")
        sys.exit(1)

    model.train()
    trainable_params = []
    for name, param in model.named_parameters():
        if any(k in name for k in ['lora', 'router', 'decomposition', 'prism', 'txt_dec', 'ingestion', 'eggroll', 'swarm', 'A', 'B', 'C', 'D']):
            param.requires_grad = True
            trainable_params.append(param)
        else:
            param.requires_grad = False

    trainable_count = sum(p.numel() for p in trainable_params)
    print(f"[MODEL] Trainable alignment parameters: {trainable_count / 1e6:.1f}M / 453.9M", flush=True)

    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps, eta_min=1e-6)

    save_dir = ROOT / 'checkpoints' / 'checkpoints_sft'
    save_dir.mkdir(parents=True, exist_ok=True)
    aligned_ckpt_path = save_dir / 'quillan_causal_aligned.pt'

    print(f"\n[TRAIN] Starting {args.steps}-step Conversational Target-Masked SFT...", flush=True)
    t0 = time.time()
    ema_loss = None

    for step in range(1, args.steps + 1):
        input_ids_list, target_ids_list = samples[(step - 1) % len(samples)]
        input_ids = torch.tensor([input_ids_list], dtype=torch.long, device=device)
        target_ids = torch.tensor([target_ids_list], dtype=torch.long, device=device)

        optimizer.zero_grad()
        out = model(input_ids)
        logits = out["logits"]  # [1, L, V]

        loss = F.cross_entropy(logits.reshape(-1, cfg.vocab_size), target_ids.reshape(-1), ignore_index=-100)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_params, 0.5)
        optimizer.step()
        scheduler.step()

        loss_val = loss.item()
        ema_loss = loss_val if ema_loss is None else 0.95 * ema_loss + 0.05 * loss_val

        if step % 25 == 0 or step == args.steps:
            elapsed = time.time() - t0
            sec_per_step = elapsed / step
            eta_hours = (args.steps - step) * sec_per_step / 3600.0
            print(f"  step {step:4d}/{args.steps}  loss={ema_loss:.4f}  resp_ce={loss_val:.4f}  {sec_per_step:.3f}s/st  ETA:{eta_hours:.1f}h", flush=True)

        if step % 100 == 0 or step == args.steps:
            torch.save({"model_state_dict": model.state_dict(), "step": step, "loss": ema_loss}, str(aligned_ckpt_path))

    print("\n" + "=" * 65)
    print("  CONVERSATIONAL TARGET-MASKED SFT COMPLETE")
    print(f"  Saved final checkpoint: {aligned_ckpt_path.name}")
    print("=" * 65, flush=True)

if __name__ == '__main__':
    main()
