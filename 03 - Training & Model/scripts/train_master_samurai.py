#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_master_samurai.py  -  Master Pretraining / SFT Engine for Quillan-Ronin (~204.92M Tokens)
================================================================================================
Trains either the 6L Mini (~520.4M) or 12L Main (~1.163B) model over the complete Samurai Set:
  - Claude Fable 5 & Opus Multi-turn Reasoning
  - GPT Sol 5.6 / 5.5 Distilled Reasoning
  - Quillan Master Sovereign Corpus V7
  - Formal Reasoning Gym & Science Sets
  - Powered by Sovereign Muon-K2 + AdamW Fused Optimizer
================================================================================================
"""

import os
import sys
import time
import math
import argparse
from pathlib import Path
from datetime import datetime

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

REPO_ROOT = Path(r"C:\02_QUILLAN")
SCRIPTS_DIR = REPO_ROOT / "03 - Training & Model" / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
if str(REPO_ROOT / "03 - Training & Model") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "03 - Training & Model"))

from quillan_v5_4_oni import QuillanOniConfig, QuillanRoninOni
from samurai_unified_dataset import SamuraiStreamingDataset
from quillan_fused_optimizer import create_quillan_muonk2_optimizer


def parse_args():
    parser = argparse.ArgumentParser(description="Master Training Harness for Quillan-Ronin")
    parser.add_argument("--model_size", type=str, choices=["6l", "12l"], default="6l",
                        help="Model variant: '6l' (~520M params) or '12l' (~1.16B params)")
    parser.add_argument("--batch_size", type=int, default=2, help="Micro batch size per step")
    parser.add_argument("--accum_steps", type=int, default=16, help="Gradient accumulation steps (effective batch = batch * accum)")
    parser.add_argument("--lr", type=float, default=2.5e-4, help="Peak learning rate for AdamW components")
    parser.add_argument("--lr_muon", type=float, default=0.015, help="Peak learning rate for Muon matrix components")
    parser.add_argument("--seq_len", type=int, default=512, help="Context sequence length")
    parser.add_argument("--max_steps", type=int, default=12500, help="Maximum training steps (covers full ~200M tokens)")
    parser.add_argument("--warmup_steps", type=int, default=250, help="Linear warmup steps")
    parser.add_argument("--save_every", type=int, default=250, help="Checkpoint saving frequency (steps)")
    parser.add_argument("--probe_every", type=int, default=50, help="Generation probing frequency (steps)")
    parser.add_argument("--priority", type=str, default="BELOW_NORMAL",
                        choices=["HIGH", "ABOVE_NORMAL", "NORMAL", "BELOW_NORMAL", "IDLE"],
                        help="Win32 process priority (default: BELOW_NORMAL to keep desktop fluid)")
    parser.add_argument("--num_threads", type=int, default=2,
                        help="CPU threads for PyTorch (default: 2, leaves 2 cores free for desktop/browser)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to train on ('cuda' or 'cpu')")
    parser.add_argument("--output_dir", type=str, default=str(REPO_ROOT / "checkpoints" / "checkpoints_oni"))
    return parser.parse_args()




def get_cosine_lr(step: int, warmup_steps: int, max_steps: int, base_lr: float, min_lr_ratio: float = 0.1) -> float:
    """Cosine learning rate schedule with linear warmup."""
    if step < warmup_steps:
        return base_lr * (step + 1) / max(1, warmup_steps)
    if step >= max_steps:
        return base_lr * min_lr_ratio
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return base_lr * (min_lr_ratio + coeff * (1.0 - min_lr_ratio))


def probe_generation(model: QuillanRoninOni, tokenizer, prompt_str: str, device: str, max_new_tokens: int = 40):
    """Probes the model with top-p generation to monitor fluency and routing health."""
    model.eval()
    try:
        input_ids = tokenizer.encode(prompt_str).ids
        x = torch.tensor([input_ids], dtype=torch.long, device=device)
        with torch.no_grad():
            for _ in range(max_new_tokens):
                out = model(x)
                logits = out[0] if isinstance(out, tuple) else out
                next_token_logits = logits[0, -1, :] / 0.7  # temp 0.7
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                x = torch.cat([x, next_token.unsqueeze(0)], dim=1)
                if next_token.item() == 0:  # EOS
                    break
        decoded = tokenizer.decode(x[0].tolist())
        return decoded
    except Exception as e:
        return f"[PROBE ERROR: {e}]"
    finally:
        model.train()


def train():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 75)
    print("QUILLAN-RONIN MASTER SAMURAI TRAINING ENGINE")
    print(f"Timestamp   : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Model Class : {args.model_size.upper()}")
    print(f"Device      : {args.device}")
    print(f"Context Len : {args.seq_len}")
    print(f"Target Token Budget: ~204.92 Million Tokens")
    # 0. Hardware and Win32 Performance Optimization
    if args.num_threads > 0:
        torch.set_num_threads(args.num_threads)
        print(f"[HARDWARE] PyTorch CPU Threads clamped to: {args.num_threads} (leaving remaining cores for desktop)")

    try:
        from quillan_pc_toolkit import QuillanPCHardwareToolkit
        pc_toolkit = QuillanPCHardwareToolkit()
        compaction = pc_toolkit.apply_performance_profile(priority=args.priority)
        print(f"[HARDWARE] Win32 Priority: {compaction.priority_class_applied} | Reclaimed Standby RAM: {compaction.memory_reclaimed_mb:.1f} MB")
    except Exception as e:
        print(f"[HARDWARE] Notice: Native toolkit hook deferred ({e})")


    # 1. Instantiate Model Config
    if args.model_size == "6l":
        cfg = QuillanOniConfig.mini_6l(device=args.device, max_seq_len=args.seq_len)
    else:
        cfg = QuillanOniConfig.main_12l(device=args.device, max_seq_len=args.seq_len)

    model = QuillanRoninOni(cfg)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"[MODEL] Total Parameters    : {total_params:,} ({total_params/1e6:.1f}M | {total_params/1e9:.3f}B)")
    print(f"[MODEL] Trainable Parameters: {trainable_params:,} ({trainable_params/1e6:.1f}M)")

    if args.device == "cuda":
        model = model.cuda()
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")

    # 2. Instantiate Sovereign Muon-K2 + AdamW Fused Optimizer
    optimizer = create_quillan_muonk2_optimizer(
        model,
        lr=args.lr,
        lr_muon=args.lr_muon,
        weight_decay=0.01,
        ns_steps=5,
    )
    print(f"[OPTIMIZER] Sovereign Muon-K2 + AdamW Initialized (lr={args.lr}, lr_muon={args.lr_muon})")

    # 3. Instantiate Unified Dataset Streamer
    ds = SamuraiStreamingDataset(seq_len=args.seq_len, shuffle_files=True)
    loader = DataLoader(ds, batch_size=args.batch_size)

    # 4. Training Loop
    model.train()
    step = 0
    running_loss = 0.0
    tokens_seen = 0
    t0 = time.time()

    print(f"[TRAIN] Launching unified training loop across {len(ds.files)} datasets...")

    optimizer.zero_grad(set_to_none=True)
    accum_count = 0

    for x_batch, y_batch in loader:
        x_batch = x_batch.to(args.device)
        y_batch = y_batch.to(args.device)

        # Forward pass
        output = model(x_batch, labels=y_batch)
        if isinstance(output, tuple):
            loss = output[1]  # loss is 2nd element when labels passed
        else:
            loss = output

        if torch.isnan(loss) or torch.isinf(loss):
            print(f"[WARN] Non-finite loss encountered at step {step}: {loss.item()}. Skipping step.")
            optimizer.zero_grad(set_to_none=True)
            accum_count = 0
            continue

        loss_scaled = loss / args.accum_steps
        loss_scaled.backward()
        accum_count += 1
        running_loss += loss.item()
        tokens_seen += (y_batch != -100).sum().item()

        if accum_count >= args.accum_steps:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # Dynamic LR Schedule
            current_lr = get_cosine_lr(step, args.warmup_steps, args.max_steps, args.lr)
            current_lr_muon = get_cosine_lr(step, args.warmup_steps, args.max_steps, args.lr_muon)
            for pg in optimizer.param_groups:
                pg["lr"] = current_lr_muon if pg.get("is_muon", False) else current_lr

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            accum_count = 0
            step += 1

            # Progress Logging
            if step % 10 == 0 or step == 1:
                elapsed = time.time() - t0
                tok_per_sec = tokens_seen / max(1.0, elapsed)
                avg_loss = running_loss / (10 if step > 1 else 1)
                running_loss = 0.0
                print(
                    f"Step {step:6d}/{args.max_steps:6d} | "
                    f"Loss: {avg_loss:6.4f} | "
                    f"LR: {current_lr:.2e} | "
                    f"Tokens: {tokens_seen/1e6:6.2f}M | "
                    f"Throughput: {tok_per_sec:6.1f} tok/s | "
                    f"Elapsed: {elapsed/60:4.1f}m"
                )

            # Probe Generation
            if step % args.probe_every == 0:
                probe_prompt = "<|user|>\nWhat is the nature of sovereignty in the Ronin code?\n<|assistant|>\n"
                gen_text = probe_generation(model, ds.tokenizer, probe_prompt, args.device)
                print("-" * 75)
                print(f"[PROBE STEP {step}] Generation output:")
                print(gen_text[:250].replace("\n", " "))
                print("-" * 75)

            # Checkpoint Save
            if step % args.save_every == 0 or step == args.max_steps:
                ckpt_name = f"quillan_{args.model_size}_samurai_step_{step}.pt"
                ckpt_path = out_dir / ckpt_name
                torch.save({
                    "step": step,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": cfg,
                    "tokens_seen": tokens_seen,
                    "timestamp": datetime.now().isoformat(),
                }, ckpt_path)
                print(f"[CHECKPOINT] Saved: {ckpt_path}")

            if step >= args.max_steps:
                break

    print("=" * 75)
    print(f"[COMPLETE] Master Training Run Completed. Total Tokens Processed: {tokens_seen/1e6:.2f}M")
    print("=" * 75)


if __name__ == "__main__":
    train()
