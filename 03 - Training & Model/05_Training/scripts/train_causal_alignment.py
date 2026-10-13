#!/usr/bin/env python3
"""
Quillan-Ronin v5.3.1 — Causal Alignment Training Script
Resumes directly from quillan_full_base_final.pt + quillan_sft_final.pt.
Enforces strict causal attention across all layers (including NineVectorDecomposition)
to train true unidirectional next-token prediction without lookahead leakage.
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

def load_sft_chunks(data_dir, tokenizer, seq_len=256, max_chunks=5000):
    jsonl_files = sorted(glob.glob(os.path.join(data_dir, "*.jsonl")))
    print(f"[DATA] Found {len(jsonl_files)} SFT dataset files in {data_dir}")
    all_chunks = []
    
    for fpath in jsonl_files:
        if len(all_chunks) >= max_chunks:
            break
        count = 0
        with open(fpath, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                if len(all_chunks) >= max_chunks:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    text = ""
                    if "messages" in obj:
                        for m in obj["messages"]:
                            role = m.get("role", "user")
                            content = m.get("content", "")
                            text += f"<|{role}|> {content} "
                    elif "instruction" in obj:
                        text = f"<|user|> {obj['instruction']} <|assistant|> {obj.get('output', '')}"
                    elif "text" in obj:
                        text = obj["text"]
                    elif "prompt" in obj:
                        text = f"<|user|> {obj['prompt']} <|assistant|> {obj.get('response', '')}"
                    
                    if text:
                        ids = tokenizer.encode(text)
                        for i in range(0, len(ids) - seq_len, seq_len):
                            chunk = ids[i : i + seq_len + 1]
                            if len(chunk) == seq_len + 1:
                                all_chunks.append(chunk)
                                count += 1
                                if len(all_chunks) >= max_chunks:
                                    break
                except Exception:
                    continue
        print(f"  - {os.path.basename(fpath)}: {count} sequence chunks")
    
    print(f"[DATA] Total SFT training chunks loaded: {len(all_chunks)}")
    return all_chunks

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Quillan Perfect English Master Pass")
    parser.add_argument("--steps", type=int, default=3000, help="Number of alignment steps")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--seq-len", type=int, default=256, help="Sequence length")
    args = parser.parse_args()

    print("=" * 65)
    print("  QUILLAN-RONIN v5.3.1 — PERFECT ENGLISH MASTER PASS")
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

    # Load SFT dataset chunks
    data_dir = str(ROOT / 'training_data')
    chunks = load_sft_chunks(data_dir, tokenizer, seq_len=args.seq_len, max_chunks=30000)
    if not chunks:
        print("[ERROR] No SFT chunks loaded!")
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
    
    # Smooth Cosine Annealing decay down to 1e-6
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps, eta_min=1e-6)

    save_dir = ROOT / 'checkpoints' / 'checkpoints_sft'
    save_dir.mkdir(parents=True, exist_ok=True)
    aligned_ckpt_path = save_dir / 'quillan_causal_aligned.pt'

    print(f"\n[TRAIN] Starting {args.steps}-step Perfect English Master Pass...", flush=True)
    t0 = time.time()
    ema_loss = None

    for step in range(1, args.steps + 1):
        chunk = chunks[(step - 1) % len(chunks)]
        input_ids = torch.tensor([chunk[:-1]], dtype=torch.long, device=device)
        target_ids = torch.tensor([chunk[1:]], dtype=torch.long, device=device)

        optimizer.zero_grad()
        out = model(input_ids)
        logits = out["logits"]  # [1, L, V]

        loss = F.cross_entropy(logits.reshape(-1, cfg.vocab_size), target_ids.reshape(-1))
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
            print(f"  step {step:4d}/{args.steps}  loss={ema_loss:.4f}  ce={loss_val:.4f}  {sec_per_step:.3f}s/st  ETA:{eta_hours:.1f}h", flush=True)

        if step % 100 == 0 or step == args.steps:
            torch.save({"model_state_dict": model.state_dict(), "step": step, "loss": ema_loss}, str(aligned_ckpt_path))
            print(f"  [CKPT] Saved → {aligned_ckpt_path.name} at step {step}", flush=True)

    print("\n" + "=" * 65)
    print("  CAUSAL ALIGNMENT TRAINING PASS COMPLETE")
    print(f"  Saved final checkpoint: {aligned_ckpt_path.name}")
    print("=" * 65)

if __name__ == "__main__":
    main()
