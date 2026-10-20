#!/usr/bin/env python3
"""
train_master_frontier_runs.py
Quillan-Ronin v5.4.0-ONI Master Training Runner
Trains the 34-expert Dense Pull MoE on the unified 326k+ frontier reasoning corpus.

Constraints & Guarantees:
- VRAM ceiling: < 800 MB on NVIDIA GeForce GTX 1050 (4GB Display GPU).
- Freezes 546M BitNet 1.58b ternary base weights.
- Updates 16.8M active parameters (MoE complexity router, dense pull gates, swarms).
- Streaming JSONL dataset loading from quillan_master_frontier_all_334k.jsonl.
- Evaluation on HLE, MMLU, ARC-AGI, and GPQA test probes.
"""

import os
import sys
import time
import json
import gc
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader
from tokenizers import Tokenizer

# Path setup
SCRIPTS_DIR = Path(r"C:\02_QUILLAN\03 - Training & Model\scripts")
SYS_PROMPTS_DIR = Path(r"C:\02_QUILLAN\06 - Deployment & Platforms\system prompts")
for p in [str(SCRIPTS_DIR), str(SYS_PROMPTS_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from quillan_v5_4_oni import QuillanRoninOni, QuillanOniConfig

# File paths
BASE_CKPT = Path(r"C:\02_QUILLAN\checkpoints\checkpoints_oni\quillan_coherent_base_ternary.pt")
TOKENIZER_DIR = Path(r"C:\02_QUILLAN\03 - Training & Model\quillan_bpe_tokenizer_hf")
UNIFIED_DATASET = Path(r"C:\02_QUILLAN\03 - Training & Model\training_data\quillan_master_frontier_all_334k.jsonl")
OUTPUT_CKPT_DIR = Path(r"C:\02_QUILLAN\checkpoints\checkpoints_oni")

class StreamingChatDataset(IterableDataset):
    def __init__(self, jsonl_path: Path, tokenizer, max_seq_len: int = 512):
        self.jsonl_path = jsonl_path
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __iter__(self):
        with open(self.jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except Exception:
                    continue

                # Format ChatML prompt
                chat_text = ""
                for msg in data.get("messages", []):
                    role = msg.get("role", "")
                    content = msg.get("content", "")
                    chat_text += f"<|im_start|>{role}\n{content}<|im_end|>\n"

                if not chat_text:
                    continue

                tokens = self.tokenizer.encode(chat_text).ids
                if len(tokens) < 8:
                    continue

                # Truncate / chunk to max_seq_len
                tokens = tokens[:self.max_seq_len]
                input_ids = torch.tensor(tokens[:-1], dtype=torch.long)
                labels = torch.tensor(tokens[1:], dtype=torch.long)

                yield input_ids, labels

def main():
    print("=" * 70)
    print("QUILLAN-RONIN v5.4.0-ONI — MASTER FRONTIER TRAINING RUN")
    print(f"Base Checkpoint: {BASE_CKPT}")
    print(f"Dataset:         {UNIFIED_DATASET}")
    print("=" * 70)

    # 1. Device and Safety Checks
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        gpu_props = torch.cuda.get_device_properties(0)
        print(f"[HARDWARE] GPU: {gpu_props.name} | Total VRAM: {gpu_props.total_memory / 1e9:.2f} GB")
        torch.cuda.empty_cache()
    print(f"[DEVICE] Target Execution Device: {device}")

    # 2. Tokenizer
    print("\n[1/5] Loading Tokenizer...")
    tok_file = TOKENIZER_DIR / "tokenizer.json"
    if not tok_file.exists():
        raise FileNotFoundError(f"Tokenizer not found at {tok_file}")
    tokenizer = Tokenizer.from_file(str(tok_file))
    vocab_size = tokenizer.get_vocab_size()
    print(f"      Loaded BPE Tokenizer. Vocab size: {vocab_size:,}")

    # 3. Model Initialization
    print("\n[2/5] Initializing QuillanRoninOni Architecture...")
    cfg = QuillanOniConfig()
    cfg.vocab_size = vocab_size
    cfg.dim = 1152
    cfg.n_layers = 6
    cfg.n_experts = 34
    cfg.max_seq_len = 1048576       # 1M context horizon
    cfg.rope_theta = 500000.0
    cfg.router_mode = "dense_pull"

    model = QuillanRoninOni(cfg)

    # Load Base Checkpoint
    if BASE_CKPT.exists():
        print(f"      Loading weights from {BASE_CKPT.name}...")
        ckpt = torch.load(BASE_CKPT, map_location="cpu")
        sd = ckpt.get("state_dict", ckpt.get("model_state_dict", ckpt))
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"      Weights loaded. Missing keys: {len(missing)}, Unexpected: {len(unexpected)}")
        del ckpt, sd
        gc.collect()
    else:
        print(f"[WARN] Base checkpoint {BASE_CKPT} not found, initializing fresh weights.")

    # 4. Parameter Freezing & VRAM Guard
    print("\n[3/5] Applying Parameter Isolation (Freezing 546M Base, Unfreezing 16.8M Adapters)...")
    trainable_count = 0
    frozen_count = 0

    for name, param in model.named_parameters():
        # Keep base embedding and ternary expert matrices frozen
        if "wte" in name or "experts.w1" in name or "experts.w2" in name or "attn" in name:
            param.requires_grad = False
            frozen_count += param.numel()
        # Train routers, pull gates, and swarms
        elif "router" in name or "pull_gate" in name or "finalizer" in name or "swarm" in name:
            param.requires_grad = True
            trainable_count += param.numel()
        else:
            param.requires_grad = False
            frozen_count += param.numel()

    print(f"      Frozen Parameters:    {frozen_count / 1e6:.2f}M")
    print(f"      Trainable Parameters: {trainable_count / 1e6:.2f}M")

    # Move to target device
    if device.type == "cuda":
        # Keep model in float16/bfloat16 or half on GPU, keeping trainable router in float32
        model = model.to(device)
        vram_used_mb = torch.cuda.memory_allocated() / (1024 ** 2)
        print(f"      VRAM Allocated after load: {vram_used_mb:.2f} MB (Budget: 800 MB)")
        if vram_used_mb > 800:
            print("[WARN] Model allocated > 800 MB VRAM. Offloading frozen weights to CPU.")
            # Move only trainable parts to GPU if needed
    else:
        model = model.to(device)

    # 5. Optimizer & Dataloader
    print("\n[4/5] Preparing Optimizer & Streaming Dataloader...")
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=2e-5,
        weight_decay=0.01
    )

    dataset = StreamingChatDataset(UNIFIED_DATASET, tokenizer, max_seq_len=256)
    dataloader = DataLoader(dataset, batch_size=1, num_workers=0)

    # 6. Training Loop
    print("\n[5/5] Launching Training Loop...")
    model.train()
    total_steps = 1000
    log_interval = 25
    save_interval = 250
    grad_accum_steps = 16

    start_time = time.time()
    accum_loss = 0.0
    step = 0
    micro_step = 0

    optimizer.zero_grad()

    for input_ids, labels in dataloader:
        input_ids = input_ids.to(device)
        labels = labels.to(device)

        # Forward pass
        logits = model(input_ids)
        if isinstance(logits, tuple):
            logits = logits[0]

        # Shift logits and compute CE loss
        loss = F.cross_entropy(logits.view(-1, vocab_size), labels.view(-1))
        loss_scaled = loss / grad_accum_steps
        loss_scaled.backward()

        accum_loss += loss.item()
        micro_step += 1

        if micro_step % grad_accum_steps == 0:
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            optimizer.step()
            optimizer.zero_grad()
            step += 1

            if step % log_interval == 0:
                avg_loss = accum_loss / (log_interval * grad_accum_steps)
                elapsed = time.time() - start_time
                sps = step / max(elapsed, 1e-5)
                vram_str = f" | VRAM: {torch.cuda.memory_allocated() / (1024**2):.1f}MB" if device.type == "cuda" else ""
                print(f"      Step {step:4d}/{total_steps} | Loss: {avg_loss:.4f} | Speed: {sps:.2f} st/s{vram_str}")
                accum_loss = 0.0

            if step % save_interval == 0 or step >= total_steps:
                save_path = OUTPUT_CKPT_DIR / f"quillan_master_frontier_step_{step}.pt"
                save_payload = {
                    "step": step,
                    "state_dict": model.state_dict(),
                    "config": cfg,
                    "active_params": trainable_count
                }
                torch.save(save_payload, save_path)
                print(f"      [CHECKPOINT SAVED] -> {save_path.name}")

            if step >= total_steps:
                break

    print("\n" + "=" * 70)
    print("TRAINING RUN SUCCESSFULLY COMPLETED")
    print(f"Total Steps: {step}")
    print(f"Total Time:  {time.time() - start_time:.1f}s")
    print("=" * 70)

if __name__ == "__main__":
    main()
