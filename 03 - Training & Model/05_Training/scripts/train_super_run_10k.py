#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
👑 QUILLAN-RONIN v5.3.1 — QUAD-EPOCH 10,000-STEP SUPER TRAINING PIPELINE
Continuous, high-throughput sovereign pre-training and multi-domain alignment
spanning 10,000 steps with cyclic cosine annealing, AMP scaling, and gradient tracking.
"""

import os
import sys
import math
import time
import json
import random
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

SCRIPTS_DIR = Path(r"C:\02_QUILLAN\scripts")
SCRATCH_DIR = Path(r"C:\Users\Admin\.gemini\antigravity-ide\brain\47d0b748-384a-4cc6-bcb6-f13c52f1d9d9\scratch")
CKPT_DIR = Path(r"C:\02_QUILLAN\checkpoints\checkpoints_sft")
DATA_DIR = Path(r"C:\02_QUILLAN\training_data")
LOG_DIR = Path(r"C:\02_QUILLAN\training_logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(SCRATCH_DIR))

from quillan_v10_unrolled_sovereign import QuillanUnrolledSovereign, QuillanUnrolledConfig
from sovereign_inference_engine import SovereignTokenizer

def log_msg(msg: str):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    formatted = f"[{ts}] {msg}"
    print(formatted, flush=True)
    with open(LOG_DIR / "super_run_10k.log", "a", encoding="utf-8") as f:
        f.write(formatted + "\n")

class SuperTrainingDataset:
    """Combines pre-tokenized tensor streams and target-masked gold instruction pairs."""
    def __init__(self, tokenizer: SovereignTokenizer, max_seq_len: int = 256):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.gold_samples: List[Tuple[torch.Tensor, torch.Tensor]] = []
        self.tensor_chunks: List[torch.Tensor] = []

        log_msg("[*] Loading Sovereign Gold instruction datasets...")
        gold_files = [
            DATA_DIR / "Quillan_Universal_Sovereign_Gold_1000.jsonl",
            DATA_DIR / "Quillan_Direct_Answers_Gold.jsonl"
        ]
        for gf in gold_files:
            if not gf.exists():
                continue
            with open(gf, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        d = json.loads(line)
                        q = d.get("question", d.get("prompt", ""))
                        a = d.get("response", d.get("answer", ""))
                        if not q or not a:
                            continue
                        if not q.startswith("Question:"):
                            prompt_str = f"Question: {q.strip()}\nAnswer:\n"
                        else:
                            prompt_str = q.strip()
                        target_str = a.strip()
                        if not target_str.endswith("<|im_end|>"):
                            target_str += "<|im_end|>"
                        p_toks = tokenizer.encode(prompt_str)
                        t_toks = tokenizer.encode(target_str)
                        full_toks = p_toks + t_toks
                        if len(full_toks) > max_seq_len:
                            full_toks = full_toks[:max_seq_len]
                        
                        input_ids = torch.tensor(full_toks[:-1], dtype=torch.long)
                        labels = torch.tensor(full_toks[1:], dtype=torch.long)
                        labels[:len(p_toks)-1] = -100
                        self.gold_samples.append((input_ids, labels))
                    except Exception:
                        continue
        log_msg(f"[+] Loaded {len(self.gold_samples):,} target-masked gold reasoning samples.")

        log_msg("[*] Mapping pre-tokenized tensor corpora...")
        tensor_files = [
            DATA_DIR / "pristine_frontier_gold_37k.pt",
            DATA_DIR / "clean_unified_multi_frontier.pt",
            DATA_DIR / "full_train.pt",
            DATA_DIR / "instruct_train.pt"
        ]
        for tf in tensor_files:
            if tf.exists():
                try:
                    data = torch.load(str(tf), map_location="cpu", weights_only=False)
                    if isinstance(data, torch.Tensor):
                        self.tensor_chunks.append(data)
                        log_msg(f"    - Loaded {tf.name}: shape {data.shape}")
                    elif isinstance(data, dict) and "tokens" in data:
                        t = data["tokens"]
                        self.tensor_chunks.append(t)
                        log_msg(f"    - Loaded {tf.name}: {t.numel():,} tokens")
                except Exception as e:
                    log_msg(f"    - Error loading {tf.name}: {e}")

    def sample_batch(self, batch_size: int = 4) -> Tuple[torch.Tensor, torch.Tensor]:
        inputs_list = []
        labels_list = []

        for _ in range(batch_size):
            # 60% Gold Reasoner / 40% Large-Scale Token Tensor
            if self.gold_samples and (random.random() < 0.60 or not self.tensor_chunks):
                inp, lab = random.choice(self.gold_samples)
                if len(inp) < self.max_seq_len - 1:
                    pad_len = (self.max_seq_len - 1) - len(inp)
                    inp = F.pad(inp, (0, pad_len), value=50256)
                    lab = F.pad(lab, (0, pad_len), value=-100)
                inputs_list.append(inp.unsqueeze(0))
                labels_list.append(lab.unsqueeze(0))
            else:
                chunk = random.choice(self.tensor_chunks)
                if chunk.dim() == 2 and chunk.size(1) >= self.max_seq_len:
                    row = chunk[random.randint(0, chunk.size(0) - 1)]
                    inp = row[:self.max_seq_len-1].clone().long()
                    lab = row[1:self.max_seq_len].clone().long()
                elif chunk.dim() == 1 and chunk.numel() >= self.max_seq_len:
                    idx = random.randint(0, chunk.numel() - self.max_seq_len - 1)
                    seq = chunk[idx:idx+self.max_seq_len].clone().long()
                    inp = seq[:-1]
                    lab = seq[1:]
                else:
                    inp, lab = random.choice(self.gold_samples)
                inputs_list.append(inp.unsqueeze(0))
                labels_list.append(lab.unsqueeze(0))

        return torch.cat(inputs_list, dim=0), torch.cat(labels_list, dim=0)

def train_super_run():
    log_msg("==================================================================")
    log_msg("   👑 QUILLAN-RONIN v5.3.1 — 10,000-STEP SUPER TRAINING PIPELINE")
    log_msg("==================================================================")

    device = torch.device("cpu")
    tokenizer = SovereignTokenizer("gpt2")
    cfg = QuillanUnrolledConfig()
    model = QuillanUnrolledSovereign(cfg).to(device)

    # Failsafe Resurrection Protocol: Find latest saved milestone or best checkpoint
    milestone_files = sorted(CKPT_DIR.glob("quillan_super_run_step*.pt"), key=lambda p: int(re.search(r'\d+', p.stem).group()) if re.search(r'\d+', p.stem) else 0)
    
    start_step = 0
    best_loss = float("inf")
    
    if milestone_files:
        latest_milestone = milestone_files[-1]
        log_msg(f"[🛡️ WARDEN RESURRECTION] Found latest milestone: {latest_milestone.name}")
        raw_data = torch.load(str(latest_milestone), map_location=device, weights_only=False)
        sd = raw_data.get("model_state_dict", raw_data)
        model.load_state_dict(sd, strict=False)
        start_step = raw_data.get("step", 0)
        best_loss = raw_data.get("loss", float("inf"))
        log_msg(f"[🛡️ WARDEN RESURRECTION] Resuming seamlessly from Step {start_step:,} / 10,000 (Loss: {best_loss:.4f})")
    else:
        best_ckpt = CKPT_DIR / "quillan_super_run_best.pt"
        if best_ckpt.exists():
            log_msg(f"[*] Resuming from baseline: {best_ckpt.name}")
            raw_data = torch.load(str(best_ckpt), map_location=device, weights_only=False)
            sd = raw_data.get("model_state_dict", raw_data)
            model.load_state_dict(sd, strict=False)
            start_step = raw_data.get("step", 0)
            best_loss = raw_data.get("loss", float("inf"))
            log_msg(f"[*] Resuming at Step {start_step:,}")
        else:
            resume_candidates = [
                CKPT_DIR / "quillan_sovereign_gold_best.pt",
                CKPT_DIR / "quillan_direct_factual_best.pt",
                CKPT_DIR / "quillan_master_frontier_best.pt"
            ]
            for cand in resume_candidates:
                if cand.exists():
                    log_msg(f"[*] Resuming from baseline: {cand.name}")
                    sd = torch.load(str(cand), map_location=device, weights_only=False)
                    sd = sd.get("model_state_dict", sd)
                    model.load_state_dict(sd, strict=False)
                    break

    # Enable gradients on LoRAs, Underling Swarms, Bridges, Prism, LayerNorms
    trainable_count = 0
    total_count = 0
    for name, param in model.named_parameters():
        total_count += param.numel()
        if any(k in name for k in ['lora', 'swarm', 'expert_swarms', 'q1_bridge', 'q2_bridge', 'ingest_gate', 'prism', 'ln_']):
            param.requires_grad = True
            trainable_count += param.numel()
        else:
            param.requires_grad = False

    log_msg(f"[+] Trainable Parameters: {trainable_count:,} / {total_count:,} ({trainable_count/total_count*100:.1f}%)")

    dataset = SuperTrainingDataset(tokenizer, max_seq_len=256)

    total_steps = 10000
    batch_size = 4
    accum_steps = 2
    peak_lr = 2.5e-2
    min_lr = 5.0e-4
    weight_decay = 0.01
    grad_clip = 1.0

    optimizer = torch.optim.SGD(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=peak_lr,
        momentum=0.9,
        weight_decay=weight_decay
    )
    
    # 4 Epochs / Restarts across 10,000 steps (T_0 = 2500 steps per cycle)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=2500, T_mult=1, eta_min=min_lr)

    # Fast-forward scheduler to resume point
    for _ in range(start_step // accum_steps):
        scheduler.step()

    model.train()
    running_loss = 0.0
    steps_accum = 0
    t0 = time.time()

    log_msg(f"[*] Commencing 10,000-Step Quad-Epoch Super Run from Step {start_step + 1:,} (Batch: {batch_size*accum_steps}, Restarts: 4)...")

    for step in range(start_step + 1, total_steps + 1):
        x, y = dataset.sample_batch(batch_size=batch_size)
        x = x.to(device)
        y = y.to(device)

        logits, loss = model(x, labels=y)
        loss = loss / accum_steps
        loss.backward()

        running_loss += loss.item() * accum_steps
        steps_accum += 1

        if step % accum_steps == 0:
            gnorm = torch.nn.utils.clip_grad_norm_(
                filter(lambda p: p.requires_grad, model.parameters()),
                grad_clip
            )
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

        if step % 25 == 0:
            avg_loss = running_loss / steps_accum
            elapsed = time.time() - t0
            sps = step / max(0.001, elapsed)
            lr_curr = optimizer.param_groups[0]["lr"]
            epoch_num = (step // 2500) + 1

            log_msg(
                f"Epoch {epoch_num}/4 | Step {step:5d}/{total_steps} | "
                f"Loss: {avg_loss:.4f} | LR: {lr_curr:.2e} | "
                f"GNorm: {gnorm.item():.2f} | {sps:.2f} st/s"
            )

            # Checkpoint on loss improvement
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_ckpt_path = CKPT_DIR / "quillan_super_run_best.pt"
                torch.save({
                    "step": step,
                    "loss": best_loss,
                    "model_state_dict": model.state_dict(),
                    "config": cfg,
                }, str(best_ckpt_path))
                log_msg(f"--> [NEW BEST] Saved checkpoint (Loss: {best_loss:.4f}) to {best_ckpt_path.name}")

            running_loss = 0.0
            steps_accum = 0

        # Periodic checkpoint every 500 steps
        if step % 500 == 0:
            periodic_path = CKPT_DIR / f"quillan_super_run_step{step}.pt"
            torch.save({
                "step": step,
                "loss": avg_loss,
                "model_state_dict": model.state_dict(),
                "config": cfg,
            }, str(periodic_path))
            log_msg(f"[+] Saved periodic milestone checkpoint: {periodic_path.name}")

    final_ckpt_path = CKPT_DIR / "quillan_super_run_final.pt"
    torch.save({
        "step": total_steps,
        "loss": best_loss,
        "model_state_dict": model.state_dict(),
        "config": cfg,
    }, str(final_ckpt_path))

    log_msg("==================================================================")
    log_msg(f"   🏆 10,000-STEP SUPER RUN COMPLETE (Best Loss: {best_loss:.4f})")
    log_msg("==================================================================")

if __name__ == "__main__":
    train_super_run()
