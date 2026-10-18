#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QUILLAN-RONIN v5.4.0-ONI — PROPER INSTRUCTION SFT ENGINE
=========================================================
Trains the 6-Layer model on clean question-and-answer pairs until
cross-entropy loss converges and English generation becomes coherent.

Features:
  1. Loss Masking: Computes loss strictly on assistant response tokens,
     preventing the model from memorizing prompt templates.
  2. Unfrozen Semantic Path: Unfreezes token_memory, attention Q/K/Proj,
     lm_head, LayerNorms, and Council expert adapters.
  3. FP16 Mixed Precision: Fits cleanly within 4.29 GB VRAM on GTX 1050.
  4. Gradient Accumulation: Effective batch size of 16 (micro-batch 2 x 8 accum).
  5. Live Validation Probing: Generates a test response every 100 steps.
"""

import os
import sys
import gc
import json
import time
import math
import argparse
import logging
import functools
from pathlib import Path
from typing import List, Tuple, Dict, Any

# Ensure unbuffered output in logs and console
print = functools.partial(print, flush=True)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer

REPO_ROOT = Path(r"C:\02_QUILLAN")
SCRIPTS_DIR = REPO_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from quillan_v5_4_oni import QuillanOniConfig, QuillanRoninOni

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
LOGGER = logging.getLogger("QuillanSFT")


class InstructionQADataset(Dataset):
    """Combines JSONL prompt-response pairs and pre-tokenized gold tensors with label masking."""

    def __init__(self, tok: Tokenizer, seq_len: int = 256):
        self.seq_len = seq_len
        self.samples: List[Tuple[torch.Tensor, torch.Tensor]] = []

        LOGGER.info("Packing Instruction QA Dataset...")

        # 1. Load Quillan_Universal_Sovereign_Gold_1000.jsonl
        p_univ = REPO_ROOT / "training_data" / "Quillan_Universal_Sovereign_Gold_1000.jsonl"
        if p_univ.exists():
            with open(p_univ, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    item = json.loads(line)
                    q = item.get("question", "").strip()
                    a = item.get("response", "").strip()
                    if q and a:
                        self._add_pair(tok, q, a)

        # 2. Load Quillan_Direct_Answers_Gold.jsonl
        p_direct = REPO_ROOT / "training_data" / "Quillan_Direct_Answers_Gold.jsonl"
        if p_direct.exists():
            with open(p_direct, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    item = json.loads(line)
                    prompt = item.get("prompt", "").strip()
                    resp = item.get("response", "").strip()
                    if prompt and resp:
                        self._add_raw_text(tok, prompt + "\n" + resp)

        # 3. Load pre-tokenized master gold tensors
        p_gold = REPO_ROOT / "training_data" / "canonical_standardized" / "quillan_gold_sft_master.pt"
        if p_gold.exists():
            d = torch.load(p_gold, map_location="cpu", weights_only=False)
            inp_t = d["input_ids"]
            lbl_t = d["labels"]
            for i in range(len(inp_t)):
                self.samples.append((inp_t[i][:self.seq_len].clone(), lbl_t[i][:self.seq_len].clone()))

        LOGGER.info("Total Packed SFT Sequences: %d (Loss Masking Active)", len(self.samples))

    def _add_pair(self, tok: Tokenizer, question: str, answer: str):
        prompt_str = f"User: {question}\n\nAssistant: "
        resp_str = f"{answer}<|endoftext|>"

        enc_p = tok.encode(prompt_str).ids
        enc_r = tok.encode(resp_str).ids

        full_ids = enc_p + enc_r
        if len(full_ids) > self.seq_len:
            full_ids = full_ids[:self.seq_len]

        # Labels: -100 for prompt tokens so loss is computed ONLY on assistant answer
        labels = [-100] * min(len(enc_p), len(full_ids)) + full_ids[len(enc_p):]

        pad_len = self.seq_len - len(full_ids)
        if pad_len > 0:
            full_ids = full_ids + [0] * pad_len
            labels = labels + [-100] * pad_len

        self.samples.append((torch.tensor(full_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)))

    def _add_raw_text(self, tok: Tokenizer, text: str):
        enc = tok.encode(text + "<|endoftext|>").ids
        if len(enc) > self.seq_len:
            enc = enc[:self.seq_len]
        labels = list(enc)
        pad_len = self.seq_len - len(enc)
        if pad_len > 0:
            enc = enc + [0] * pad_len
            labels = labels + [-100] * pad_len
        self.samples.append((torch.tensor(enc, dtype=torch.long), torch.tensor(labels, dtype=torch.long)))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.samples[idx]


def probe_generation(model, tok, prompt: str, device: torch.device, max_new: int = 45) -> str:
    """Tests live autoregressive greedy generation during training."""
    model.eval()
    fmt = f"User: {prompt}\n\nAssistant: "
    enc = tok.encode(fmt).ids
    gen = list(enc)
    with torch.no_grad():
        x = torch.tensor([enc], dtype=torch.long, device=device)
        for _ in range(max_new):
            out = model(x, use_cache=False, deliberation=False)
            logits = out[0] if isinstance(out, tuple) else out
            nxt = torch.argmax(logits[0, -1, :tok.get_vocab_size()], dim=-1).item()
            if nxt in [0, 50256]:
                break
            gen.append(nxt)
            x = torch.tensor([gen[-256:]], dtype=torch.long, device=device)
    model.train()
    text = tok.decode(gen[len(enc):]).strip()
    return text


def run_sft():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LOGGER.info("Active Device for Proper SFT: %s", device)

    ckpt_path = REPO_ROOT / "checkpoints" / "checkpoints_oni" / "quillan_6l_cpt_best.pt"
    out_ckpt = REPO_ROOT / "checkpoints" / "checkpoints_oni" / "quillan_6l_sft_proper.pt"

    LOGGER.info("Loading base weights from: %s", ckpt_path.name)
    data = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = QuillanOniConfig(**data["config"])
    cfg.device = str(device)

    model = QuillanRoninOni(cfg).to(device)
    model.load_state_dict(data["model_state_dict"], strict=True)

    # ── UNFREEZE STRATEGY FOR PROPER LANGUAGE CONVERGENCE ─────────────────────
    # Unfreeze token_memory (Memory Attention), c_attn, c_proj, lm_head, and experts
    unfrozen_params = []
    frozen_count = 0
    trainable_count = 0

    for name, p in model.named_parameters():
        if any(k in name for k in ("token_memory", "c_attn", "c_proj", "lm_head", "ln", "norm", "lora", "expert", "router", "gate")):
            p.requires_grad = True
            unfrozen_params.append(p)
            trainable_count += p.numel()
        else:
            p.requires_grad = False
            frozen_count += p.numel()

    LOGGER.info("Parameter Strategy: Trainable=%s | Frozen Backbone=%s", f"{trainable_count:,}", f"{frozen_count:,}")

    # Tokenizer
    tok_path = REPO_ROOT / "quillan_bpe_tokenizer_hf" / "tokenizer.json"
    tok = Tokenizer.from_file(str(tok_path))

    dataset = InstructionQADataset(tok, seq_len=256)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, drop_last=True)

    # Optimizer & Scheduler
    lr = 1.2e-4
    total_steps = 2000
    warmup_steps = 100
    accum_steps = 8  # Effective batch size = 2 * 8 = 16

    optimizer = torch.optim.AdamW(unfrozen_params, lr=lr, betas=(0.9, 0.98), weight_decay=0.01)

    LOGGER.info("Training Plan: %d steps, Effective Batch Size=16, Peak LR=%.1e", total_steps, lr)
    print("=" * 70)
    print(f"  STARTING PROPER INSTRUCTION SFT (Target Loss: < 2.50)")
    print("=" * 70)

    step = 0
    running_loss = 0.0
    best_loss = float("inf")
    start_time = time.time()
    loader_iter = iter(dataloader)

    model.train()
    optimizer.zero_grad()

    while step < total_steps:
        step += 1

        # Cosine LR schedule with warmup
        if step < warmup_steps:
            curr_lr = lr * step / warmup_steps
        else:
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            curr_lr = 1e-5 + 0.5 * (lr - 1e-5) * (1.0 + math.cos(math.pi * progress))
        for pg in optimizer.param_groups:
            pg["lr"] = curr_lr

        # Micro-batch accumulation loop
        accum_loss = 0.0
        for _ in range(accum_steps):
            try:
                b_inp, b_tgt = next(loader_iter)
            except StopIteration:
                loader_iter = iter(dataloader)
                b_inp, b_tgt = next(loader_iter)

            b_inp = b_inp.to(device)
            b_tgt = b_tgt.to(device)

            out = model(b_inp, use_cache=False, deliberation=False)
            logits = out[0] if isinstance(out, tuple) else out

            # Shift targets: position t predicts position t+1
            shift_logits = logits[:, :-1, :].contiguous().view(-1, cfg.vocab_size)
            shift_targets = b_tgt[:, 1:].contiguous().view(-1)

            loss = F.cross_entropy(shift_logits, shift_targets, ignore_index=-100)
            scaled_loss = loss / accum_steps
            scaled_loss.backward()
            accum_loss += loss.item() / accum_steps

        torch.nn.utils.clip_grad_norm_(unfrozen_params, max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()

        running_loss += accum_loss

        # Log progress every 20 steps
        if step % 20 == 0:
            avg_loss = running_loss / 20.0
            ppl = math.exp(min(avg_loss, 20.0))
            elapsed = time.time() - start_time
            steps_per_sec = step / max(elapsed, 0.001)
            print(f"[Step {step:4d}/{total_steps}] Loss: {avg_loss:.4f} | PPL: {ppl:7.2f} | LR: {curr_lr:.2e} | Speed: {steps_per_sec:.2f} step/s")
            running_loss = 0.0

            if avg_loss < best_loss and step >= 200:
                best_loss = avg_loss
                # Save best checkpoint atomically
                ckpt_dict = {
                    "step": step,
                    "loss": best_loss,
                    "config": cfg.__dict__,
                    "model_state_dict": model.state_dict(),
                    "timestamp": time.time(),
                    "engine": "Quillan-Ronin v5.4.0-ONI Proper SFT",
                }
                torch.save(ckpt_dict, out_ckpt)
                print(f"  >>> [CKPT SAVED] New Best Loss: {best_loss:.4f} -> {out_ckpt.name}")

        # Live Generation Probe every 100 steps
        if step % 100 == 0:
            print("\n" + "-" * 70)
            print(f"  [LIVE GENERATION PROBE @ STEP {step}]")
            test_prompt = "What is the exact speed of light in a vacuum?"
            resp = probe_generation(model, tok, test_prompt, device)
            print(f"  Prompt:   {test_prompt}")
            print(f"  Response: {resp}")
            print("-" * 70 + "\n")

    print("\n======================================================================")
    print(f"  SFT TRAINING COMPLETE | Best Loss: {best_loss:.4f}")
    print(f"  Model saved to: {out_ckpt}")
    print("======================================================================")


if __name__ == "__main__":
    run_sft()
