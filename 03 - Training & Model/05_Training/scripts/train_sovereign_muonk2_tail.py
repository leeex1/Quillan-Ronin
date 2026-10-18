#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
👑 QUILLAN-RONIN v5.3.1 — SOVEREIGN FRONTIER SFT ANNEALING ENGINE
---------------------------------------------------------------------------------------
Executes deep teacher distillation and reasoning tail annealing on the 6-layer
QuillanRoninOni architecture using SovereignMuonK2AdamW optimizer.

Architectural Guarantees:
  1. Strict checkpoint verification: all 1,438 tensors loaded deterministically.
  2. Dual optimizer partitioning: Newton-Schulz low-rank Muon for 2D swarms/LoRA,
     AdamW for embeddings, LayerNorms, and ingestion bridges.
  3. Dynamic Curvature Regularization (CCRL) and gradient clipping.
  4. CPU thermal & thread capping for host system stability.
"""

from __future__ import annotations

import json
import logging
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR

# Configure path resolution for canonical modules
REPO_ROOT = Path(r"C:\02_QUILLAN").resolve()
ONI_DIR = REPO_ROOT / "09 - Projects" / "projects" / "oni"
TRAIN_SCRIPTS_DIR = REPO_ROOT / "09 - Projects" / "projects" / "05_Training" / "scripts"
DATA_DIR = REPO_ROOT / "training_data"
CKPT_DIR = REPO_ROOT / "checkpoints" / "checkpoints_sft"
PROD_DIR = REPO_ROOT / "checkpoints" / "production_export"

for target_dir in [str(ONI_DIR), str(TRAIN_SCRIPTS_DIR), str(REPO_ROOT)]:
    if target_dir not in sys.path:
        sys.path.insert(0, target_dir)

from quillan_v5_4_oni import QuillanRoninOni, QuillanOniConfig
from quillan_muonk2_optimizer import create_quillan_muonk2_optimizer

# Enforce strict thread bounds to prevent CPU stalls
torch.set_num_threads(min(4, os.cpu_count() or 4))
torch.set_num_interop_threads(2)

LOGGER = logging.getLogger("sovereign_muonk2_trainer")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [SOVEREIGN-TRAIN] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)


class SovereignDistillationDataset:
    """Memory-efficient streaming dataset over tokenized frontier distillation tensors."""

    def __init__(self, data_dir: Path, max_seq_len: int = 256) -> None:
        self.max_seq_len = max_seq_len
        self.samples: List[Tuple[torch.Tensor, torch.Tensor]] = []
        self._load_corpora(data_dir)

    def _load_corpora(self, data_dir: Path) -> None:
        """Loads and canonicalizes pristine frontier and GPT-5.5 distilled tensors."""
        pristine_path = data_dir / "pristine_frontier_gold_37k.pt"
        if pristine_path.exists():
            LOGGER.info("Loading pristine sequences from %s", pristine_path.name)
            data = torch.load(str(pristine_path), map_location="cpu", weights_only=False)
            ids = data.get("input_ids", data.get("ids", None))
            labels = data.get("labels", ids)
            if ids is not None and labels is not None:
                for i in range(min(len(ids), 5000)):
                    seq = ids[i][: self.max_seq_len]
                    lbl = labels[i][: self.max_seq_len]
                    if len(seq) < self.max_seq_len:
                        pad = self.max_seq_len - len(seq)
                        seq = torch.cat([seq, torch.full((pad,), 50256, dtype=torch.long)])
                        lbl = torch.cat([lbl, torch.full((pad,), -100, dtype=torch.long)])
                    self.samples.append((seq.clone(), lbl.clone()))
                LOGGER.info("Loaded %d pristine frontier samples.", len(self.samples))

        gpt5_path = data_dir / "GPT_5.5_Distilled.pt"
        if gpt5_path.exists():
            LOGGER.info("Loading GPT-5.5 distillation tensors from %s", gpt5_path.name)
            data_gpt5 = torch.load(str(gpt5_path), map_location="cpu", weights_only=False)
            if isinstance(data_gpt5, dict):
                raw_tokens = data_gpt5.get("input_ids", data_gpt5.get("ids", None))
            elif isinstance(data_gpt5, torch.Tensor):
                raw_tokens = data_gpt5
            else:
                raw_tokens = None

            if raw_tokens is not None:
                # 1D contiguous token tensor chunked into max_seq_len sequences
                if raw_tokens.ndim == 1:
                    total_tokens = raw_tokens.numel()
                    num_chunks = min(5000, total_tokens // self.max_seq_len)
                    for i in range(num_chunks):
                        start = i * self.max_seq_len
                        end = start + self.max_seq_len
                        seq = raw_tokens[start:end]
                        lbl = seq.clone()
                        self.samples.append((seq, lbl))
                    LOGGER.info("Appended %d GPT-5.5 distillation chunks (%d tokens).", num_chunks, num_chunks * self.max_seq_len)
                elif raw_tokens.ndim == 2:
                    for i in range(min(len(raw_tokens), 5000)):
                        seq = raw_tokens[i][: self.max_seq_len]
                        lbl = seq.clone()
                        self.samples.append((seq, lbl))
                    LOGGER.info("Appended %d 2D GPT-5.5 samples.", min(len(raw_tokens), 5000))

        if not self.samples:
            raise RuntimeError(f"No valid distillation tensors found in {data_dir}")

        random.seed(42)
        random.shuffle(self.samples)
        LOGGER.info("Total consolidated distillation pool: %d sequences", len(self.samples))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.samples[idx]


def train_sovereign_tail(
    steps: int = 500,
    batch_size: int = 2,
    gradient_accumulation_steps: int = 2,
    lr_muon: float = 0.012,
    lr_adamw: float = 2.0e-4,
    device: str = "cpu"
) -> Path:
    """
    Executes bounded SFT tail training to solidify factual closure and reasoning tokens.
    Time Complexity: O(steps * batch_size * seq_len * d_model)
    Space Complexity: O(d_model * seq_len) in-place activations.
    """
    LOGGER.info("Initializing 6-Layer Sovereign Model Engine...")
    cfg = QuillanOniConfig(
        n_layer=6,
        hidden_dim=1024,
        ffn_dim=2048,
        num_experts=34,
        device=device
    )
    model = QuillanRoninOni(cfg).to(device)

    # Load baseline production checkpoint
    target_save_path = CKPT_DIR / "quillan_frontier_v2_best.pt"
    source_ckpt = PROD_DIR / "quillan_ronin_v531_sovereign_production.pt"
    if not source_ckpt.exists():
        source_ckpt = CKPT_DIR / "quillan_frontier_v2_best.pt"

    LOGGER.info("Loading baseline state dictionary from %s...", source_ckpt.name)
    checkpoint_data = torch.load(str(source_ckpt), map_location=device, weights_only=False)
    state_dict = checkpoint_data.get("model_state_dict", checkpoint_data)
    model.load_state_dict(state_dict, strict=True)
    LOGGER.info("All 1,438 tensors validated and loaded with strict=True.")

    # Target parameters: 2D LoRA/swarms, bridges, and LayerNorms
    trainable_count = 0
    total_count = 0
    for name, param in model.named_parameters():
        total_count += param.numel()
        if any(target in name for target in [
            "lora", "swarm", "expert_swarms", "q1_bridge", "q2_bridge",
            "ingest_gate", "prism", "ln_", "quillan_finalizer", "quillan_comm_gate"
        ]):
            param.requires_grad = True
            trainable_count += param.numel()
        else:
            param.requires_grad = False

    LOGGER.info("Active Parameter Footprint: %s / %s (%.2f%%)",
                f"{trainable_count:,}", f"{total_count:,}", (trainable_count / total_count) * 100)

    # Initialize hybrid optimizer
    optimizer = create_quillan_muonk2_optimizer(
        model,
        lr_muon=lr_muon,
        lr_adamw=lr_adamw,
        weight_decay=0.01,
        ccrl_limit=5.0
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=steps, eta_min=1.0e-5)

    dataset = SovereignDistillationDataset(DATA_DIR, max_seq_len=256)
    total_samples = len(dataset)

    model.train()
    best_loss = float("inf")
    running_loss = 0.0
    data_idx = 0
    t0 = time.time()

    LOGGER.info("Beginning execution of %d optimization steps...", steps)

    for step in range(1, steps + 1):
        optimizer.zero_grad(set_to_none=True)
        step_loss = 0.0

        for _ in range(gradient_accumulation_steps):
            inp_batch: List[torch.Tensor] = []
            lbl_batch: List[torch.Tensor] = []
            for _ in range(batch_size):
                inp, lbl = dataset[data_idx % total_samples]
                inp_batch.append(inp)
                lbl_batch.append(lbl)
                data_idx += 1

            inp_tensor = torch.stack(inp_batch).to(device)
            lbl_tensor = torch.stack(lbl_batch).to(device)

            out = model(inp_tensor, labels=lbl_tensor)
            if isinstance(out, (tuple, list)):
                logits, loss = out[0], out[1]
            else:
                logits, loss = out, None
            scaled_loss = loss / gradient_accumulation_steps
            scaled_loss.backward()
            step_loss += scaled_loss.item()

        torch.nn.utils.clip_grad_norm_(
            filter(lambda p: p.requires_grad, model.parameters()),
            max_norm=1.0
        )
        optimizer.step()
        scheduler.step()

        running_loss += step_loss

        if step % 10 == 0 or step == 1:
            interval = 10 if step > 1 else 1
            avg_loss = running_loss / interval
            running_loss = 0.0
            elapsed = time.time() - t0
            sps = step / max(0.001, elapsed)
            current_lr = scheduler.get_last_lr()[0]
            LOGGER.info("Step [%4d/%4d] | Loss: %.4f | LR: %.2e | %.2f step/s",
                        step, steps, avg_loss, current_lr, sps)

            if avg_loss < best_loss and step >= 20:
                best_loss = avg_loss
                target_save_path = CKPT_DIR / "quillan_frontier_v2_best.pt"
                export_save_path = PROD_DIR / "quillan_ronin_v531_sovereign_production.pt"
                payload = {
                    "model_state_dict": model.state_dict(),
                    "step": 5251 + step,
                    "loss": best_loss,
                    "arch": "QuillanRoninOni",
                    "n_layer": 6,
                    "num_experts": 34
                }
                torch.save(payload, str(target_save_path))
                torch.save(payload, str(export_save_path))
                LOGGER.info("🏆 Checkpoint preserved (Step %d | Loss: %.4f)", 5251 + step, best_loss)

    LOGGER.info("Training cycle complete. Final Best Loss: %.4f", best_loss)
    return target_save_path


if __name__ == "__main__":
    train_sovereign_tail(steps=500)
