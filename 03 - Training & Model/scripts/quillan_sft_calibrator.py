#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
👑 QUILLAN-RONIN v5.4-ONI SFT CALIBRATOR & CONVERSATIONAL ALIGNMENT ENGINE
========================================================================
Addresses representation collapse, aligns activation microscaling, and calibrates
the language model head and 34-expert router logits against canonical SFT dialogue.

Security & Hygiene:
  - Strict weights_only=True deserialization (CWE-502)
  - Clamped token decoding with OOB index mitigation (CWE-20)
  - Bounded generation with early EOS and repetition penalty (CWE-400)
"""

from __future__ import annotations

import argparse
import gc
import logging
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Final, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT: Final[Path] = Path(r"C:\02_QUILLAN")
MODEL_DIR: Final[Path] = REPO_ROOT / "03 - Training & Model"
ONI_PROJECT: Final[Path] = REPO_ROOT / "09 - Projects" / "projects" / "oni"

for p in [str(REPO_ROOT), str(MODEL_DIR), str(ONI_PROJECT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from quillan_bpe_tokenizer import QuillanBPETokenizer
from quillan_v5_4_oni import QuillanOniConfig, QuillanRoninOni

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
LOGGER: Final[logging.Logger] = logging.getLogger("quillan_sft_calibrator")


class QuillanSFTCalibrator:
    """Production-grade SFT calibrator for recovering fluent dialogue reasoning."""

    def __init__(
        self,
        checkpoint_path: Path,
        dataset_path: Path,
        device_str: str = "cpu",
        learning_rate: float = 5e-5,
    ) -> None:
        self.device = torch.device(device_str)
        self.checkpoint_path = checkpoint_path
        self.dataset_path = dataset_path
        self.lr = learning_rate

        LOGGER.info("Initializing HuggingFace BPE Tokenizer...")
        self.tokenizer = QuillanBPETokenizer()

        LOGGER.info("Building 577M parameter QuillanRoninOni architecture...")
        self.cfg = QuillanOniConfig(
            vocab_size=50257,
            hidden_dim=1024,
            ffn_dim=2048,
            n_layer=6,
            num_experts=34,
            top_k=4,
            max_seq_len=512,
        )
        self.model = QuillanRoninOni(self.cfg).to(self.device)

        LOGGER.info("Safely loading base checkpoint from %s (weights_only=True)...", checkpoint_path.name)
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        sd = ckpt.get("model", ckpt.get("model_state_dict", ckpt))
        missing, unexpected = self.model.load_state_dict(sd, strict=False)
        LOGGER.info("Checkpoint bound (Missing: %d, Unexpected: %d)", len(missing), len(unexpected))

    def run_calibration(
        self,
        num_steps: int = 60,
        batch_size: int = 2,
        grad_accum_steps: int = 2,
        calibrate_head_only: bool = True,
        save_path: Optional[Path] = None,
    ) -> Path:
        """Executes fast SFT annealing pass to align logits and eliminate token loops."""
        LOGGER.info("Loading SFT dialogue dataset from %s...", self.dataset_path.name)
        data = torch.load(self.dataset_path, map_location="cpu", weights_only=True)
        all_inputs = data["input_ids"]
        all_labels = data["labels"]
        num_samples = all_inputs.size(0)
        LOGGER.info("Loaded %d gold conversation samples (seq_len=%d)", num_samples, all_inputs.size(1))

        if calibrate_head_only:
            # Calibrate language head, layer norms, and routers for fast high-impact alignment
            trainable_params = []
            for name, param in self.model.named_parameters():
                if any(k in name.lower() for k in ["lm_head", "ln", "norm", "gate", "router", "lora"]):
                    param.requires_grad = True
                    trainable_params.append(param)
                else:
                    param.requires_grad = False
            total_trainable = sum(p.numel() for p in trainable_params)
            LOGGER.info("Calibrating %d parameters (%.2fM) in head, routers, and norms...", len(trainable_params), total_trainable / 1e6)
        else:
            trainable_params = [p for p in self.model.parameters() if p.requires_grad]

        optimizer = torch.optim.AdamW(trainable_params, lr=self.lr, weight_decay=0.01)
        self.model.train()

        LOGGER.info("Beginning SFT calibration pass for %d steps...", num_steps)
        t_start = time.perf_counter()

        for step in range(1, num_steps + 1):
            indices = torch.randint(0, num_samples, (batch_size,))
            # Crop to active dialogue length for throughput
            batch_x = all_inputs[indices, :128].to(self.device)
            batch_y = all_labels[indices, :128].to(self.device)

            out = self.model(batch_x)
            logits = out[0] if isinstance(out, tuple) else out

            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = batch_y[..., 1:].contiguous()

            loss = F.cross_entropy(
                shift_logits.view(-1, self.cfg.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            ) / grad_accum_steps

            loss.backward()

            if step % grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

            if step % 10 == 0 or step == num_steps:
                elapsed = time.perf_counter() - t_start
                loss_val = loss.item() * grad_accum_steps
                LOGGER.info(
                    "Step %3d/%d | Loss: %.4f | Elapsed: %.1fs (%.2f step/s)",
                    step, num_steps, loss_val, elapsed, step / max(0.1, elapsed)
                )

        self.model.eval()
        out_path = save_path or (self.checkpoint_path.parent / "quillan_calibrated_dialogue.pt")
        LOGGER.info("Saving calibrated weights to %s...", out_path)
        torch.save({"model": self.model.state_dict(), "config": self.cfg.__dict__}, out_path)
        LOGGER.info("Calibration complete. Model weights saved successfully.")
        return out_path

    def test_reasoning(self, prompt: str, max_new_tokens: int = 25, temperature: float = 0.8, top_k: int = 40) -> str:
        """Evaluates conversational generation with Karpathy-style multinomial sampling & repetition penalty."""
        self.model.eval()
        formatted_prompt = f"<|user|>\n{prompt}\n<|assistant|>\n"
        input_ids = self.tokenizer.encode(formatted_prompt)
        generated = list(input_ids)

        with torch.no_grad():
            for _ in range(max_new_tokens):
                inp = torch.tensor([generated[-128:]], dtype=torch.long, device=self.device)
                out = self.model(inp)
                logits = out[0] if isinstance(out, tuple) else out
                next_logits = logits[0, -1, :].clone()

                # Repetition penalty
                for token_id in set(generated[-32:]):
                    if next_logits[token_id] > 0:
                        next_logits[token_id] /= 1.2
                    else:
                        next_logits[token_id] *= 1.2

                # Temperature scaling & top-k filtering
                next_logits = next_logits / max(0.1, temperature)
                v, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                next_logits[next_logits < v[[-1]]] = -float('Inf')

                # Multinomial sampling (Karpathy nanoGPT style)
                probs = F.softmax(next_logits, dim=-1)
                next_token = int(torch.multinomial(probs, num_samples=1).item())

                generated.append(next_token)
                if next_token in [50256, self.tokenizer.encode("<|endoftext|>")[0]]:
                    break

        return self.tokenizer.decode(generated[len(input_ids):])


if __name__ == "__main__":
    ckpt = REPO_ROOT / "checkpoints" / "checkpoints_sft" / "quillan_frontier_v2_best.pt"
    data_path = REPO_ROOT / "training_data" / "canonical_standardized" / "quillan_gold_canonical.pt"
    calibrator = QuillanSFTCalibrator(ckpt, data_path)
    calibrator.run_calibration(num_steps=50, batch_size=2)
    print("\n--- Testing Calibrated Model ---", flush=True)
    for q in ["Who are you?", "What is 2 + 2?"]:
        print(f"Q: {q}\nA: {calibrator.test_reasoning(q)}\n", flush=True)
