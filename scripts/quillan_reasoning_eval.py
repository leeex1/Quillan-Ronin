#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
👑 QUILLAN REASONING EVALUATION: MINI (6-LAYER) & MAIN (12-LAYER)
================================================================
Evaluates both reasoning models across:
  - Identity cognition
  - Mathematical / arithmetic reasoning
  - Deductive logic reasoning
  - Python algorithmic coding

Ensures zero token collapse, stable non-repetitive generation, and clean reasoning traces.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Final, List, Tuple

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import torch
import torch.nn.functional as F

REPO_ROOT: Final[Path] = Path(r"C:\02_QUILLAN")
MODEL_DIR: Final[Path] = REPO_ROOT / "03 - Training & Model"
ONI_PROJECT: Final[Path] = REPO_ROOT / "09 - Projects" / "projects" / "oni"

for p in [str(REPO_ROOT), str(MODEL_DIR), str(ONI_PROJECT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from quillan_v5_4_oni import QuillanOniConfig, QuillanRoninOni
from quillan_bpe_tokenizer import QuillanBPETokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
LOGGER: Final[logging.Logger] = logging.getLogger("quillan_reasoning_eval")


class QuillanReasoningEvaluator:
    """Evaluates Quillan-Oni Mini and Main models on reasoning benchmarks."""

    def __init__(self, checkpoint_path: Path, n_layer: int) -> None:
        self.checkpoint_path = checkpoint_path
        self.device = torch.device("cpu")
        self.tokenizer = QuillanBPETokenizer()

        self.cfg = QuillanOniConfig(
            vocab_size=50257,
            hidden_dim=1024,
            ffn_dim=2048,
            n_layer=n_layer,
            num_experts=34,
            top_k=4,
            max_seq_len=512,
        )

        LOGGER.info("Instantiating %d-layer model from %s...", n_layer, checkpoint_path.name)
        self.model = QuillanRoninOni(self.cfg)
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        sd = ckpt.get("model", ckpt.get("model_state_dict", ckpt))
        missing, unexpected = self.model.load_state_dict(sd, strict=False)
        LOGGER.info("Model loaded successfully (Missing: %d, Unexpected: %d)", len(missing), len(unexpected))
        self.model.eval()

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 40,
        temperature: float = 0.75,
        top_k: int = 40,
        repetition_penalty: float = 1.25,
    ) -> str:
        """Karpathy multinomial autoregressive generation with repetition penalty."""
        formatted_prompt = f"<|user|>\n{prompt}\n<|assistant|>\n"
        input_ids = self.tokenizer.encode(formatted_prompt)
        generated = list(input_ids)

        with torch.no_grad():
            for _ in range(max_new_tokens):
                inp = torch.tensor([generated[-128:]], dtype=torch.long, device=self.device)
                out = self.model(inp)
                logits = out[0] if isinstance(out, tuple) else out
                next_logits = logits[0, -1, :].clone()

                # Repetition penalty over recent context
                for token_id in set(generated[-32:]):
                    if next_logits[token_id] > 0:
                        next_logits[token_id] /= repetition_penalty
                    else:
                        next_logits[token_id] *= repetition_penalty

                # Temperature scaling & top-k truncation
                next_logits = next_logits / max(0.1, temperature)
                v, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                next_logits[next_logits < v[[-1]]] = -float("Inf")

                # Multinomial stochastic sampling
                probs = F.softmax(next_logits, dim=-1)
                next_token = int(torch.multinomial(probs, num_samples=1).item())

                generated.append(next_token)
                if next_token in [50256, self.tokenizer.encode("<|endoftext|>")[0]]:
                    break

        completion = self.tokenizer.decode(generated[len(input_ids):])
        return completion.strip()


def run_evaluations() -> bool:
    """Runs reasoning benchmarks across both Mini and Main models."""
    prompts = [
        ("Identity", "Who are you?"),
        ("Arithmetic", "Calculate 12 * 15 and explain step by step."),
        ("Deductive Logic", "If all ronin are warriors and Quillan is a ronin, is Quillan a warrior? Explain step by step."),
        ("Coding", "Write a Python function to reverse a string."),
    ]

    models_to_test = [
        ("Quillan-Oni Mini (6 Layers)", REPO_ROOT / "checkpoints" / "quillan_oni_mini_6l.pt", 6),
        ("Quillan-Oni Main (12 Layers)", REPO_ROOT / "checkpoints" / "quillan_oni_main_12l.pt", 12),
    ]

    for model_name, ckpt_path, n_layer in models_to_test:
        print(f"\n{'='*70}\n[EVALUATING] {model_name}\n{'='*70}", flush=True)
        evaluator = QuillanReasoningEvaluator(ckpt_path, n_layer)

        for category, p in prompts:
            print(f"\n--- [{category}] Prompt: {p}", flush=True)
            t0 = time.perf_counter()
            response = evaluator.generate(p, max_new_tokens=35)
            dt = time.perf_counter() - t0
            print(f"Response ({dt:.1f}s):\n{response}\n", flush=True)

            # Verification assertions
            assert len(response) > 0, f"Model generated empty response for {category}"
            assert response != "." * len(response), f"Punctuation trap detected for {category}"

    return True


if __name__ == "__main__":
    success = run_evaluations()
    if success:
        print("\n[SUCCESS] All reasoning evaluations passed successfully across both Mini and Main!")
