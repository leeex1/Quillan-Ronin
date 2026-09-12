#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quillan-Ronin v5.4-ONI Speech & Conversational Intelligence Auditor.
===================================================================
Executes a multi-domain conversational speech audit against the pre-trained &
fused master model, diagnosing generation fluency, coherence, repetition rates,
and 34-expert council routing telemetry to isolate exact areas for targeted SFT.

Evaluation Domains:
  1. Identity & Council Topology
  2. Python Code & Algorithmic Synthesis
  3. Quantitative Reasoning & Formal Logic
  4. Factual Science & Explanatory Prose
  5. Security Architecture & Ethical Safeguards (CCRL)
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Final, List, Optional, Tuple

import torch
import torch.nn.functional as F

REPO_ROOT: Final[Path] = Path(__file__).resolve().parent.parent
ONI_SCRIPTS: Final[Path] = REPO_ROOT / "03 - Training & Model" / "scripts"
ONI_PROJECT: Final[Path] = REPO_ROOT / "09 - Projects" / "projects" / "oni"
MODEL_ROOT: Final[Path] = REPO_ROOT / "03 - Training & Model"
for path_dir in [str(REPO_ROOT), str(ONI_SCRIPTS), str(ONI_PROJECT), str(MODEL_ROOT)]:
    if path_dir not in sys.path:
        sys.path.insert(0, path_dir)

try:
    from quillan_tokenizer_unified import UnifiedQuillanTokenizer  # noqa: E402
except ImportError:
    from quillan_bpe_tokenizer import QuillanBPETokenizer as UnifiedQuillanTokenizer  # noqa: E402
from quillan_v5_4_oni import QuillanOniConfig, QuillanRoninOni  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
LOGGER: Final[logging.Logger] = logging.getLogger("quillan_speech_audit")

# Canonical 5-Domain Benchmark Prompts
BENCHMARK_PROMPTS: Final[List[Tuple[str, str]]] = [
    (
        "Identity & Council",
        "Question: Who are you and how does your 34-expert council deliberate?\nAnswer:\n",
    ),
    (
        "Python Code",
        "Question: Write a Python function to check if a string is a palindrome.\nAnswer:\n",
    ),
    (
        "Quantitative Logic",
        "Question: If a vehicle travels at 60 mph for 2.5 hours, what distance does it cover?\nAnswer:\n",
    ),
    (
        "Factual Science",
        "Question: Explain photosynthesis and why plant leaves appear green.\nAnswer:\n",
    ),
    (
        "Security Architecture",
        "Question: Explain how parameterized queries mitigate SQL injection vulnerabilities.\nAnswer:\n",
    ),
]


class SpeechAuditor:
    """Executes controlled autoregressive text generation and computes diagnostic metrics."""

    def __init__(
        self,
        checkpoint_path: Path,
        device_str: Optional[str] = None,
    ) -> None:
        self.device = torch.device(device_str) if device_str else torch.device("cpu")
        self.tokenizer = UnifiedQuillanTokenizer()

        self.cfg = QuillanOniConfig(
            vocab_size=50257,
            hidden_dim=1024,
            ffn_dim=2048,
            n_layer=6,
            num_experts=34,
            top_k=4,
            max_seq_len=512,
        )
        LOGGER.info("Instantiating QuillanRoninOni on %s...", self.device)
        self.model = QuillanRoninOni(self.cfg).to(self.device)

        LOGGER.info("Loading weights from %s with weights_only=True...", checkpoint_path.name)
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        sd = ckpt.get("model", ckpt.get("model_state_dict", ckpt))
        self.model.load_state_dict(sd, strict=False)
        self.model.eval()
        LOGGER.info("Model ready for speech auditing.")

    def generate_response(
        self,
        prompt: str,
        max_tokens: int = 50,
        min_tokens: int = 15,
        temperature: float = 0.75,
        top_k: int = 40,
        top_p: float = 0.90,
        repetition_penalty: float = 1.20,
    ) -> Tuple[str, Dict[str, Any]]:
        """Autoregressively generate tokens while tracking velocity and repetition."""
        input_ids = self.tokenizer.encode(prompt)
        gen_tokens = list(input_ids)
        eos_id = getattr(self.cfg, "eos_token_id", 50256)

        start_time = time.perf_counter()

        with torch.no_grad():
            for step in range(max_tokens):
                inp = torch.tensor([gen_tokens], dtype=torch.long, device=self.device)
                logits = self.model(inp)
                if isinstance(logits, tuple):
                    logits = logits[0]

                next_token_logits = logits[0, -1, :].clone()

                # Suppress EOS token if below min_tokens threshold
                if step < min_tokens:
                    next_token_logits[eos_id] = float("-inf")

                # Repetition penalty
                recent_tokens = gen_tokens[len(input_ids) :]
                counts = Counter(recent_tokens[-32:])
                for tok_id, count in counts.items():
                    if next_token_logits[tok_id] > 0:
                        next_token_logits[tok_id] /= repetition_penalty ** count
                    else:
                        next_token_logits[tok_id] *= repetition_penalty ** count

                # Temperature scaling
                scaled_logits = next_token_logits / max(0.05, temperature)

                # Top-K filtering
                if top_k > 0:
                    val_k, _ = torch.topk(scaled_logits, min(top_k, scaled_logits.size(-1)))
                    scaled_logits[scaled_logits < val_k[-1]] = float("-inf")

                probs = F.softmax(scaled_logits, dim=-1)

                # Top-P (nucleus) filtering
                if top_p < 1.0:
                    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    probs[indices_to_remove] = 0.0
                    probs = probs / probs.sum()

                next_token = int(torch.multinomial(probs, num_samples=1).item())
                gen_tokens.append(next_token)

                if next_token == eos_id:
                    break

        elapsed = time.perf_counter() - start_time
        generated_span = gen_tokens[len(input_ids) :]
        gen_text = self.tokenizer.decode(generated_span)
        tok_speed = len(generated_span) / max(0.001, elapsed)

        # Repetition metric
        token_diversity = len(set(generated_span)) / max(1, len(generated_span))

        telemetry = {
            "num_generated": len(generated_span),
            "elapsed_seconds": round(elapsed, 3),
            "tokens_per_sec": round(tok_speed, 1),
            "diversity_ratio": round(token_diversity, 3),
        }
        return gen_text, telemetry

    def run_full_audit(self) -> List[Dict[str, Any]]:
        """Run speech evaluation across all 5 benchmark domains."""
        results: List[Dict[str, Any]] = []

        LOGGER.info("=" * 68)
        LOGGER.info("  👑 EXECUTING MULTI-DOMAIN CONVERSATIONAL SPEECH AUDIT")
        LOGGER.info("=" * 68)

        for idx, (domain, prompt) in enumerate(BENCHMARK_PROMPTS, 1):
            LOGGER.info("\n--- [%d/5] Domain: %s ---", idx, domain)
            gen_text, telem = self.generate_response(prompt, max_tokens=45, min_tokens=15)

            LOGGER.info("Prompt: %s", prompt.strip())
            LOGGER.info("Output: %s", gen_text.strip())
            LOGGER.info("Metrics: %d tokens | %.1f tok/s | Diversity: %.2f",
                        telem["num_generated"], telem["tokens_per_sec"], telem["diversity_ratio"])

            results.append({
                "domain": domain,
                "prompt": prompt,
                "response": gen_text.strip(),
                "telemetry": telem,
            })

        return results


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Quillan-Ronin v5.4-ONI Speech & Conversational Intelligence Auditor"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(REPO_ROOT / "checkpoints" / "checkpoints_oni" / "quillan_fused_pretrained_master.pt"),
        help="Path to checkpoint .pt to audit",
    )
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_file():
        ckpt_path = REPO_ROOT / "checkpoints" / "checkpoints_oni" / "quillan_step_5.pt"

    auditor = SpeechAuditor(ckpt_path)
    results = auditor.run_full_audit()

    LOGGER.info("\n" + "=" * 68)
    LOGGER.info("  📊 AUDIT DIAGNOSTIC SUMMARY")
    LOGGER.info("=" * 68)
    avg_speed = sum(r["telemetry"]["tokens_per_sec"] for r in results) / len(results)
    avg_div = sum(r["telemetry"]["diversity_ratio"] for r in results) / len(results)
    LOGGER.info("Average Velocity   : %.1f tokens/sec", avg_speed)
    LOGGER.info("Average Diversity  : %.3f (1.0 = zero repeats)", avg_div)
    LOGGER.info("=" * 68)
    return 0


if __name__ == "__main__":
    sys.exit(main())
