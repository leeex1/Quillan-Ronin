#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
👑 QUILLAN-RONIN v5.3.1 / v5.4 — STREAMING INTERACTIVE SOVEREIGN TERMINAL
==================================================================
Run this script to chat directly with Quillan-Ronin v5.3.1 (Canonical) / v5.4 (Frontier).
Automatically selects the best available checkpoint.
"""

import sys
import time
from pathlib import Path
import torch

SCRIPTS_DIR = Path(r"C:\02_QUILLAN\scripts")
CHECKPOINT_DIR = Path(r"C:\02_QUILLAN\checkpoints")

sys.path.insert(0, str(SCRIPTS_DIR))

from quillan_v10_unrolled_sovereign import QuillanConfig, QuillanSovereignUnifiedModel
from sovereign_inference_engine import SovereignTokenizer, SovereignInferenceEngine, SamplingParams

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def select_best_checkpoint() -> Path:
    """Auto-select best available checkpoint in priority order."""
    candidates = [
        CHECKPOINT_DIR / "quillan_frontier_v2_best.pt",
        CHECKPOINT_DIR / "quillan_ronin_v531_sovereign_production.pt",
        CHECKPOINT_DIR / "checkpoints_sft" / "quillan_super_run_best.pt",
        CHECKPOINT_DIR / "checkpoints_sft" / "quillan_sovereign_gold_best.pt",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError("No Quillan checkpoint found. Run training first.")


def main():
    ckpt_path = select_best_checkpoint()

    print("==================================================================", flush=True)
    print("   👑 QUILLAN-RONIN v5.3.1 / v5.4 — INTERACTIVE SOVEREIGN TERMINAL", flush=True)
    print(f"   Active Checkpoint: {ckpt_path.name}", flush=True)
    print("==================================================================\n", flush=True)

    device = torch.device("cpu")
    tokenizer = SovereignTokenizer("gpt2")
    cfg = QuillanConfig(
        vocab_size=50257,
        hidden_dim=1024,
        num_layers=16,
        num_heads=32,
        num_experts=34,
        num_experts_active=4,
        max_seq_len=16384,
    )

    print("[*] Loading Quillan-Ronin weights...", flush=True)
    model = QuillanSovereignUnifiedModel(cfg).to(device)
    engine = SovereignInferenceEngine.load_from_checkpoint(
        model_factory=lambda: model,
        checkpoint_path=ckpt_path,
        device=device,
    )

    print("[✓] Model ready! Type 'exit' or 'quit' to end session.\n", flush=True)

    history = ""
    params = SamplingParams(
        max_new_tokens=512,
        temperature=0.65,
        top_p=0.85,
        repetition_penalty=1.20,
        frequency_penalty=0.30,
        presence_penalty=0.30,
    )

    while True:
        try:
            user_input = input("You > ").strip()
            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit"):
                print("Exiting session.")
                break

            prompt = f"<|user|>\n{user_input}\n<|assistant|>\n"
            print("Quillan > ", end="", flush=True)

            def _stream_cb(token: str):
                print(token, end="", flush=True)

            engine.generate(prompt=prompt, params=params, stream_callback=_stream_cb)
            print("\n")

        except KeyboardInterrupt:
            print("\nSession interrupted.")
            break


if __name__ == "__main__":
    main()
