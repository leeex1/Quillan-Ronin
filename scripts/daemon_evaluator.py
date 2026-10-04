#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
👑 QUILLAN-RONIN v5.3.1 — AUTONOMOUS CHECKPOINT EVALUATION DAEMON
=================================================================
Monitors checkpoint directory for new `*_best.pt` models and automatically
runs benchmark probes without interrupting the primary training loop.
Operates with single-thread CPU cap to prevent core contention.
"""

import sys
import time
import os
import json
import logging
from pathlib import Path

# CPU constraint: strictly 1 thread for background evaluation
import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

SCRIPTS_DIR = Path(r"C:\02_QUILLAN\scripts")
CKPT_DIR    = Path(r"C:\02_QUILLAN\checkpoints\checkpoints_sft")
OUT_FILE    = SCRIPTS_DIR / "daemon_eval_results.json"

sys.path.insert(0, str(SCRIPTS_DIR))

from quillan_v10_unrolled_sovereign import QuillanRoninSovereign, QuillanArchConfig
from sovereign_inference_engine import SovereignTokenizer, SovereignInferenceEngine, SamplingParams

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [EVAL-DAEMON] %(message)s")
LOGGER = logging.getLogger("quillan.daemon.evaluator")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

TARGET_CKPT = CKPT_DIR / "quillan_frontier_v2_best.pt"
POLL_INTERVAL_SEC = 60

TEST_PROMPTS = [
    ("Logic", "If all humans are mortal and Socrates is human, is Socrates mortal? Explain step by step."),
    ("Math", "A right triangle has legs of length 5 and 12. What is the length of the hypotenuse? Show work."),
    ("CS", "What is the difference between synchronous and asynchronous execution in distributed systems?"),
    ("Science", "State Einstein's mass-energy equivalence equation E=mc^2 and explain the physical meaning of each variable."),
]


def run_evaluation(ckpt_mtime: float) -> dict:
    LOGGER.info("Evaluating checkpoint: %s (mtime: %f)", TARGET_CKPT.name, ckpt_mtime)
    device = torch.device("cpu")
    tokenizer = SovereignTokenizer("gpt2")
    cfg = QuillanArchConfig()
    model = QuillanRoninSovereign(cfg).to(device)

    # Secure shape-safe load
    ckpt = torch.load(str(TARGET_CKPT), map_location=device, weights_only=False)
    sd = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
    model_sd = model.state_dict()
    filtered = {k: v for k, v in sd.items() if k in model_sd and model_sd[k].shape == v.shape}
    model.load_state_dict(filtered, strict=False)
    model.eval()

    engine = SovereignInferenceEngine(model=model, tokenizer=tokenizer, device=device)
    params = SamplingParams(
        max_new_tokens=250,
        temperature=0.35,
        top_k=40,
        top_p=0.85,
        repetition_penalty=1.20,
        frequency_penalty=0.40,
        presence_penalty=0.30,
        stop_strings=("<|im_end|>", "<|endoftext|>")
    )

    eval_records = []
    recorded_loss = float(ckpt.get("loss", 0.0))
    recorded_step = int(ckpt.get("step", 0))

    for category, question in TEST_PROMPTS:
        prompt = f"<|user|>\n{question}\n<|assistant|>\n"
        t0 = time.time()
        ans = engine.generate(prompt, params=params)
        elapsed = time.time() - t0
        eval_records.append({
            "category": category,
            "question": question,
            "response": ans,
            "latency_sec": round(elapsed, 2)
        })

    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "checkpoint": TARGET_CKPT.name,
        "checkpoint_step": recorded_step,
        "checkpoint_loss": recorded_loss,
        "eval_mtime": ckpt_mtime,
        "results": eval_records
    }

    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    LOGGER.info("Evaluation complete. Results saved to %s", OUT_FILE.name)
    return report


def main():
    LOGGER.info("Evaluation Daemon started. Monitoring: %s", TARGET_CKPT)
    last_evaluated_mtime = 0.0

    while True:
        try:
            if TARGET_CKPT.exists():
                curr_mtime = TARGET_CKPT.stat().st_mtime
                if curr_mtime > last_evaluated_mtime:
                    # Allow 5s for write lock to clear
                    time.sleep(5)
                    run_evaluation(curr_mtime)
                    last_evaluated_mtime = curr_mtime
        except Exception as e:
            LOGGER.error("Evaluator loop error: %s", e)

        time.sleep(POLL_INTERVAL_SEC)


if __name__ == "__main__":
    main()
