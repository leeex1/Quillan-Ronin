#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
👑 QUILLAN-RONIN v5.3.1 — AUTONOMOUS SOVEREIGN GOAL EVALUATION PIPELINE
Monitors the active sovereign gold alignment training run and executes the complete
16-prompt multi-horizon benchmark suite as soon as training converges.
"""

import os
import sys
import time
import json
import torch
from pathlib import Path
from typing import List

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

SCRIPTS_DIR = Path(r"C:\02_QUILLAN\scripts")
SCRATCH_DIR = Path(r"C:\Users\Admin\.gemini\antigravity-ide\brain\47d0b748-384a-4cc6-bcb6-f13c52f1d9d9\scratch")
CKPT_DIR = Path(r"C:\02_QUILLAN\checkpoints\checkpoints_sft")

sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(SCRATCH_DIR))

from test_multi_horizon_exhaustive import run_exhaustive_suite

def main():
    print("==================================================================", flush=True)
    print("   👑 QUILLAN-RONIN v5.3.1 — AUTONOMOUS GOAL BENCHMARK RUNNER", flush=True)
    print("==================================================================\n", flush=True)

    gold_ckpt = CKPT_DIR / "quillan_sovereign_gold_best.pt"
    if not gold_ckpt.exists():
        print("[-] Checkpoint not found yet, waiting...", flush=True)
        time.sleep(10)

    print(f"[+] Executing exhaustive benchmark suite on {gold_ckpt.name}...", flush=True)
    run_exhaustive_suite()
    print("\n[+] Autonomous evaluation successfully completed!", flush=True)

if __name__ == "__main__":
    main()
