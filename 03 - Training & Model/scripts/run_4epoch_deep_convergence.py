#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QUILLAN-RONIN — 4-Epoch Deep Convergence Engine (100 Steps Total)
Executes 4 epochs (4 x 25 steps = 100 steps) with eval_interval=25 to drive
validation loss below 1.80 (Perplexity < 6.0) for rock-solid long-form coherence.
"""

import os
import sys
import subprocess
import urllib.request
import json
from pathlib import Path

REPO_ROOT = Path("C:/02_QUILLAN")
PYTHON_EXE = REPO_ROOT / "venv_oni_cu126" / "Scripts" / "python.exe"

def run_4epoch_deep_run(layers: int = 6):
    print("=" * 75)
    print(f"  [QUILLAN-RONIN] 4-EPOCH DEEP CONVERGENCE RUN ({layers}-LAYER MODEL)")
    print("  Schedule: 4 Epochs x 25 Steps = 100 Steps Total | Eval Interval: 25 Steps")
    print("  Target: Validation CE Loss < 1.80 | Perplexity < 6.0 (Long-Form Coherence)")
    print("=" * 75)

    freeze_layers = 4 if layers == 6 else 10
    cmd = [
        str(PYTHON_EXE), "-u", "scripts/run_stabilized_sft_train.py",
        "--layers", str(layers),
        "--use_ma",
        "--freeze_layers", str(freeze_layers),
        "--steps", "100",
        "--eval_interval", "25",
        "--device", "cuda",
    ]
    subprocess.run(cmd, cwd=str(REPO_ROOT), check=True)

    # Hot-reload into Gateway
    try:
        req = urllib.request.Request("http://127.0.0.1:8000/api/reload", method="POST", data=b"{}")
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            print(f"[PIPELINE] Hot-reload confirmed after {layers}L 4-epoch run:", data)
    except Exception as e:
        print("[PIPELINE] Gateway reload note:", e)

if __name__ == "__main__":
    target_layers = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    run_4epoch_deep_run(target_layers)
