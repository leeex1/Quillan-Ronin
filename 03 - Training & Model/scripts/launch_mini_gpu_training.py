#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/launch_mini_gpu_training.py
Autonomous GPU training launcher and gateway hot-reload integration.
"""

from __future__ import annotations

import logging
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")
LOGGER = logging.getLogger("trainer_launcher")

REPO_ROOT = Path(r"C:\02_QUILLAN")
PYTHON_EXE = REPO_ROOT / "venv_oni_cu126" / "Scripts" / "python.exe"


def run_training_step(steps: int = 50) -> bool:
    """Executes stabilized SFT training on GTX 1050 with zero PCIe paging."""
    cmd = [
        str(PYTHON_EXE),
        "-u",
        str(REPO_ROOT / "scripts" / "run_stabilized_sft_train.py"),
        "--layers", "6",
        "--use_ma",
        "--freeze_layers", "4",
        "--steps", str(steps),
        "--device", "cuda",
    ]
    LOGGER.info("Starting %d-step GPU training on System 1 Mini-6L...", steps)
    t0 = time.perf_counter()
    res = subprocess.run(cmd, cwd=str(REPO_ROOT))
    elapsed = time.perf_counter() - t0
    LOGGER.info("Training process completed in %.1f seconds with exit code %d", elapsed, res.returncode)
    return res.returncode == 0


def reload_gateway() -> None:
    """Notifies the active gateway daemon to reload updated checkpoint weights."""
    try:
        req = urllib.request.Request("http://127.0.0.1:8000/api/reload", data=b"{}", headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            LOGGER.info("Gateway hot-reload response: %s", resp.read().decode())
    except Exception as e:
        LOGGER.warning("Gateway hot-reload trigger deferred: %s", e)


if __name__ == "__main__":
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    if run_training_step(steps):
        reload_gateway()
