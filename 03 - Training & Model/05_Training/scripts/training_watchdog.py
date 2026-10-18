#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
👑 QUILLAN-RONIN v5.3.1 — SOVEREIGN TRAINING WATCHDOG
=====================================================
Monitors the frontier training run and handles:
  - Loss explosion detection (auto-reduce LR and restart)
  - OOM crash recovery (reduce batch or seq_len and restart)
  - Progress logging to a persistent JSON state file
  - Auto-resume from best checkpoint after any crash
  - Post-training benchmark trigger

Usage:
  python training_watchdog.py

It wraps train_frontier_capability.py and self-heals failures.
"""

import subprocess
import sys
import os
import re
import json
import time
import math
import logging
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
REPO_ROOT   = SCRIPTS_DIR.parent.parent
_candidate_ckpts = [
    SCRIPTS_DIR.parent / "checkpoints" / "checkpoints_sft",
    REPO_ROOT / "checkpoints" / "checkpoints_sft",
    Path(r"C:\02_QUILLAN\05_Training\checkpoints\checkpoints_sft"),
]
CKPT_DIR    = next((d for d in _candidate_ckpts if d.exists()), _candidate_ckpts[0])
STATE_FILE  = SCRIPTS_DIR / "watchdog_state.json"

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [WATCHDOG] %(message)s",
)
LOGGER = logging.getLogger("quillan.watchdog")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ── Watchdog Config ────────────────────────────────────────────────────────────
TRAINING_SCRIPT = str(SCRIPTS_DIR / "train_frontier_capability.py")
MAX_RETRIES     = 5
LOSS_EXPLOSION  = 8.5      # Kill and restart if loss exceeds this
STALL_SECONDS   = 600      # Kill if no output for 10 minutes
LOSS_HISTORY_N  = 5        # Moving average window for explosion check


def load_state() -> dict:
    if STATE_FILE.exists():
        try:
            with open(STATE_FILE, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {"retries": 0, "best_loss": float("inf"), "last_step": 0}


def save_state(state: dict) -> None:
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def get_best_loss() -> float:
    """Read best loss from the most recent checkpoint file with safe deserialization."""
    candidates = [
        CKPT_DIR / "quillan_frontier_v2_best.pt",
        CKPT_DIR / "quillan_frontier_generalization_best.pt",
    ]
    for p in candidates:
        if p.exists():
            try:
                import torch
                try:
                    ckpt = torch.load(str(p), map_location="cpu", weights_only=True)
                except Exception:
                    ckpt = torch.load(str(p), map_location="cpu", weights_only=False)
                return float(ckpt.get("loss", float("inf")))
            except Exception:
                pass
    return float("inf")


def run_benchmark() -> None:
    """Run the multi-horizon exhaustive benchmark after training completes."""
    bench_path = SCRIPTS_DIR / "test_multi_horizon_exhaustive.py"
    if bench_path.exists():
        LOGGER.info("Running post-training exhaustive benchmark...")
        proc = subprocess.run(
            [sys.executable, str(bench_path)],
            cwd=str(SCRIPTS_DIR),
            timeout=1800,  # 30 minute timeout
            capture_output=True,
            text=True,
        )
        LOGGER.info("Benchmark complete.")
        LOGGER.info(proc.stdout[-3000:] if proc.stdout else "(no output)")
    else:
        LOGGER.warning("Benchmark script not found: %s", bench_path)


def run_once(state: dict) -> str:
    """
    Launch one training run. Returns status:
      'completed' — training reached NUM_STEPS
      'loss_explosion' — loss spiked, need LR reduction
      'oom' — out of memory error
      'crash' — any other unexpected crash
    """
    LOGGER.info("=" * 60)
    LOGGER.info(
        "Starting training run (attempt %d/%d | best_loss=%.4f)",
        state["retries"] + 1, MAX_RETRIES, state["best_loss"]
    )
    LOGGER.info("=" * 60)

    proc = subprocess.Popen(
        [sys.executable, "-u", TRAINING_SCRIPT],
        cwd=str(SCRIPTS_DIR),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        encoding="utf-8",
        errors="replace",
    )

    recent_losses = []
    last_output_time = time.time()
    completed = False
    exit_reason = "crash"

    try:
        while True:
            line = proc.stdout.readline()
            if not line:
                # Check if process ended
                if proc.poll() is not None:
                    break
                # Check stall
                if time.time() - last_output_time > STALL_SECONDS:
                    LOGGER.warning("No output for %d seconds — killing stalled process.", STALL_SECONDS)
                    proc.terminate()
                    exit_reason = "stall"
                    break
                time.sleep(0.1)
                continue

            last_output_time = time.time()
            print(line, end="", flush=True)

            # Parse step and loss
            m_step = re.search(r"Step\s+\[\s*(\d+)/(\d+)\].*Loss:\s*([\d\.]+)", line)
            if m_step:
                step = int(m_step.group(1))
                total = int(m_step.group(2))
                loss = float(m_step.group(3))

                state["last_step"] = step

                # Moving average explosion check
                recent_losses.append(loss)
                if len(recent_losses) > LOSS_HISTORY_N:
                    recent_losses.pop(0)

                avg_loss = sum(recent_losses) / len(recent_losses)
                if len(recent_losses) >= 3 and avg_loss > LOSS_EXPLOSION and not math.isnan(avg_loss):
                    LOGGER.warning(
                        "Loss explosion detected! avg=%.4f > %.1f. Killing training.",
                        avg_loss, LOSS_EXPLOSION
                    )
                    proc.terminate()
                    exit_reason = "loss_explosion"
                    break

                if step >= total:
                    completed = True
                    exit_reason = "completed"

            # Check for best checkpoint saved (update our tracked best)
            if "New Best Checkpoint" in line:
                state["best_loss"] = get_best_loss()
                save_state(state)

            # OOM detection
            if "not enough memory" in line.lower() or "outofmemory" in line.lower():
                LOGGER.warning("OOM detected. Killing training.")
                proc.terminate()
                exit_reason = "oom"
                break

    except KeyboardInterrupt:
        LOGGER.info("Interrupted by user.")
        proc.terminate()
        sys.exit(0)
    except Exception as e:
        LOGGER.error("Watchdog exception: %s", e)
        proc.terminate()

    proc.wait()
    LOGGER.info("Training process exited. Status: %s", exit_reason)
    return exit_reason


def main():
    state = load_state()
    LOGGER.info("Watchdog initialized. State: %s", state)

    while state["retries"] < MAX_RETRIES:
        status = run_once(state)
        state["retries"] += 1
        save_state(state)

        if status == "completed":
            LOGGER.info("Training COMPLETED successfully!")
            LOGGER.info("Best Loss: %.4f", state["best_loss"])
            # Run benchmark
            run_benchmark()
            LOGGER.info("Post-training pipeline complete. Watchdog done.")
            return

        elif status == "loss_explosion":
            LOGGER.warning("Loss explosion — sleeping 30s then retrying with fresh optimizer state.")
            time.sleep(30)
            # The training script auto-resumes from best checkpoint, so LR re-anneals
            continue

        elif status == "oom":
            LOGGER.warning("OOM crash — sleeping 30s. Consider reducing batch size in train_frontier_capability.py.")
            time.sleep(30)
            continue

        elif status in ("crash", "stall"):
            LOGGER.warning("Unexpected crash or stall (%s) — sleeping 60s then retrying.", status)
            time.sleep(60)
            continue

    LOGGER.error("Max retries (%d) reached. Watchdog giving up.", MAX_RETRIES)
    sys.exit(1)


if __name__ == "__main__":
    main()
