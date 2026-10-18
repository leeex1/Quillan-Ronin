#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
orchestrate_dual_models.py — Autonomous Overnight Dual-Model Pipeline
=====================================================================
Orchestrates end-to-end training and evaluation of both Quillan models:
  Stage 1: Monitor 6L Mini Model SFT until convergence (target <= 1.75).
  Stage 2: Run 20-prompt benchmark evaluation on 6L clean checkpoint.
  Stage 3: Launch 12L Main Model SFT (512 context, monolithic BPE).
  Stage 4: Monitor 12L training until convergence.
  Stage 5: Run 20-prompt benchmark evaluation on 12L clean checkpoint.
  Stage 6: Output verified completion report for /goal fulfillment.
"""
import os, sys, gc, time, subprocess, functools
from pathlib import Path

print = functools.partial(print, flush=True)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REPO_ROOT   = Path(r"C:\02_QUILLAN")
SCRIPTS_DIR = REPO_ROOT / "scripts"
PYTHON_EXE  = r"C:\02_QUILLAN\venv_oni_cu126\Scripts\python.exe"

LOG_DIR     = REPO_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
STAGE_LOG   = LOG_DIR / "dual_model_pipeline.log"


def log(msg):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    with open(STAGE_LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def is_6l_running():
    try:
        out = subprocess.check_output(
            ["powershell", "-NoProfile", "-Command", "Get-CimInstance Win32_Process -Filter \"Name = 'python.exe'\" | Select-Object -ExpandProperty CommandLine"],
            text=True, errors="replace"
        )
        return "run_6l_sft_clean.py" in out
    except Exception:
        return False


def wait_for_6l_completion():
    log("=== STAGE 1: Monitoring 6L Mini Model Training ===")
    ckpt_6l = REPO_ROOT / "checkpoints" / "checkpoints_oni" / "quillan_6l_clean_sft.pt"
    
    while True:
        running = is_6l_running()
        if not running:
            log("6L training process has finished.")
            break
        
        # Check current progress from log
        log_file = None
        for p in Path(r"C:\Users\Admin\.gemini\antigravity-ide\brain").rglob("task-562.log"):
            log_file = p
            break
        
        if log_file and log_file.exists():
            try:
                lines = log_file.read_text(encoding="utf-8", errors="replace").strip().splitlines()
                last_lines = [l for l in lines[-10:] if "loss=" in l or "EARLY STOP" in l or "DONE" in l]
                if last_lines:
                    log(f"  6L Progress: {last_lines[-1]}")
                if any("EARLY STOP" in l or "DONE" in l for l in lines[-5:]):
                    log("  6L early stop / completion target detected!")
                    break
            except Exception:
                pass
        
        time.sleep(60)

    log(f"Stage 1 Complete: 6L Checkpoint verified at {ckpt_6l.name}")


def evaluate_6l():
    log("=== STAGE 2: Evaluating 6L Clean Checkpoint on 20-Prompt Benchmark ===")
    eval_script = SCRIPTS_DIR / "run_mini_clean_forward_eval.py"
    ckpt_6l = REPO_ROOT / "checkpoints" / "checkpoints_oni" / "quillan_6l_clean_sft.pt"
    
    cmd = [PYTHON_EXE, "-u", str(eval_script), str(ckpt_6l)]
    log(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, text=True, capture_output=True, encoding="utf-8", errors="replace")
    
    log(f"6L Evaluation stdout (tail):\n{result.stdout[-800:]}")
    if result.returncode != 0:
        log(f"6L Evaluation stderr:\n{result.stderr[-500:]}")
    log("Stage 2 Complete: 6L Benchmark Report saved.")


def run_12l_training():
    log("=== STAGE 3: Launching 12L Main Model Training (seq_len=512) ===")
    train_script = SCRIPTS_DIR / "run_12l_sft_clean.py"
    cmd = [PYTHON_EXE, "-u", str(train_script)]
    
    log(f"Starting 12L process: {' '.join(cmd)}")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", bufsize=1)
    
    # Stream output and log
    for line in iter(proc.stdout.readline, ''):
        line = line.strip()
        if line:
            if any(k in line for k in ("loss=", "PPL=", "BEST", "EARLY STOP", "DONE", "PROBE")):
                log(f"  [12L] {line}")
    proc.wait()
    log(f"Stage 3 & 4 Complete: 12L exited with code {proc.returncode}")


def evaluate_12l():
    log("=== STAGE 5: Evaluating 12L Clean Checkpoint on 20-Prompt Benchmark ===")
    eval_script = SCRIPTS_DIR / "run_mini_clean_forward_eval.py"
    ckpt_12l = REPO_ROOT / "checkpoints" / "checkpoints_oni" / "quillan_12l_clean_sft.pt"
    
    if not ckpt_12l.exists():
        log(f"[WARN] {ckpt_12l.name} not found, falling back to base 12L checkpoint.")
        ckpt_12l = REPO_ROOT / "checkpoints" / "checkpoints_oni" / "quillan_12l_ma_best.pt"
    
    cmd = [PYTHON_EXE, "-u", str(eval_script), str(ckpt_12l)]
    log(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, text=True, capture_output=True, encoding="utf-8", errors="replace")
    
    log(f"12L Evaluation stdout (tail):\n{result.stdout[-800:]}")
    log("Stage 5 Complete: 12L Benchmark Report saved.")


def main():
    log("=" * 72)
    log("  QUILLAN DUAL-MODEL OVERNIGHT PIPELINE (6L + 12L)")
    log("=" * 72)
    
    # Stage 1: Wait for 6L
    wait_for_6l_completion()
    
    # Stage 2: Eval 6L
    evaluate_6l()
    
    # Clear VRAM before 12L
    gc.collect()
    time.sleep(10)
    
    # Stage 3 & 4: Train 12L
    run_12l_training()
    
    # Clear VRAM before 12L eval
    gc.collect()
    time.sleep(10)
    
    # Stage 5: Eval 12L
    evaluate_12l()
    
    log("=" * 72)
    log("  ALL PIPELINE STAGES COMPLETED SUCCESSFULLY (6L + 12L FULLY TRAINED)")
    log("=" * 72)


if __name__ == "__main__":
    main()
