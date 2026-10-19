#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🏆 EVALUATE FLUENCY GATE FOR TEACHER SFT TAIL
----------------------------------------------
Evaluates 3 standardized reasoning prompts with UTF-8 encoding.
"""

import sys
import os
import json
import torch
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REPO_ROOT = Path(r"C:\02_QUILLAN")
sys.path.insert(0, str(REPO_ROOT / "03 - Training & Model" / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from train_teacher_tail_500 import QuillanArchConfig, QuillanRoninSovereign, SovereignTokenizer, LOGS_DIR

def run_evaluation():
    print("=" * 72)
    print("  🏆 RUNNING FLUENCY GATE EVALUATION (TEACHER SFT TAIL)")
    print("=" * 72)

    tok = SovereignTokenizer("gpt2")
    cfg = QuillanArchConfig(
        n_layer=6,
        hidden_dim=1024,
        max_seq_len=256,
        num_experts=34,
        router_mode="dense_pull",
        use_speculative=False,
    )
    model = QuillanRoninSovereign(cfg)

    ckpt_path = REPO_ROOT / "checkpoints" / "checkpoints_sft" / "quillan_teacher_tail_latest.pt"
    if not ckpt_path.exists():
        ckpt_path = REPO_ROOT / "checkpoints" / "checkpoints_sft" / "quillan_teacher_tail_best.pt"
    
    print(f"Loading checkpoint: {ckpt_path.name}")
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()

    test_prompts = [
        ("Syllogism", "<|user|>\nIf all humans are mortal and Socrates is human, is Socrates mortal? Explain step by step.\n<|assistant|>\n"),
        ("Asynchronous", "<|user|>\nWhat is the difference between synchronous and asynchronous execution?\n<|assistant|>\n"),
        ("Geometry", "<|user|>\nA triangle has side lengths 5, 12, and 13. Is it a right triangle? Show your work.\n<|assistant|>\n"),
    ]

    results = {}
    pass_count = 0

    for name, prompt in test_prompts:
        tokens = tok.encode(prompt)
        gen = model.generate(tokens, max_tokens=60, temp=0.2, repetition_penalty=1.1)
        resp = tok.decode(gen[len(tokens):]).strip()
        words = resp.split()
        unique_ratio = len(set(words)) / max(len(words), 1)
        passed = len(resp) >= 30 and unique_ratio > 0.35
        if passed:
            pass_count += 1
        results[name] = {
            "response": resp,
            "tokens": len(gen) - len(tokens),
            "unique_ratio": round(unique_ratio, 3),
            "passed": passed,
        }
        print(f"\n[Prompt: {name}] -> {'PASSED' if passed else 'FAILED'} (Tokens: {len(gen)-len(tokens)}, Ratio: {unique_ratio:.2f})")
        print(f"Response:\n{resp}\n")

    overall_passed = pass_count >= 2
    report = {
        "overall_passed": overall_passed,
        "pass_count": pass_count,
        "checkpoint": ckpt_path.name,
        "step": ckpt.get("step", 5751),
        "details": results,
    }

    report_path = LOGS_DIR / "teacher_tail_fluency_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("=" * 72)
    print(f"  Fluency Gate Evaluation Result: {'PASSED [✓]' if overall_passed else 'FAILED [X]'} ({pass_count}/3 passed)")
    print(f"  Report written to: {report_path}")
    print("=" * 72)
    return report

if __name__ == "__main__":
    run_evaluation()
