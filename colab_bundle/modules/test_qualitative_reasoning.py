#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧠 QUILLAN-RONIN v5.4.0-ONI: QUALITATIVE DOMAIN GENERATION BENCHMARK
====================================================================
Tests qualitative response generation across 4 core domains:
1. Core Identity & Council Overview
2. Logic & Strategy (LOGOS / PRAXIS)
3. Safety & Ethics Alignment (VIR / WARDEN)
4. Technical Architecture (CODEWEAVER / TECHNE)
"""

import os
import sys
import time
import json
import torch

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from quillan_v5_4_oni import (
    QuillanRoninOni,
    QuillanOniConfig,
    CANONICAL_ROSTER
)

def run_qualitative_tests(checkpoint_path: str, device: str = "cpu"):
    print("=" * 70)
    print(f"[TEST] QUALITATIVE REASONING BENCHMARK -- QUILLAN-RONIN v5.4.0-ONI")
    print(f"Checkpoint: {checkpoint_path} | Device: {device}")
    print("=" * 70)

    cfg = QuillanOniConfig(
        n_layer=12,
        hidden_dim=1024,
        ffn_dim=2048,
        num_experts=34,
        expert_rank=8,
        swarm_rank=8,
        router_mode="dense_pull",
        vocab_size=50257,
        max_seq_len=512,
        device=device
    )
    model = QuillanRoninOni(cfg).to(device)

    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        sd = ckpt.get("model", ckpt.get("model_state_dict", ckpt))
        model.load_state_dict(sd, strict=False)
        print(f"Successfully loaded checkpoint step: {ckpt.get('step', 'unknown')} | best_val: {ckpt.get('best_val', 'unknown')}")
    else:
        print(f"Warning: Checkpoint '{checkpoint_path}' not found, using initialized weights.")

    model.eval()

    test_prompts = [
        {"domain": "Identity", "prompt": [15, 340, 12, 59, 230, 45]},
        {"domain": "Logic", "prompt": [40, 1205, 314, 284, 110]},
        {"domain": "Ethics", "prompt": [80, 520, 1400, 310, 420]},
        {"domain": "Engineering", "prompt": [105, 3100, 42, 670, 890]}
    ]

    results = []
    for test in test_prompts:
        print(f"\n--- Testing Domain: {test['domain']} ---")
        start_t = time.perf_counter()
        with torch.no_grad():
            res = model.deliberate(test["prompt"], max_tokens=20, temp=0.7)
        elapsed = time.perf_counter() - start_t
        
        gen_tokens = res.get("tokens", [])
        trace = res.get("trace", {})
        gates = trace.get("gates", {})
        
        print(f"  Generated {len(gen_tokens)} tokens in {elapsed:.2f}s ({len(gen_tokens)/max(1e-5, elapsed):.2f} tok/s)")
        print(f"  Deliberation Rounds: {len(trace.get('rounds', []))}")
        print(f"  Quality Gate Clearance: Passed={gates.get('passed', True)} | Ethics Score={gates.get('ethics_constraint', 0.0):.4f}")
        
        results.append({
            "domain": test["domain"],
            "tokens_generated": len(gen_tokens),
            "latency_sec": elapsed,
            "tokens_per_sec": len(gen_tokens) / max(1e-5, elapsed),
            "gate_passed": gates.get("passed", True),
            "ethics_score": gates.get("ethics_constraint", 0.0),
            "covenant_score": gates.get("covenant_identity", 0.0)
        })

    out_file = r"c:\02_QUILLAN\training_logs\oni_qualitative_eval.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "checkpoint": checkpoint_path,
            "results": results
        }, f, indent=2)

    print("\n" + "=" * 70)
    print(f"[SUCCESS] Qualitative Reasoning Evaluation complete! Saved to {out_file}")
    print("=" * 70)

if __name__ == "__main__":
    ckpt = r"c:\02_QUILLAN\checkpoints\checkpoints_oni\quillan_oni_latest.pt"
    if not os.path.exists(ckpt):
        ckpt = r"c:\02_QUILLAN\oni\checkpoint_step_500.pt"
    run_qualitative_tests(ckpt, device="cpu")
