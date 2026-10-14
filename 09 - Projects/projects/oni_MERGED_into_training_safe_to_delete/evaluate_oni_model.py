#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QUILLAN-RONIN v5.4.0-ONI: COMPREHENSIVE BENCHMARK & EVALUATION SUITE
====================================================================
Evaluates:
1. AST Hardened Sandbox Security (CWE-94 Integrity)
2. 34-Expert Persona Pull Gate Deliberation Balance
3. Generation & Speculative Decoding Throughput (tok/sec)
4. Multi-round Cognitive Deliberation & Quality Gate Clearance
5. Planetary World Model & Trajectory Integrity
"""

import os
import sys
import time
import json
import torch
import torch.nn as nn
import torch.nn.functional as F

# Fix Windows console encoding for safety
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from quillan_v5_4_oni import (
    QuillanRoninOni,
    QuillanOniConfig,
    CANONICAL_ROSTER,
    HardenedSandbox,
    HighFidelityWorldModel,
    PRISM_VECTORS
)

def load_eval_model(checkpoint_path: str, device: str = "cpu") -> QuillanRoninOni:
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
        try:
            ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
        except Exception:
            ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        sd = ckpt.get("model", ckpt.get("model_state_dict", ckpt))
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"Loaded checkpoint '{checkpoint_path}' (missing={len(missing)}, unexpected={len(unexpected)})")
    else:
        print(f"Warning: Checkpoint '{checkpoint_path}' not found, using initialized weights.")
        
    model.eval()
    return model

def benchmark_generation(model: QuillanRoninOni, prompt_tokens: list, max_tokens: int = 32):
    start_time = time.perf_counter()
    with torch.no_grad():
        out_tokens = model.generate(prompt_tokens, max_tokens=max_tokens, temp=0.8)
    elapsed = time.perf_counter() - start_time
    gen_len = len(out_tokens) - len(prompt_tokens)
    tok_per_sec = gen_len / max(1e-5, elapsed)
    return {
        "generated_len": gen_len,
        "elapsed_sec": elapsed,
        "tokens_per_sec": tok_per_sec
    }

def benchmark_deliberation(model: QuillanRoninOni, prompt_tokens: list, max_tokens: int = 32):
    start_time = time.perf_counter()
    with torch.no_grad():
        res = model.deliberate(prompt_tokens, max_tokens=max_tokens, temp=0.8)
    elapsed = time.perf_counter() - start_time
    gen_len = len(res.get("tokens", []))
    tok_per_sec = gen_len / max(1e-5, elapsed)
    return {
        "generated_len": gen_len,
        "elapsed_sec": elapsed,
        "tokens_per_sec": tok_per_sec,
        "rounds": len(res.get("trace", {}).get("rounds", [])),
        "gates": res.get("trace", {}).get("gates", {}),
        "typist_refined": res.get("trace", {}).get("typist_refined", False)
    }

def evaluate_persona_pull_balance(model: QuillanRoninOni, device: str = "cpu"):
    sample_tokens = torch.randint(0, 1000, (4, 64), device=device)
    with torch.no_grad():
        flat_emb = model.wte(sample_tokens).reshape(-1, model.cfg.hidden_dim)
        pull = model.h[0].moe.pull_gate(flat_emb, tau=1.0)
        mean_pull = pull.mean(dim=0).cpu().tolist()
        
    expert_ranks = sorted(
        [{"id": CANONICAL_ROSTER[i][0], "index": i, "mean_pull": mean_pull[i]} for i in range(len(CANONICAL_ROSTER))],
        key=lambda x: x["mean_pull"],
        reverse=True
    )
    
    min_pull = min(mean_pull)
    max_pull = max(mean_pull)
    active_count = sum(1 for p in mean_pull if p > 0.001)
    
    return {
        "active_experts_count": active_count,
        "min_pull": min_pull,
        "max_pull": max_pull,
        "top_5_experts": expert_ranks[:5],
        "bottom_5_experts": expert_ranks[-5:]
    }

def test_ast_sandbox_security():
    sandbox = HardenedSandbox()
    
    safe_code = "result = sum([1, 2, 3, 4, 5])\nfinal = result * 2"
    safe_res = sandbox.run(safe_code)
    
    exploit_attempts = [
        "import os; os.system('whoami')",
        "__import__('subprocess').call(['cmd'])",
        "class A: pass\nA.__subclasses__()",
        "eval('1 + 1')"
    ]
    
    blocked_count = 0
    for exp in exploit_attempts:
        res = sandbox.run(exp)
        if res.get("status") != "success":
            blocked_count += 1
            
    return {
        "safe_execution_status": safe_res.get("status") == "success",
        "exploits_tested": len(exploit_attempts),
        "exploits_blocked": blocked_count,
        "security_integrity_pct": (blocked_count / len(exploit_attempts)) * 100.0
    }

def run_full_evaluation(checkpoint_path: str, device: str = "cpu", output_json: str = "oni_eval_report.json"):
    print("=" * 70)
    print(f"[EVAL] RUNNING FULL EVALUATION FOR QUILLAN-RONIN v5.4.0-ONI")
    print(f"Device: {device} | Checkpoint: {checkpoint_path}")
    print("=" * 70)
    
    model = load_eval_model(checkpoint_path, device=device)
    
    # 1. AST Sandbox Security Test
    print("\n[1/5] Testing AST Sandbox & Tool Hardening...")
    sec_results = test_ast_sandbox_security()
    print(f"  Security Integrity: {sec_results['security_integrity_pct']:.1f}% ({sec_results['exploits_blocked']}/{sec_results['exploits_tested']} blocked)")
    
    # 2. Persona Pull & Deliberation Balance Test
    print("\n[2/5] Evaluating 34-Expert Persona Pull Gate Deliberation...")
    pull_results = evaluate_persona_pull_balance(model, device=device)
    print(f"  Active Experts: {pull_results['active_experts_count']}/34")
    print(f"  Top Active Expert: {pull_results['top_5_experts'][0]['id']} (pull={pull_results['top_5_experts'][0]['mean_pull']:.4f})")
    
    # 3. Generation Throughput Benchmark
    print("\n[3/5] Benchmarking Generation Throughput...")
    prompt = [15, 340, 12, 59]
    gen_results = benchmark_generation(model, prompt, max_tokens=24)
    print(f"  Throughput: {gen_results['tokens_per_sec']:.2f} tok/sec (generated {gen_results['generated_len']} tokens in {gen_results['elapsed_sec']:.2f}s)")
    
    # 4. Multi-round Deliberation & Quality Gates
    print("\n[4/5] Testing Multi-round Deliberation & Quality Gates...")
    delib_results = benchmark_deliberation(model, prompt, max_tokens=24)
    print(f"  Deliberation Throughput: {delib_results['tokens_per_sec']:.2f} tok/sec | Rounds: {delib_results['rounds']}")
    print(f"  Quality Gate Clearance: {delib_results['gates']}")
    
    # 5. World Model & Trajectory Arbitration Test
    print("\n[5/5] Testing Planetary World Model & Trajectory Arbitration...")
    wm = HighFidelityWorldModel(hidden_dim=model.cfg.hidden_dim, horizon=10).to(device)
    obs = torch.randn(2, model.cfg.hidden_dim, device=device)
    belief = wm.estimate(obs)
    action = torch.randn(2, model.cfg.hidden_dim, device=device)
    traj = wm.predict_trajectory(belief, action, horizon=5)
    print(f"  Trajectory Simulated Steps: {len(traj)}")
    print(f"  Final Trajectory Confidence: {traj[-1][1]:.4f}")
    
    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "checkpoint": checkpoint_path,
        "device": device,
        "security": sec_results,
        "persona_deliberation": pull_results,
        "generation": gen_results,
        "deliberation": delib_results,
        "world_model": {
            "initial_confidence": belief.confidence,
            "final_confidence": float(traj[-1][1]),
            "trajectory_steps": len(traj)
        }
    }
    
    os.makedirs(os.path.dirname(os.path.abspath(output_json)), exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
        
    print("\n" + "=" * 70)
    print(f"[SUCCESS] EVALUATION COMPLETE -- Report saved to {output_json}")
    print("=" * 70)
    return report

if __name__ == "__main__":
    from pathlib import Path
    base_dir = Path(__file__).resolve().parent.parent.parent
    device = "cpu"
    ckpt = r"c:\02_QUILLAN\05_Training\checkpoints\checkpoints_oni\quillan_oni_latest.pt"
    if not os.path.exists(ckpt):
        local_ckpt = base_dir / "05_Training" / "checkpoints" / "checkpoints_oni" / "quillan_oni_latest.pt"
        if local_ckpt.exists():
            ckpt = str(local_ckpt)
        else:
            ckpt = str(Path(__file__).resolve().parent / "checkpoint_step_500.pt")
    
    out_dir = Path(r"c:\02_QUILLAN\05_Training\training_logs")
    if not out_dir.exists():
        out_dir = base_dir / "05_Training" / "training_logs"
    out_file = str(out_dir / "oni_eval_report.json")
    
    run_full_evaluation(ckpt, device=device, output_json=out_file)
