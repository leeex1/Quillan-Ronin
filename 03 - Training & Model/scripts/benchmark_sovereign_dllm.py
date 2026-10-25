#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
benchmark_sovereign_dllm.py — CPU Benchmark: Inception dLLM vs Causal Autoregressive
=====================================================================================
Loads the Quillan 6L Sovereign model and benchmarks:
  1. Standard Autoregressive Token Generation (Causal serial passes)
  2. Inception Labs SEDD Discrete Diffusion Generation (16-step parallel unmasking)

Measures:
  - Throughput (tokens/second)
  - Wall-clock generation latency
  - Step counts
"""

import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from quillan_v5_4_oni import QuillanOniConfig, QuillanRoninOni
from quillan_live_training_upgrades import vectorize_model_in_place
from quillan_discrete_diffusion import DiscreteDiffusionSampler


def main():
    print("=" * 76)
    print("QUILLAN-RONIN: INCEPTION LABS dLLM (DIFFUSION) VS AUTOREGRESSIVE BENCHMARK")
    print("=" * 76, flush=True)

    device = torch.device("cpu")
    torch.set_num_threads(4)

    # 1. Initialize 6L Configuration matching active model
    cfg = QuillanOniConfig(
        vocab_size=50262,
        eos_token_id=0,
        max_seq_len=512,
        hidden_dim=1152,
        n_layer=6,
        n_head=18,
        head_dim=64,
        ffn_dim=3072,
        num_experts=34,
        router_mode="dense_pull",
        top_k=4,
        device="cpu",
    )

    print("\n[1/3] Instantiating 6L Sovereign Model on CPU...", flush=True)
    t0 = time.time()
    model = QuillanRoninOni(cfg).to(device)
    model.eval()
    vectorize_model_in_place(model)
    print(f"  Model instantiated in {time.time() - t0:.2f}s.")

    # 2. Test Setup: 32 and 64 Target Tokens
    prompt_ids = torch.tensor([[101, 2054, 2003]], dtype=torch.long, device=device)  # 3 prompt tokens
    test_token_lengths = [32, 64]

    def model_forward(x: torch.Tensor) -> torch.Tensor:
        out = model(x)
        if isinstance(out, tuple):
            return out[0]
        return out

    sampler = DiscreteDiffusionSampler(mask_token_id=50261, schedule="cosine")

    print("\n[2/3] Benchmarking Inference Paradigms on Host CPU...", flush=True)

    results = []

    for target_len in test_token_lengths:
        print(f"\n--- Benchmark Target: {target_len} Tokens ---")

        # Test A: Autoregressive (1 pass per token = target_len passes)
        print(f"  [A] Autoregressive Generation ({target_len} sequential forward passes)...", flush=True)
        t_ar_start = time.time()
        curr_seq = prompt_ids.clone()
        with torch.no_grad():
            for _ in range(target_len):
                logits = model_forward(curr_seq)
                next_tok = torch.argmax(logits[:, -1:, :], dim=-1)
                curr_seq = torch.cat([curr_seq, next_tok], dim=1)
        t_ar = time.time() - t_ar_start
        ar_tps = target_len / max(1e-4, t_ar)
        print(f"      Latency: {t_ar:.2f}s | Speed: {ar_tps:.2f} tok/s")

        # Test B: Inception dLLM Diffusion (16 parallel passes)
        num_diffusion_steps = 16
        print(f"  [B] Inception SEDD Diffusion ({num_diffusion_steps} parallel refinement passes)...", flush=True)
        t_diff_start = time.time()
        with torch.no_grad():
            diff_tokens, meta = sampler.sample(
                model_fn=model_forward,
                prefix_ids=prompt_ids,
                target_len=target_len,
                num_steps=num_diffusion_steps,
                temperature=0.0,
                device=device,
            )
        t_diff = time.time() - t_diff_start
        diff_tps = target_len / max(1e-4, t_diff)
        print(f"      Latency: {t_diff:.2f}s | Speed: {diff_tps:.2f} tok/s ({meta['steps_executed']} steps)")

        speedup = t_ar / max(1e-4, t_diff)
        results.append({
            "target_len": target_len,
            "ar_time": t_ar,
            "ar_tps": ar_tps,
            "diff_time": t_diff,
            "diff_tps": diff_tps,
            "speedup": speedup,
        })
        print(f"      >>> SPEEDUP: {speedup:.2f}x FASTER via Discrete Diffusion")

    print("\n[3/3] Final Benchmark Summary")
    print("=" * 76)
    print(f"{'Length':<10} | {'Autoregressive':<18} | {'SEDD Diffusion':<18} | {'Speedup':<10}")
    print("-" * 76)
    for r in results:
        print(
            f"{r['target_len']:<10} | {r['ar_tps']:>6.2f} tok/s ({r['ar_time']:>5.2f}s) | "
            f"{r['diff_tps']:>6.2f} tok/s ({r['diff_time']:>5.2f}s) | {r['speedup']:>6.2f}x"
        )
    print("=" * 76)


if __name__ == "__main__":
    main()
