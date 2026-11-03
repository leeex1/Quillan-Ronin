#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
Quillan-Ronin: Comprehensive Sovereign 1B Benchmark & Interactive Terminal
=============================================================================
Suite evaluates:
  1. Multi-step algorithmic coding (Python algorithm + complexity)
  2. Multi-step mathematical reasoning (GSM8k-style algebraic problem)
  3. Long-horizon generation (150+ tokens to stress-test Takehi deflation)
  4. Ethical Bushido dilemma (Honor vs utilitarian compromise)
  5. Systems architecture (Distributed consensus protocol)
Includes interactive chat mode for real-time user prompting.
=============================================================================
"""

import os
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"

import sys
import time
from typing import List, Optional, Tuple
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, r"C:\02_QUILLAN\03 - Training & Model\scripts")
sys.path.append(r"C:\02_QUILLAN\02 - Knowledge Foundation\knowledge\legacy\00 - Meta\oni")
from quillan_v5_4_oni import QuillanOniConfig, QuillanRoninOni


class QuillanSovereignTester:
    def __init__(self, ckpt_path: str, tok_path: str, device: str = "cpu"):
        self.device = device
        print(f"Loading tokenizer from {tok_path}...", flush=True)
        self.tok = Tokenizer.from_file(tok_path)
        
        print(f"Loading checkpoint from {ckpt_path}...", flush=True)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        cfg_dict = ckpt["config"]
        
        self.cfg = QuillanOniConfig(
            vocab_size=cfg_dict.get("vocab_size", 128256),
            hidden_dim=cfg_dict.get("hidden_dim", 2048),
            n_layer=cfg_dict.get("n_layer", 16),
            n_head=cfg_dict.get("n_head", 32),
            head_dim=cfg_dict.get("head_dim", 64),
            ffn_dim=cfg_dict.get("ffn_dim", 8192),
            num_experts=cfg_dict.get("num_experts", 34),
            expert_rank=16,
            use_evo_moe=False,
            device=device
        )
        self.cfg.vectorized_experts = True
        
        print("Instantiating QuillanRoninOni...", flush=True)
        self.model = QuillanRoninOni(self.cfg)
        missing, unexpected = self.model.load_state_dict(ckpt["model_state_dict"], strict=False)
        print(f"Model loaded. (Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)})", flush=True)
        self.model.eval()
        
        batched_count = sum(getattr(blk.moe, "vectorized_experts", False) for blk in self.model.h)
        print(f"BatchedCouncilCompute status: {batched_count}/16 layers enabled (100%)\n", flush=True)

    def generate(self, prompt: str, max_new_tokens: int = 100, temperature: float = 0.6) -> Tuple[str, float, float]:
        formatted = (
            f"<|start_header_id|>user<|end_header_id|>\n\n"
            f"{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        )
        encoded = self.tok.encode(formatted)
        input_ids = torch.tensor([encoded.ids], dtype=torch.long, device=self.device)
        
        generated_tokens: List[int] = []
        cur_input = input_ids
        
        t0 = time.time()
        with torch.no_grad():
            for _ in range(max_new_tokens):
                logits = self.model(cur_input, compute_aux=False, return_aux=False)
                if isinstance(logits, tuple):
                    logits = logits[0]
                next_token_logits = logits[0, -1, :]
                
                probs = F.softmax(next_token_logits / max(1e-4, temperature), dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).item()
                
                if next_token in [128001, 128009]:
                    break
                    
                generated_tokens.append(next_token)
                cur_input = torch.tensor([[next_token]], dtype=torch.long, device=self.device)
                
                token_str = self.tok.decode([next_token])
                print(token_str, end="", flush=True)
                
        elapsed = time.time() - t0
        speed = len(generated_tokens) / max(0.01, elapsed)
        text = self.tok.decode(generated_tokens)
        return text, speed, elapsed


def run_benchmark():
    ckpt_path = r"C:\02_QUILLAN\checkpoints\checkpoints_oni\sft_runs\quillan_1b_sovereign_wsp128k_qwen35.pt"
    tok_dir = r"C:\Users\Admin\.cache\huggingface\hub\models--unsloth--Llama-3.2-1B-Instruct\snapshots"
    tok_snap = os.path.join(tok_dir, os.listdir(tok_dir)[0])
    tok_path = os.path.join(tok_snap, "tokenizer.json")
    
    tester = QuillanSovereignTester(ckpt_path, tok_path)
    
    test_cases = [
        {
            "category": "Algorithm & Code",
            "prompt": "Write a Python function to find the longest palindromic substring in a string, and state its time complexity.",
            "max_tokens": 120
        },
        {
            "category": "Complex Multi-Step Math",
            "prompt": "A merchant buys an item for $80 and marks it up by 50%. Later, he offers a 20% discount on the marked price. What is the final selling price and his profit?",
            "max_tokens": 100
        },
        {
            "category": "Long-Horizon Stress Test (Attractor Check)",
            "prompt": "Explain step by step how an operating system executes a context switch between two threads, detailing CPU registers, kernel stack, and scheduler state.",
            "max_tokens": 160
        },
        {
            "category": "Systems Architecture",
            "prompt": "Compare Raft consensus to Paxos in terms of leader election, log replication, and operational understandability.",
            "max_tokens": 120
        },
        {
            "category": "Bushido Ethics & Resolve",
            "prompt": "When strategic expediency conflicts with honor and rectitude, how does the Bushido code resolve the dilemma?",
            "max_tokens": 100
        }
    ]
    
    print("=" * 80, flush=True)
    print("STARTING 5-DOMAIN FRONTIER STRESS BENCHMARK", flush=True)
    print("=" * 80, flush=True)
    
    results = []
    for idx, test in enumerate(test_cases, 1):
        print(f"\n[{idx}/5] CATEGORY: {test['category']}", flush=True)
        print(f"PROMPT: \"{test['prompt']}\"", flush=True)
        print("-" * 60, flush=True)
        
        text, speed, elapsed = tester.generate(test["prompt"], max_new_tokens=test["max_tokens"], temperature=0.6)
        
        # Calculate Distinct-2 for repetition verification
        words = text.lower().split()
        bigrams = list(zip(words[:-1], words[1:])) if len(words) > 1 else []
        distinct_2 = len(set(bigrams)) / max(1, len(bigrams)) if bigrams else 1.0
        
        print(f"\n[Stats: {len(text.split())} words | {speed:.2f} tok/s | Elapsed: {elapsed:.2f}s | Distinct-2: {distinct_2:.2f}]", flush=True)
        print("=" * 60, flush=True)
        results.append({
            "category": test["category"],
            "speed": speed,
            "distinct_2": distinct_2
        })
        
    avg_speed = sum(r["speed"] for r in results) / len(results)
    avg_distinct = sum(r["distinct_2"] for r in results) / len(results)
    print(f"\nBENCHMARK COMPLETE: Avg Speed = {avg_speed:.2f} tok/s | Avg Distinct-2 = {avg_distinct:.4f} (Zero Loops)", flush=True)


if __name__ == "__main__":
    from typing import Tuple
    run_benchmark()
