#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
👑 QUILLAN-RONIN v5.3.1 — EXHAUSTIVE MULTI-HORIZON EVALUATION SUITE
Covers 16 comprehensive benchmark prompts across 4 distinct horizon tiers:
  - Tier 1: Crisp Short-Form Factuals (30-80 tokens)
  - Tier 2: Multi-Step Deductive Logic & Mathematics (100-200 tokens)
  - Tier 3: Long-Form Technical Architecture & Algorithmic Code (250-450 tokens)
  - Tier 4: Complex Scientific, Theoretical & Edge-Case Reasoning (200-400 tokens)
"""

import sys
import time
import json
import torch
import torch.nn.functional as F
from collections import Counter
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Callable

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

SCRIPTS_DIR = Path(r"C:\02_QUILLAN\scripts")
SCRATCH_DIR = Path(r"C:\Users\Admin\.gemini\antigravity-ide\brain\47d0b748-384a-4cc6-bcb6-f13c52f1d9d9\scratch")
CKPT_DIR = Path(r"C:\02_QUILLAN\checkpoints\checkpoints_sft")

sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(SCRATCH_DIR))

from quillan_v10_unrolled_sovereign import QuillanUnrolledSovereign, QuillanUnrolledConfig
from sovereign_inference_engine import SovereignTokenizer

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

@dataclass
class BenchmarkPrompt:
    tier: str
    category: str
    question: str
    max_tokens: int
    temp: float
    top_p: float
    top_k: int
    freq_penalty: float
    pres_penalty: float

BENCHMARK_SUITE: List[BenchmarkPrompt] = [
    # ─── TIER 1: CRISP SHORT-FORM FACTUALS ──────────────────────────────────
    BenchmarkPrompt(
        tier="Tier 1: Short-Form Factual",
        category="Physics Constant",
        question="What is the exact speed of light in a vacuum in SI units?",
        max_tokens=60,
        temp=0.20,
        top_p=0.80,
        top_k=25,
        freq_penalty=0.50,
        pres_penalty=0.30
    ),
    BenchmarkPrompt(
        tier="Tier 1: Short-Form Factual",
        category="Networking (HTTP)",
        question="What is HTTP status code 403 and how does it differ from 401?",
        max_tokens=80,
        temp=0.20,
        top_p=0.80,
        top_k=25,
        freq_penalty=0.50,
        pres_penalty=0.30
    ),
    BenchmarkPrompt(
        tier="Tier 1: Short-Form Factual",
        category="Chemistry",
        question="What is the chemical symbol and atomic number of Gold?",
        max_tokens=50,
        temp=0.20,
        top_p=0.80,
        top_k=25,
        freq_penalty=0.50,
        pres_penalty=0.30
    ),
    BenchmarkPrompt(
        tier="Tier 1: Short-Form Factual",
        category="Operating Systems",
        question="What is the purpose of an inode in a Unix filesystem?",
        max_tokens=80,
        temp=0.20,
        top_p=0.80,
        top_k=25,
        freq_penalty=0.50,
        pres_penalty=0.30
    ),

    # ─── TIER 2: MULTI-STEP DEDUCTIVE LOGIC & MATHEMATICS ──────────────────
    BenchmarkPrompt(
        tier="Tier 2: Deductive Logic & Math",
        category="Formal Logic (Syllogism)",
        question="If all humans are mortal and Socrates is human, is Socrates mortal? Provide the formal syllogistic proof.",
        max_tokens=150,
        temp=0.25,
        top_p=0.80,
        top_k=30,
        freq_penalty=0.50,
        pres_penalty=0.30
    ),
    BenchmarkPrompt(
        tier="Tier 2: Deductive Logic & Math",
        category="Bayes' Theorem",
        question="State Bayes' theorem mathematically and define each variable (prior, posterior, likelihood, evidence).",
        max_tokens=180,
        temp=0.25,
        top_p=0.80,
        top_k=30,
        freq_penalty=0.50,
        pres_penalty=0.30
    ),
    BenchmarkPrompt(
        tier="Tier 2: Deductive Logic & Math",
        category="Algorithmic Complexity",
        question="Compare the average-case and worst-case time complexity of Quicksort, Mergesort, and Heapsort.",
        max_tokens=180,
        temp=0.25,
        top_p=0.80,
        top_k=30,
        freq_penalty=0.50,
        pres_penalty=0.30
    ),
    BenchmarkPrompt(
        tier="Tier 2: Deductive Logic & Math",
        category="Discrete Math (Prime Factorization)",
        question="Explain why the prime factorization of any positive integer greater than 1 is unique (Fundamental Theorem of Arithmetic).",
        max_tokens=180,
        temp=0.25,
        top_p=0.80,
        top_k=30,
        freq_penalty=0.50,
        pres_penalty=0.30
    ),

    # ─── TIER 3: LONG-FORM TECHNICAL ARCHITECTURE & CODE ────────────────────
    BenchmarkPrompt(
        tier="Tier 3: Long-Form Technical & Code",
        category="Python Algorithm (LRU Cache)",
        question="Write a Python class implementing an LRU (Least Recently Used) Cache with get and put methods.",
        max_tokens=300,
        temp=0.25,
        top_p=0.80,
        top_k=30,
        freq_penalty=0.50,
        pres_penalty=0.30
    ),
    BenchmarkPrompt(
        tier="Tier 3: Long-Form Technical & Code",
        category="Distributed Systems",
        question="Explain how the Raft consensus algorithm handles leader election and log replication across nodes.",
        max_tokens=260,
        temp=0.25,
        top_p=0.80,
        top_k=30,
        freq_penalty=0.50,
        pres_penalty=0.30
    ),
    BenchmarkPrompt(
        tier="Tier 3: Long-Form Technical & Code",
        category="Database Architecture",
        question="Describe how B-Tree indexes work in relational databases and why they are preferred over Hash indexes for range queries.",
        max_tokens=250,
        temp=0.25,
        top_p=0.80,
        top_k=30,
        freq_penalty=0.50,
        pres_penalty=0.30
    ),
    BenchmarkPrompt(
        tier="Tier 3: Long-Form Technical & Code",
        category="Python Concurrency",
        question="Write a Python script demonstrating how to use threading.Lock to prevent a race condition when incrementing a shared counter.",
        max_tokens=280,
        temp=0.25,
        top_p=0.80,
        top_k=30,
        freq_penalty=0.50,
        pres_penalty=0.30
    ),

    # ─── TIER 4: COMPLEX SCIENTIFIC & THEORETICAL REASONING ────────────────
    BenchmarkPrompt(
        tier="Tier 4: Complex Scientific & Theoretical",
        category="Theoretical Physics (E=mc²)",
        question="State Einstein's mass-energy equivalence equation and explain its physical implications.",
        max_tokens=200,
        temp=0.25,
        top_p=0.80,
        top_k=30,
        freq_penalty=0.50,
        pres_penalty=0.30
    ),
    BenchmarkPrompt(
        tier="Tier 4: Complex Scientific & Theoretical",
        category="Biochemistry (Photosynthesis)",
        question="Explain the chemical process of photosynthesis in plants, including the balanced chemical equation.",
        max_tokens=220,
        temp=0.25,
        top_p=0.80,
        top_k=30,
        freq_penalty=0.50,
        pres_penalty=0.30
    ),
    BenchmarkPrompt(
        tier="Tier 4: Complex Scientific & Theoretical",
        category="Quantum Mechanics (EPR Paradox)",
        question="What is quantum entanglement, and how did Bell's theorem resolve Einstein's EPR paradox regarding local hidden variables?",
        max_tokens=240,
        temp=0.25,
        top_p=0.80,
        top_k=30,
        freq_penalty=0.50,
        pres_penalty=0.30
    ),
    BenchmarkPrompt(
        tier="Tier 4: Complex Scientific & Theoretical",
        category="Computational Complexity (P vs NP)",
        question="Explain the P versus NP problem in theoretical computer science and what proving P = NP would imply for modern cryptography.",
        max_tokens=240,
        temp=0.25,
        top_p=0.80,
        top_k=30,
        freq_penalty=0.50,
        pres_penalty=0.30
    )
]

@torch.no_grad()
def stateful_generate_stream(
    model: QuillanUnrolledSovereign,
    tokenizer: SovereignTokenizer,
    prompt_tokens: List[int],
    max_tokens: int = 150,
    temp: float = 0.25,
    top_k: int = 30,
    top_p: float = 0.80,
    frequency_penalty: float = 0.50,
    presence_penalty: float = 0.30,
    callback: Optional[Callable[[str], None]] = None
) -> List[int]:
    model.eval()
    gen = list(prompt_tokens)
    device = next(model.parameters()).device
    
    inp = torch.tensor([gen], dtype=torch.long, device=device)
    logits, kv_cache = model.forward(inp, use_cache=True)
    
    generated_tokens: List[int] = []
    
    for _ in range(max_tokens):
        curr_logits = logits[:, -1, :].clone()
        
        if generated_tokens:
            counts = Counter(generated_tokens)
            for t, count in counts.items():
                curr_logits[0, t] -= (count * frequency_penalty + presence_penalty)
        
        if temp <= 0.01:
            next_tok = int(torch.argmax(curr_logits, dim=-1).item())
        else:
            curr_logits = curr_logits / max(0.05, temp)
            probs = F.softmax(curr_logits, dim=-1)
            
            if top_k > 0:
                val_k, _ = torch.topk(probs, min(top_k, probs.size(-1)))
                probs[probs < val_k[:, -1:]] = 0.0
                probs = probs / probs.sum(dim=-1, keepdim=True)
                
            if top_p < 1.0:
                sorted_p, sorted_i = torch.sort(probs, descending=True, dim=-1)
                cum_p = torch.cumsum(sorted_p, dim=-1)
                cutoff = cum_p > top_p
                cutoff[..., 1:] = cutoff[..., :-1].clone()
                cutoff[..., 0] = False
                sorted_p[cutoff] = 0.0
                sorted_p = sorted_p / sorted_p.sum(dim=-1, keepdim=True)
                probs.scatter_(1, sorted_i, sorted_p)
                
            next_tok = int(torch.multinomial(probs, num_samples=1).item())
            
        gen.append(next_tok)
        generated_tokens.append(next_tok)
        
        tok_str = tokenizer.decode([next_tok])
        if callback:
            callback(tok_str)
            
        if next_tok in (50256,):  # EOT / EOS
            break
        full_gen_text = tokenizer.decode(generated_tokens)
        if any(stop_seq in full_gen_text for stop_seq in ["<|im_end|>", "<|endoftext|>", "\n\n\n\n"]):
            break
            
        inp_single = torch.tensor([[next_tok]], dtype=torch.long, device=device)
        logits, kv_cache = model.forward(inp_single, past_key_values=kv_cache, use_cache=True)
        
    return generated_tokens

def run_exhaustive_suite():
    # Priority order: best trained checkpoint first
    candidates = [
        CKPT_DIR / "quillan_frontier_v2_best.pt",
        CKPT_DIR / "quillan_frontier_generalization_best.pt",
        CKPT_DIR / "quillan_sovereign_gold_best.pt",
        CKPT_DIR / "quillan_direct_factual_best.pt",
        CKPT_DIR / "quillan_gold_precision_best.pt",
    ]
    ckpt_path = None
    for c in candidates:
        if c.exists():
            ckpt_path = c
            break
    if ckpt_path is None:
        raise FileNotFoundError("No checkpoint found. Run training first.")

    print("==================================================================", flush=True)
    print("   👑 QUILLAN-RONIN v5.3.1 — EXHAUSTIVE MULTI-HORIZON EVALUATION", flush=True)
    print(f"   Model Checkpoint: {ckpt_path.name}", flush=True)
    print(f"   Total Test Prompts: {len(BENCHMARK_SUITE)} (4 Distinct Horizon Tiers)", flush=True)
    print("==================================================================\n", flush=True)

    device = torch.device("cpu")
    tokenizer = SovereignTokenizer("gpt2")
    cfg = QuillanUnrolledConfig()

    print("[*] Loading Quillan-Ronin Sovereign Brain...", flush=True)
    model = QuillanUnrolledSovereign(cfg).to(device)
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    sd = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(sd, strict=False)
    model.eval()
    print("[+] Model loaded successfully!\n", flush=True)

    results_summary = []
    total_tokens_all = 0
    total_time_all = 0.0

    current_tier = ""
    for idx, bp in enumerate(BENCHMARK_SUITE, 1):
        if bp.tier != current_tier:
            current_tier = bp.tier
            print(f"\n{'='*70}", flush=True)
            print(f"   🚀 {current_tier.upper()}", flush=True)
            print(f"{'='*70}\n", flush=True)

        # Use the sovereign chat template format (matches training data)
        prompt_str = f"<|user|>\n{bp.question}\n<|assistant|>\n"
        prompt_toks = tokenizer.encode(prompt_str)
        
        print(f"[{idx}/{len(BENCHMARK_SUITE)}] [{bp.category}]", flush=True)
        print(f"Q: {bp.question}", flush=True)
        print("-" * 65, flush=True)

        streamed_pieces: List[str] = []
        t0 = time.time()
        def on_token(t: str):
            streamed_pieces.append(t)
            print(t, end="", flush=True)

        gen_tokens = stateful_generate_stream(
            model=model,
            tokenizer=tokenizer,
            prompt_tokens=prompt_toks,
            max_tokens=bp.max_tokens,
            temp=bp.temp,
            top_k=bp.top_k,
            top_p=bp.top_p,
            frequency_penalty=bp.freq_penalty,
            presence_penalty=bp.pres_penalty,
            callback=on_token
        )
        elapsed = time.time() - t0
        num_tokens = len(gen_tokens)
        tps = num_tokens / max(0.001, elapsed)
        
        total_tokens_all += num_tokens
        total_time_all += elapsed

        full_ans = tokenizer.decode(gen_tokens).strip()
        for tag in ["<|im_end|>", "<|endoftext|>"]:
            full_ans = full_ans.replace(tag, "").strip()

        print("\n" + "-" * 65, flush=True)
        print(f"Stats: {num_tokens} tokens in {elapsed:.2f}s ({tps:.1f} tok/s)\n", flush=True)

        results_summary.append({
            "index": idx,
            "tier": bp.tier,
            "category": bp.category,
            "question": bp.question,
            "answer": full_ans,
            "tokens": num_tokens,
            "time_sec": round(elapsed, 2),
            "tok_per_sec": round(tps, 1)
        })

    print("\n==================================================================", flush=True)
    print("   🏆 EXHAUSTIVE MULTI-HORIZON BENCHMARK COMPLETE", flush=True)
    print(f"   Total Generated Tokens: {total_tokens_all}", flush=True)
    print(f"   Total Evaluation Time:  {total_time_all:.2f}s", flush=True)
    print(f"   Overall Average Speed:  {total_tokens_all/max(0.001, total_time_all):.1f} tok/s", flush=True)
    print("==================================================================", flush=True)

    summary_file = Path(r"C:\02_QUILLAN\benchmark_results.json")
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(results_summary, f, indent=2)
    print(f"[+] Summary report saved to {summary_file}", flush=True)

if __name__ == "__main__":
    run_exhaustive_suite()
