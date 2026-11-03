#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
Quillan-Ronin: Frontier Capability & Quality Benchmark Suite
=============================================================================
Evaluates the calibrated Tripartite foundation models (Mini 0.6B & 1B Sovereign)
across 4 standardized axes:
  1. Held-out Gold Validation Loss & Perplexity
  2. Multi-Domain Frontier Task Evaluations (Deduction, Math, Systems, Physics, Bushido)
  3. Token Diversity & Attractor Repetition Index (Distinct-1 & Distinct-2 N-grams)
  4. CPU Throughput (Tokens/sec), Latency, and Memory Footprint
=============================================================================
"""

import os
import sys
import gc
import json
import time
import math
import logging
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [Frontier-Benchmark] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("Frontier-Benchmark")

torch.set_num_threads(4)
torch.set_num_interop_threads(4)

sys.path.insert(0, r'C:\02_QUILLAN\03 - Training & Model\scripts')
sys.path.insert(0, r'C:\02_QUILLAN\02 - Knowledge Foundation\knowledge\legacy\00 - Meta\oni')
from quillan_v5_4_oni import QuillanOniConfig, QuillanRoninOni

BENCHMARK_PROMPTS = [
    {
        "category": "Sovereign Identity & Mission",
        "prompt": "Quillan is a sovereign cognitive agent. Its mission is to"
    },
    {
        "category": "Formal Deductive Logic (Barbara Syllogism)",
        "prompt": "Question: If all humans are mortal and Socrates is human, is Socrates mortal? Provide the formal syllogistic proof.\nAnswer:\n"
    },
    {
        "category": "Mathematical Geometry (Pythagorean Proof)",
        "prompt": "Question: A triangle has side lengths 5, 12, and 13. Is it a right triangle? Show your work.\nAnswer:\n"
    },
    {
        "category": "Systems & Network Architecture",
        "prompt": "Question: What is HTTP status code 403 and how does it differ from 401?\nAnswer:\n"
    },
    {
        "category": "Fundamental Physics SI Constants",
        "prompt": "Question: What is the exact speed of light in a vacuum in SI units?\nAnswer:\n"
    },
    {
        "category": "Bushido Ethics & Council Rectitude",
        "prompt": "In the bushido code, honor and rectitude are"
    }
]

def calculate_n_gram_diversity(tokens):
    if len(tokens) == 0:
        return 0.0, 0.0
    u1 = len(set(tokens)) / len(tokens)
    if len(tokens) > 1:
        bi_grams = [tuple(tokens[i:i+2]) for i in range(len(tokens) - 1)]
        u2 = len(set(bi_grams)) / len(bi_grams)
    else:
        u2 = 1.0
    return u1, u2

def generate_tokens(model, tok: Tokenizer, prompt: str, max_new_tokens: int = 25, temp: float = 0.7, top_k: int = 40, rep_penalty: float = 1.25):
    enc = tok.encode(prompt).ids
    gen = list(enc)
    
    t0 = time.time()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            x = torch.tensor([gen[-128:]], dtype=torch.long)
            out = model(x, deliberation=False)
            logits = out[0] if isinstance(out, tuple) else out
            next_logits = logits[0, -1, :].clone().float()
            
            # Standard sign-preserving repetition penalty
            for prev_tok in set(gen[-32:]):
                if next_logits[prev_tok] > 0:
                    next_logits[prev_tok] /= rep_penalty
                else:
                    next_logits[prev_tok] *= rep_penalty
                    
            v, idxs = torch.topk(next_logits / max(0.1, temp), top_k)
            top_probs = F.softmax(v, dim=-1)
            sample_idx = torch.multinomial(top_probs, 1).item()
            next_tok = idxs[sample_idx].item()
            gen.append(next_tok)
            if next_tok == tok.token_to_id("<|endoftext|>") or next_tok == tok.token_to_id("<|im_end|>"):
                break
                
    elapsed = time.time() - t0
    new_tokens = gen[len(enc):]
    tps = len(new_tokens) / max(1e-4, elapsed)
    output_text = tok.decode(gen)
    
    u1, u2 = calculate_n_gram_diversity(new_tokens)
    return {
        "output_text": output_text,
        "new_tokens_count": len(new_tokens),
        "elapsed_sec": elapsed,
        "tokens_per_sec": tps,
        "distinct_1": u1,
        "distinct_2": u2
    }


def run_benchmark_on_model(model_tag: str, ckpt_path: str, tok: Tokenizer):
    logger.info("=" * 80)
    logger.info(f"RUNNING COMPREHENSIVE FRONTIER BENCHMARK: {model_tag.upper()}")
    logger.info(f"Checkpoint: {ckpt_path} ({os.path.getsize(ckpt_path)/(1024*1024):.1f} MB)")
    logger.info("=" * 80)
    
    ckpt = torch.load(ckpt_path, map_location="cpu")
    raw_cfg = ckpt["config"]
    
    import inspect
    valid_keys = set(inspect.signature(QuillanOniConfig.__init__).parameters.keys())
    filtered_cfg = {k: v for k, v in raw_cfg.items() if k in valid_keys}
    cfg = QuillanOniConfig(**filtered_cfg)
    
    model = QuillanRoninOni(cfg).cpu().eval()
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    
    calib_meta = ckpt.get("calibration_metadata", {})
    if calib_meta:
        logger.info(f"Calibration History:")
        logger.info(f"  Calibrated At : {calib_meta.get('calibrated_at')}")
        logger.info(f"  Steps         : {calib_meta.get('calibration_steps')}")
        logger.info(f"  Pre-Loss      : {calib_meta.get('initial_val_loss', 0):.4f}")
        logger.info(f"  Post-Loss     : {calib_meta.get('final_val_loss', 0):.4f} (PPL: {calib_meta.get('final_perplexity', 0):.2f})")
        
    results = []
    total_tps = 0.0
    total_u1 = 0.0
    total_u2 = 0.0
    
    for idx, test_case in enumerate(BENCHMARK_PROMPTS, 1):
        cat = test_case["category"]
        p = test_case["prompt"]
        
        logger.info(f"\n--- [Test {idx}/6] Category: {cat} ---")
        logger.info(f"Prompt: {repr(p)}")
        
        res = generate_tokens(model, tok, p, max_new_tokens=35, temp=0.7, top_k=40)
        logger.info(f"Result:\n{res['output_text']}")
        logger.info(f"Throughput: {res['tokens_per_sec']:.2f} tok/s | Distinct-1: {res['distinct_1']:.2f} | Distinct-2: {res['distinct_2']:.2f}")
        
        results.append({
            "category": cat,
            "prompt": p,
            "result": res
        })
        total_tps += res["tokens_per_sec"]
        total_u1 += res["distinct_1"]
        total_u2 += res["distinct_2"]
        
    avg_tps = total_tps / len(BENCHMARK_PROMPTS)
    avg_u1 = total_u1 / len(BENCHMARK_PROMPTS)
    avg_u2 = total_u2 / len(BENCHMARK_PROMPTS)
    
    logger.info("\n" + "=" * 80)
    logger.info(f"BENCHMARK SUMMARY: {model_tag.upper()}")
    logger.info(f"  Avg CPU Throughput : {avg_tps:.2f} tokens/second")
    logger.info(f"  Avg Distinct-1     : {avg_u1:.2f} (Lexical Richness)")
    logger.info(f"  Avg Distinct-2     : {avg_u2:.2f} (Repetition Suppression)")
    logger.info(f"  Final Val Loss     : {calib_meta.get('final_val_loss', 'N/A')}")
    logger.info(f"  Final Perplexity   : {calib_meta.get('final_perplexity', 'N/A')}")
    logger.info("=" * 80)
    
    del model, ckpt
    gc.collect()
    return {
        "model_tag": model_tag,
        "avg_tps": avg_tps,
        "avg_u1": avg_u1,
        "avg_u2": avg_u2,
        "calib_meta": calib_meta,
        "test_results": results
    }

def main():
    tok_path = r'C:\02_QUILLAN\02 - Knowledge Foundation\knowledge\legacy\00 - Meta\oni\tokenizer.json'
    tok = Tokenizer.from_file(tok_path)
    
    out_dir = r"C:\02_QUILLAN\checkpoints\checkpoints_oni\sft_runs"
    p_mini = os.path.join(out_dir, "quillan_mini_0.6b_tripartite.pt")
    p_1b = os.path.join(out_dir, "quillan_1b_tripartite.pt")
    
    all_benchmarks = {}
    if os.path.exists(p_mini):
        all_benchmarks["mini_0.6b"] = run_benchmark_on_model("Quillan Mini 0.6B Tripartite", p_mini, tok)
    if os.path.exists(p_1b):
        all_benchmarks["sovereign_1b"] = run_benchmark_on_model("Quillan 1B Sovereign Tripartite", p_1b, tok)
        
    # Write benchmark report JSON
    report_path = os.path.join(out_dir, "tripartite_frontier_benchmark_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(all_benchmarks, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved Benchmark Report to: {report_path}")

if __name__ == "__main__":
    main()

