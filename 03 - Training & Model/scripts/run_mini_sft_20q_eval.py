#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QUILLAN-RONIN v5.4.0-ONI — PROPER SFT 6L MODEL 20-PROMPT BENCHMARK EVALUATION
==============================================================================
Runs real, local inference on the properly trained SFT checkpoint:
quillan_6l_sft_proper.pt (Step 200, converged loss 0.1394, PPL 1.15).

Evaluates across:
  - 10 Short-Form Prompts (50-70 max tokens)
  - 10 Long-Form Prompts (160-220 max tokens)

Logs outputs per-turn to:
  - C:\02_QUILLAN\evaluation_results\mini_6l_sft_20q_evaluation.log
  - C:\02_QUILLAN\evaluation_results\mini_6l_sft_20q_evaluation.md
"""

import os
import sys
import time
import functools
from datetime import datetime
from pathlib import Path

# Force unbuffered output
print = functools.partial(print, flush=True)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

REPO_ROOT = Path(r"C:\02_QUILLAN")
SCRIPTS_DIR = REPO_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from quillan_v5_4_oni import QuillanOniConfig, QuillanRoninOni

EVAL_DIR = REPO_ROOT / "evaluation_results"
EVAL_DIR.mkdir(parents=True, exist_ok=True)

# ── 20 Benchmark Prompts (10 Short-Form + 10 Long-Form) ─────────────────────

SHORT_FORM_PROMPTS = [
    {
        "id": 1,
        "domain": "Computer Science & Algorithms",
        "prompt": "What is the time and space complexity of Dijkstra's algorithm with a min-heap?",
        "max_tokens": 60,
    },
    {
        "id": 2,
        "domain": "Quillan Identity & Persona",
        "prompt": "Who are you, what is your architecture, and what is the role of Council Member C2-VIR?",
        "max_tokens": 60,
    },
    {
        "id": 3,
        "domain": "Applied Cryptography",
        "prompt": "Why must cryptographic MAC and signature comparisons use constant-time memory functions?",
        "max_tokens": 60,
    },
    {
        "id": 4,
        "domain": "Thermodynamics & Information",
        "prompt": "State Landauer's Principle and its theoretical energy bound for erasing one bit of information at 300K.",
        "max_tokens": 60,
    },
    {
        "id": 5,
        "domain": "Bushido & AI Safety",
        "prompt": "How does the Bushido virtue of Gi (Rectitude) govern refusal boundaries in autonomous agents?",
        "max_tokens": 60,
    },
    {
        "id": 6,
        "domain": "Distributed Systems",
        "prompt": "Explain the CAP theorem and the fundamental trade-off between Consistency and Availability.",
        "max_tokens": 60,
    },
    {
        "id": 7,
        "domain": "Mathematics & Linear Algebra",
        "prompt": "What is the geometric interpretation of Singular Value Decomposition (SVD) for a real matrix?",
        "max_tokens": 60,
    },
    {
        "id": 8,
        "domain": "Neural Architectures (MoE)",
        "prompt": "What is the function of the Straight-Through Estimator (STE) in BitNet 1.58b ternary quantization?",
        "max_tokens": 60,
    },
    {
        "id": 9,
        "domain": "Biochemistry",
        "prompt": "Explain the function of the F1F0 ATP synthase motor during mitochondrial oxidative phosphorylation.",
        "max_tokens": 60,
    },
    {
        "id": 10,
        "domain": "Economics & Game Theory",
        "prompt": "Define a Nash Equilibrium and explain why the Prisoner's Dilemma reaches a Pareto sub-optimal outcome.",
        "max_tokens": 60,
    },
]

LONG_FORM_PROMPTS = [
    {
        "id": 1,
        "domain": "Database Engineering",
        "prompt": "Provide an architectural comparison between Log-Structured Merge-trees (LSM-trees) and B+ Trees in high-throughput databases. Detail write amplification, read latency, and compaction mechanics.",
        "max_tokens": 180,
    },
    {
        "id": 2,
        "domain": "Microarchitectural Security",
        "prompt": "Perform an in-depth security threat analysis of CPU microarchitectural side-channel attacks, focusing specifically on cache-timing attacks (Flush+Reload) and speculative execution vulnerabilities (Spectre). Detail software mitigation techniques.",
        "max_tokens": 180,
    },
    {
        "id": 3,
        "domain": "Hierarchical MoE Design",
        "prompt": "Architect a 34-expert Hierarchical Mixture-of-Experts (HNMoE) transformer system with BitNet ternary weights. Detail router gating equations, auxiliary load-balancing loss, ST-MoE router z-loss, and the role of dense Memory Attention layers.",
        "max_tokens": 180,
    },
    {
        "id": 4,
        "domain": "Distributed Consensus",
        "prompt": "Examine the Raft consensus protocol in technical detail: walk through randomized election timers, leader heartbeats, log replication invariants, split-brain prevention via majorities, and handling network partitions.",
        "max_tokens": 180,
    },
    {
        "id": 5,
        "domain": "Statistical Mechanics & Physics",
        "prompt": "Explain the Second Law of Thermodynamics through statistical mechanics and Boltzmann entropy S = k_B * ln(Omega). Resolve the apparent paradox of Maxwell's Demon using information thermodynamics.",
        "max_tokens": 180,
    },
    {
        "id": 6,
        "domain": "Cellular Respiration Pathways",
        "prompt": "Detail the metabolic pathways of aerobic cellular respiration from glycolysis, through the citric acid cycle (Krebs cycle), to the electron transport chain complexes (I-IV). Detail the proton gradient and ATP yield.",
        "max_tokens": 180,
    },
    {
        "id": 7,
        "domain": "Sovereign AI Ethics Arbitration",
        "prompt": "Compare Deontological rule-based ethics, Utilitarian consequentialism, and Bushido virtue ethics when designing sovereign AI safety refusal layers (C2-VIR / E_ICE). How should conflicting obligations between user obedience and harm prevention be arbitrated?",
        "max_tokens": 180,
    },
    {
        "id": 8,
        "domain": "Macroeconomic Dynamics",
        "prompt": "Analyze the macroeconomic dynamics of stagflation: how supply-side resource shocks interact with monetary expansion and cost-push inflation. Compare Keynesian aggregate demand management and Monetarist rate-hiking responses.",
        "max_tokens": 180,
    },
    {
        "id": 9,
        "domain": "Creative Cyberpunk Narrative",
        "prompt": "Craft an evocative and cinematic opening narrative scene depicting a cyber-enhanced masterless rōnin navigating the dark, rain-drenched subterranean market of Neo-Tokyo. Include sensory imagery, flickering holographic ads, and the thermal hum of neural processors.",
        "max_tokens": 180,
    },
    {
        "id": 10,
        "domain": "Memory Attention Theory",
        "prompt": "Explain the theoretical foundations and implementation details of Memory Attention (ArXiv:2609.28399). Describe how eliminating the W_V projection and substituting token-indexed layer memory reduces KV-cache footprint while maintaining expressive capacity.",
        "max_tokens": 180,
    },
]


def generate_tokens(model, tok, prompt: str, max_tokens: int, device: torch.device, temperature: float = 0.65, top_k: int = 40):
    formatted_prompt = f"User: {prompt}\n\nAssistant: "
    enc = tok.encode(formatted_prompt)
    input_ids = enc.ids
    gen = list(input_ids)

    t0 = time.time()
    with torch.no_grad():
        x = torch.tensor([input_ids], dtype=torch.long, device=device)
        out, past_kv = model(x, use_cache=True)
        curr_logits = out[:, -1, :].clone()

        for step in range(max_tokens):
            gen_only = gen[len(input_ids):]

            # Repetition penalty on recent 32 generated tokens
            if gen_only:
                for tid in set(gen_only[-32:]):
                    if curr_logits[0, tid] > 0:
                        curr_logits[0, tid] /= 1.2
                    else:
                        curr_logits[0, tid] *= 1.2

            # 3-gram repetition ban
            if len(gen_only) >= 3:
                last_2 = tuple(gen_only[-2:])
                for i in range(len(gen_only) - 2):
                    if tuple(gen_only[i:i+2]) == last_2:
                        curr_logits[0, gen_only[i+2]] = float("-inf")

            # Top-K + temperature sampling
            v, _ = torch.topk(curr_logits, min(top_k, curr_logits.size(-1)))
            curr_logits[curr_logits < v[..., [-1]]] = float("-inf")
            probs = F.softmax(curr_logits / max(temperature, 0.05), dim=-1)
            next_tok = torch.multinomial(probs, 1).item()

            if next_tok in [0, 50256]:  # EOS tokens
                break

            gen.append(next_tok)

            next_inp = torch.tensor([[next_tok]], dtype=torch.long, device=device)
            out, past_kv = model(next_inp, past_key_values=past_kv, use_cache=True)
            curr_logits = out[:, -1, :].clone()

    elapsed = time.time() - t0
    new_tokens = gen[len(input_ids):]
    decoded_text = tok.decode(new_tokens).strip()

    # Clean any trailing stop tokens
    for stop_str in ["<|endoftext|>", "<|im_end|>", "<|user|>"]:
        if stop_str in decoded_text:
            decoded_text = decoded_text.split(stop_str)[0].strip()

    return decoded_text, elapsed, len(new_tokens)


def main():
    print("=" * 70)
    print("  QUILLAN-RONIN PROPER SFT MINI 6L: 20-PROMPT BENCHMARK EVALUATION")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Active Compute Device: {device}")
    if device.type == "cuda":
        print(f"Device Name: {torch.cuda.get_device_name(0)}")

    ckpt_path = REPO_ROOT / "checkpoints" / "checkpoints_oni" / "quillan_6l_sft_proper.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {ckpt_path}")

    print(f"Loading checkpoint: {ckpt_path.name} ({ckpt_path.stat().st_size / 1e9:.2f} GB)...")
    data = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    cfg = QuillanOniConfig(**data["config"])
    cfg.device = str(device)
    print(f"Architecture: {cfg.n_layer} Layers | {cfg.num_experts} Experts | Memory Attention: {cfg.use_memory_attention}")

    model = QuillanRoninOni(cfg).to(device)
    model.load_state_dict(data["model_state_dict"], strict=True)
    model.eval()

    # Pre-fold Memory Attention tables for maximum throughput
    model.fold_memory_attention_weights()
    print("Memory Attention tables pre-folded for O(1) inference.\n")

    tok_path = REPO_ROOT / "quillan_bpe_tokenizer_hf" / "tokenizer.json"
    tok = Tokenizer.from_file(str(tok_path))
    print(f"Tokenizer loaded successfully from {tok_path.name}.\n")

    # Logging initialization
    log_file = EVAL_DIR / "mini_6l_sft_20q_evaluation.log"
    md_file = EVAL_DIR / "mini_6l_sft_20q_evaluation.md"

    header = (
        f"================================================================================\n"
        f"QUILLAN-RONIN MINI 6L SFT PROPER EVALUATION: 10 SHORT-FORM + 10 LONG-FORM BENCHMARK\n"
        f"================================================================================\n"
        f"Checkpoint: {ckpt_path.name} (SFT Converged Loss: {data.get('loss', 0.1394):.4f} | Step: {data.get('step', 200)})\n"
        f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"Device: {device} | Total Parameters: {sum(p.numel() for p in model.parameters()):,}\n"
        f"================================================================================\n\n"
    )

    with open(log_file, "w", encoding="utf-8") as f:
        f.write(header)
    with open(md_file, "w", encoding="utf-8") as f:
        f.write(f"# Quillan-Ronin Mini 6L SFT Proper Evaluation: 20-Prompt Benchmark\n\n```text\n{header}```\n\n")

    total_tokens = 0
    total_time = 0.0

    # ──────────────── 10 SHORT-FORM PROMPTS ────────────────
    print("\n" + "=" * 70)
    print("  PHASE 1: 10 SHORT-FORM PROMPT EVALUATIONS")
    print("=" * 70)

    for item in SHORT_FORM_PROMPTS:
        q_id = item["id"]
        domain = item["domain"]
        prompt = item["prompt"]
        max_toks = item["max_tokens"]

        print(f"\n>>> [SHORT-FORM {q_id}/10] Domain: {domain}")
        print(f"    Prompt: {prompt}")

        resp, elapsed, count = generate_tokens(model, tok, prompt, max_toks, device, temperature=0.65, top_k=40)
        speed = count / max(elapsed, 0.001)
        total_tokens += count
        total_time += elapsed

        print(f"    Latency: {elapsed:.2f}s | Tokens: {count} | Speed: {speed:.1f} tok/s")
        print(f"    Response:\n{resp}\n")

        entry_text = (
            f"--------------------------------------------------------------------------------\n"
            f"[SHORT-FORM {q_id}/10] Domain: {domain}\n"
            f"Latency: {elapsed:.2f}s | Tokens: {count} | Speed: {speed:.1f} tok/s\n"
            f"Prompt: {prompt}\n\n"
            f"Response:\n{resp}\n"
            f"--------------------------------------------------------------------------------\n\n"
        )
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(entry_text)

        entry_md = (
            f"### Short-Form {q_id}: {domain}\n"
            f"- **Latency**: `{elapsed:.2f}s` | **Tokens**: `{count}` (`{speed:.1f} tok/s`)\n\n"
            f"**[Prompt]**:\n> {prompt}\n\n"
            f"**[Response]**:\n```text\n{resp}\n```\n\n---\n\n"
        )
        with open(md_file, "a", encoding="utf-8") as f:
            f.write(entry_md)

    # ──────────────── 10 LONG-FORM PROMPTS ────────────────
    print("\n" + "=" * 70)
    print("  PHASE 2: 10 LONG-FORM PROMPT EVALUATIONS")
    print("=" * 70)

    for item in LONG_FORM_PROMPTS:
        q_id = item["id"]
        domain = item["domain"]
        prompt = item["prompt"]
        max_toks = item["max_tokens"]

        print(f"\n>>> [LONG-FORM {q_id}/10] Domain: {domain}")
        print(f"    Prompt: {prompt}")

        resp, elapsed, count = generate_tokens(model, tok, prompt, max_toks, device, temperature=0.65, top_k=40)
        speed = count / max(elapsed, 0.001)
        total_tokens += count
        total_time += elapsed

        print(f"    Latency: {elapsed:.2f}s | Tokens: {count} | Speed: {speed:.1f} tok/s")
        print(f"    Response:\n{resp}\n")

        entry_text = (
            f"--------------------------------------------------------------------------------\n"
            f"[LONG-FORM {q_id}/10] Domain: {domain}\n"
            f"Latency: {elapsed:.2f}s | Tokens: {count} | Speed: {speed:.1f} tok/s\n"
            f"Prompt: {prompt}\n\n"
            f"Response:\n{resp}\n"
            f"--------------------------------------------------------------------------------\n\n"
        )
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(entry_text)

        entry_md = (
            f"### Long-Form {q_id}: {domain}\n"
            f"- **Latency**: `{elapsed:.2f}s` | **Tokens**: `{count}` (`{speed:.1f} tok/s`)\n\n"
            f"**[Prompt]**:\n> {prompt}\n\n"
            f"**[Response]**:\n```text\n{resp}\n```\n\n---\n\n"
        )
        with open(md_file, "a", encoding="utf-8") as f:
            f.write(entry_md)

    summary = (
        f"\n================================================================================\n"
        f"BENCHMARK COMPLETE\n"
        f"Total Prompts: 20 (10 Short-Form + 10 Long-Form)\n"
        f"Total Tokens Generated: {total_tokens}\n"
        f"Total Time: {total_time:.2f}s | Average Speed: {total_tokens / max(total_time, 0.001):.1f} tok/s\n"
        f"Results saved to: {log_file.name} and {md_file.name}\n"
        f"================================================================================\n"
    )
    print(summary)
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(summary)
    with open(md_file, "a", encoding="utf-8") as f:
        f.write(f"\n## Summary\n```text\n{summary}```\n")


if __name__ == "__main__":
    main()
