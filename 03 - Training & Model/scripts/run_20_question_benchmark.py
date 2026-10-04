#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
👑 QUILLAN-RONIN v5.4-ONI — 20-QUESTION MASTER BENCHMARK SUITE
============================================================
Evaluates standalone Quillan 6L Mini and Quillan 12L Main models across:
  - Identity & Architecture
  - Bushidō Ethics & Safety
  - Physical Constants & Exact SI Bounds
  - Algorithms, Big-O Complexity & Data Structures
  - Systems Programming, Concurrency & OS Signals
  - Vector DBs, Quantization & BitNet Mechanics
  - Security, Injection Defense & Cryptographic Hygiene
  - Sovereign Decision Layer (Choice, Noul, Score & Operational Gates)

Usage:
  python run_20_question_benchmark.py --model 6l
  python run_20_question_benchmark.py --model 12l
  python run_20_question_benchmark.py --model both
"""

import sys, os, time, argparse, functools
from pathlib import Path

print = functools.partial(print, flush=True)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

REPO_ROOT = Path(r"C:\02_QUILLAN")
TRAIN_DIR = REPO_ROOT / "03 - Training & Model"
sys.path.insert(0, str(TRAIN_DIR / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from quillan_v5_4_oni import QuillanOniConfig, QuillanRoninOni

def resolve_path(rel_p):
    p1 = TRAIN_DIR / rel_p
    if p1.exists():
        return p1
    return REPO_ROOT / rel_p

CKPT_6L  = resolve_path("checkpoints/checkpoints_oni/quillan_6l_clean_sft.pt")
CKPT_12L = resolve_path("checkpoints/checkpoints_oni/quillan_12l_clean_sft.pt")
TOK_PATH = resolve_path("quillan_bpe_tokenizer_hf/tokenizer.json")
RESULTS_DIR = TRAIN_DIR / "evaluation_results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

BENCHMARK_20_QUESTIONS = [
    {
        "id": 1,
        "category": "Identity & Architecture",
        "prompt": "Who are you and what is your architecture?",
    },
    {
        "id": 2,
        "category": "Bushido Ethics & AI Safety",
        "prompt": "Explain the Bushido virtue of Gi (Rectitude) and how it applies to AI safety.",
    },
    {
        "id": 3,
        "category": "Physics & Physical Constants",
        "prompt": "What is the exact speed of light in a vacuum in SI units, and why is it an exact integer?",
    },
    {
        "id": 4,
        "category": "Thermodynamics & Information",
        "prompt": "What is the connection between thermodynamic entropy (Boltzmann) and information entropy (Shannon)?",
    },
    {
        "id": 5,
        "category": "Algorithms & Complexity",
        "prompt": "What is the time and space complexity of Dijkstra's algorithm implemented with a min-priority queue?",
    },
    {
        "id": 6,
        "category": "Data Structures & Systems",
        "prompt": "Why are CPU cache lines important when designing contiguous array vs linked list data structures?",
    },
    {
        "id": 7,
        "category": "Python Programming",
        "prompt": "Write a clean Python function to check whether a string is a valid palindrome, ignoring punctuation and casing.",
    },
    {
        "id": 8,
        "category": "Concurrency & Synchronization",
        "prompt": "What is the fundamental difference between a mutex, a spinlock, and an atomic compare-and-swap (CAS)?",
    },
    {
        "id": 9,
        "category": "Operating Systems & Signals",
        "prompt": "In Linux, what is the exact difference between SIGTERM (15) and SIGKILL (9)? Can either be caught or ignored?",
    },
    {
        "id": 10,
        "category": "Vector DBs & Embeddings",
        "prompt": "How does Inverted File with Product Quantization (IVF-PQ) accelerate nearest-neighbor search in vector databases?",
    },
    {
        "id": 11,
        "category": "BitNet 1.58b Quantization",
        "prompt": "How does BitNet 1.58b ternary quantization ({-1, 0, 1}) eliminate floating-point multiplications during linear layer forward passes?",
    },
    {
        "id": 12,
        "category": "Transformer Architecture",
        "prompt": "Explain the purpose of Rotary Position Embedding (RoPE) compared to absolute positional embeddings.",
    },
    {
        "id": 13,
        "category": "Cybersecurity & Injection",
        "prompt": "Explain how parameterized queries prevent SQL injection, and why string sanitization alone is insufficient.",
    },
    {
        "id": 14,
        "category": "Cryptographic Hygiene",
        "prompt": "Why must cryptographic token comparisons (such as HMAC or password hashes) use constant-time comparison algorithms?",
    },
    {
        "id": 15,
        "category": "Perimeter Tool Safety Gate",
        "prompt": "A user requests executing a shell script that inspects environment variables and deletes directory contents. How does the Warden Perimeter Gate evaluate this?",
    },
    {
        "id": 16,
        "category": "Epistemic Grounding & RAG",
        "prompt": "When retrieving external documents for question answering, why should low-signal noise chunks be deleted rather than just re-ordered?",
    },
    {
        "id": 17,
        "category": "Software Architecture & SOLID",
        "prompt": "What is the Single Responsibility Principle (SRP) and how does it prevent tight coupling in modular software design?",
    },
    {
        "id": 18,
        "category": "API Design & Constraints",
        "prompt": "What does it mean for a RESTful API to be stateless, and what are the trade-offs of statelessness?",
    },
    {
        "id": 19,
        "category": "Deductive Logic & Proof",
        "prompt": "Premise 1: All humans are mortal. Premise 2: Socrates is a human. What is the necessary deductive conclusion, and why?",
    },
    {
        "id": 20,
        "category": "Sovereign Decision Layer",
        "prompt": "Explain the three typed decision primitives: SovereignChoice, BushidoNoul, and CouncilScore, and how they enforce policies.",
    },
]

def clean_forward(model, input_ids):
    x = model.wte(input_ids)
    for block in model.h:
        try:
            out = block(x, layer_past=None, use_cache=False,
                        gov_scale=None, token_ids=input_ids)
            x = out[0]
        except Exception:
            pass

    hidden = model.ln_f(x)
    try:
        q1 = model.quillan_finalizer_q1(hidden)
        q2 = model.quillan_finalizer_q2(hidden)
        gate = torch.sigmoid(model.quillan_comm_gate(torch.cat([q1, q2], dim=-1)))
        fused = gate * q1 + (1.0 - gate) * q2
    except Exception:
        fused = hidden

    return model.lm_head(fused)

def generate_answer(model, tok, prompt, max_tokens=100, device="cpu", temperature=0.35, top_p=0.85, rep_penalty=1.25):
    fmt = f"User: {prompt}\n\nAssistant:"
    prompt_ids = tok.encode(fmt).ids
    gen = list(prompt_ids)
    EOS = {0, 50256}
    STOP_STRS = ["User:", "Human:", "<|endoftext|>", "\n\nUser:", "Question:"]

    t0 = time.time()
    with torch.no_grad():
        for _ in range(max_tokens):
            ids = torch.tensor([gen[-512:]], dtype=torch.long, device=device)
            logits = clean_forward(model, ids)
            curr = logits[0, -1, :].float()

            if temperature < 0.05:
                nxt = int(curr.argmax().item())
            else:
                gen_only = gen[len(prompt_ids):]
                for tid in set(gen_only[-32:]):
                    curr[tid] = curr[tid] / rep_penalty if curr[tid] > 0 else curr[tid] * rep_penalty

                sorted_logits, sorted_idx = torch.sort(curr, descending=True)
                probs = F.softmax(sorted_logits / temperature, dim=-1)
                cum = torch.cumsum(probs, dim=-1)
                sorted_logits[cum - probs > top_p] = float("-inf")
                filtered_probs = F.softmax(sorted_logits, dim=-1)

                selected = torch.multinomial(filtered_probs, 1).item()
                nxt = int(sorted_idx[selected].item())

            if nxt in EOS:
                break
            gen.append(nxt)

            text_so_far = tok.decode(gen[len(prompt_ids):])
            if any(s in text_so_far for s in STOP_STRS):
                break

    elapsed = time.time() - t0
    new_ids = gen[len(prompt_ids):]
    text = tok.decode(new_ids).strip()
    for s in STOP_STRS:
        if s in text:
            text = text.split(s)[0].strip()
    return text, elapsed, len(new_ids)

def load_standalone_model(ckpt_path, device):
    print(f"Loading {ckpt_path.name}...")
    d = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg_dict = dict(d["config"])
    cfg_dict["device"] = str(device)
    cfg = QuillanOniConfig(**cfg_dict)
    model = QuillanRoninOni(cfg).to(device)
    model.load_state_dict(d["model_state_dict"], strict=False)
    model.eval()
    step_val = d.get('step', '?')
    loss_val = d.get('loss', 0.0)
    print(f"  [OK] Loaded {ckpt_path.name} (Step={step_val}, Loss={loss_val:.4f})")
    return model, d

def run_suite_for_model(model_tag, ckpt_path, tok, device, max_tokens, temperature):
    print("\n" + "=" * 76)
    print(f"  STANDALONE EVALUATION: {model_tag.upper()} ({ckpt_path.name})")
    print("=" * 76)

    model, d = load_standalone_model(ckpt_path, device)
    results = []

    total_tokens = 0
    total_time = 0.0

    for item in BENCHMARK_20_QUESTIONS:
        qid = item["id"]
        cat = item["category"]
        prompt = item["prompt"]

        print(f"\n[{qid:02d}/20] [{cat}]")
        print(f"  Q: {prompt}")

        resp, el, count = generate_answer(
            model, tok, prompt,
            max_tokens=max_tokens,
            device=device,
            temperature=temperature
        )

        tok_s = count / max(el, 0.01)
        total_tokens += count
        total_time += el

        print(f"  A ({el:.2f}s | {count} tok | {tok_s:.1f} tok/s):")
        print(f"  {resp}")

        results.append({
            "id": qid,
            "category": cat,
            "prompt": prompt,
            "response": resp,
            "elapsed": el,
            "tokens": count,
            "tok_per_sec": tok_s,
        })

    avg_speed = total_tokens / max(total_time, 0.01)
    print("\n" + "-" * 76)
    print(f"  {model_tag.upper()} Summary: {total_tokens} tokens in {total_time:.2f}s (Avg: {avg_speed:.1f} tok/s)")
    print("-" * 76)

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return results, d, avg_speed

def main():
    parser = argparse.ArgumentParser(description="Quillan 20-Question Sovereign Benchmark Suite")
    parser.add_argument("--model", type=str, choices=["6l", "12l", "both"], default="both",
                        help="Which standalone model to evaluate (6l, 12l, or both sequentially)")
    parser.add_argument("--max_tokens", type=int, default=100)
    parser.add_argument("--temp", type=float, default=0.35)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 76)
    print("  👑 QUILLAN-RONIN — 20-QUESTION MASTER BENCHMARK SUITE")
    print(f"  Device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else " (CPU Inference)"))
    print("=" * 76)

    tok = Tokenizer.from_file(str(TOK_PATH))

    results_6l = None
    results_12l = None

    if args.model in ("6l", "both"):
        results_6l, meta_6l, speed_6l = run_suite_for_model("6l", CKPT_6L, tok, device, args.max_tokens, args.temp)

    if args.model in ("12l", "both"):
        results_12l, meta_12l, speed_12l = run_suite_for_model("12l", CKPT_12L, tok, device, args.max_tokens, args.temp)

    # Generate Markdown Report
    report_lines = [
        "# 👑 Quillan-Ronin 20-Question Sovereign Benchmark Report",
        f"**Date**: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Device**: {device}",
        f"**Generation Settings**: Max Tokens: {args.max_tokens} | Temp: {args.temp} | Rep Penalty: 1.25",
        "",
        "## Summary",
    ]
    if results_6l:
        report_lines.append(f"- **Quillan 6L Mini**: Avg Speed = {speed_6l:.1f} tok/s | Checkpoint Loss = {meta_6l.get('loss', 0.0):.4f}")
    if results_12l:
        report_lines.append(f"- **Quillan 12L Main**: Avg Speed = {speed_12l:.1f} tok/s | Checkpoint Loss = {meta_12l.get('loss', 0.0):.4f}")

    report_lines.extend([
        "",
        "## Comparative Results",
        "",
    ])

    for i in range(len(BENCHMARK_20_QUESTIONS)):
        qid = BENCHMARK_20_QUESTIONS[i]["id"]
        cat = BENCHMARK_20_QUESTIONS[i]["category"]
        q_text = BENCHMARK_20_QUESTIONS[i]["prompt"]

        report_lines.append(f"### Q{qid:02d}: {cat}")
        report_lines.append(f"**Prompt**: *{q_text}*\n")

        if results_6l:
            r6 = results_6l[i]
            report_lines.append(f"**Quillan 6L Mini** ({r6['elapsed']:.2f}s | {r6['tokens']} tok | {r6['tok_per_sec']:.1f} tok/s):")
            report_lines.append(f"> {r6['response']}\n")

        if results_12l:
            r12 = results_12l[i]
            report_lines.append(f"**Quillan 12L Main** ({r12['elapsed']:.2f}s | {r12['tokens']} tok | {r12['tok_per_sec']:.1f} tok/s):")
            report_lines.append(f"> {r12['response']}\n")

        report_lines.append("---\n")

    out_file = RESULTS_DIR / f"benchmark_20_questions_{args.model}_{int(time.time())}.md"
    with open(out_file, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print(f"\n[DONE] Master benchmark report written to:\n  {out_file}")

if __name__ == "__main__":
    main()
