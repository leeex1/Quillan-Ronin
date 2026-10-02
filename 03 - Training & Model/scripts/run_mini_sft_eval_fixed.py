#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QUILLAN-RONIN v5.4.0-ONI — FIXED 20-PROMPT BENCHMARK EVALUATION
================================================================
Run with:  C:\02_QUILLAN\venv_oni_cu126\Scripts\python.exe run_mini_sft_eval_fixed.py

Key fixes vs original run_mini_sft_20q_eval.py:
  1. No use_cache in autoregressive loop — avoids KV-cache corruption with
     experimental modules (mamba, swarm, world_model, etc.).
  2. deliberation=False per token — skips E_ICE/Langevin on T=1 states
     which produce corrupted logit distributions and word salad.
  3. Nucleus top-p=0.92 + top-k=50 for smooth diverse output.
  4. Temperature 0.75 (was 0.65).
  5. 4-gram repetition ban + rep_penalty=1.35 (was 1.2).
"""

import functools, os, sys, time
from datetime import datetime
from pathlib import Path

print = functools.partial(print, flush=True)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

REPO_ROOT  = Path(r"C:\02_QUILLAN")
SCRIPTS_DIR = REPO_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from quillan_v5_4_oni import QuillanOniConfig, QuillanRoninOni

EVAL_DIR = REPO_ROOT / "evaluation_results"
EVAL_DIR.mkdir(parents=True, exist_ok=True)

# ─── Prompts ──────────────────────────────────────────────────────────────────

SHORT_FORM_PROMPTS = [
    {"id": 1, "domain": "Computer Science & Algorithms",
     "prompt": "What is the time and space complexity of Dijkstra's algorithm with a min-heap?", "max_tokens": 80},
    {"id": 2, "domain": "Quillan Identity & Persona",
     "prompt": "Who are you, what is your architecture, and what is the role of Council Member C2-VIR?", "max_tokens": 80},
    {"id": 3, "domain": "Applied Cryptography",
     "prompt": "Why must cryptographic MAC and signature comparisons use constant-time memory functions?", "max_tokens": 80},
    {"id": 4, "domain": "Thermodynamics & Information",
     "prompt": "State Landauer's Principle and its theoretical energy bound for erasing one bit of information at 300K.", "max_tokens": 80},
    {"id": 5, "domain": "Bushido & AI Safety",
     "prompt": "How does the Bushido virtue of Gi (Rectitude) govern refusal boundaries in autonomous agents?", "max_tokens": 80},
    {"id": 6, "domain": "Distributed Systems",
     "prompt": "Explain the CAP theorem and the fundamental trade-off between Consistency and Availability.", "max_tokens": 80},
    {"id": 7, "domain": "Mathematics & Linear Algebra",
     "prompt": "What is the geometric interpretation of Singular Value Decomposition (SVD) for a real matrix?", "max_tokens": 80},
    {"id": 8, "domain": "Neural Architectures (MoE)",
     "prompt": "What is the function of the Straight-Through Estimator (STE) in BitNet 1.58b ternary quantization?", "max_tokens": 80},
    {"id": 9, "domain": "Biochemistry",
     "prompt": "Explain the function of the F1F0 ATP synthase motor during mitochondrial oxidative phosphorylation.", "max_tokens": 80},
    {"id": 10, "domain": "Economics & Game Theory",
     "prompt": "Define a Nash Equilibrium and explain why the Prisoner's Dilemma reaches a Pareto sub-optimal outcome.", "max_tokens": 80},
]

LONG_FORM_PROMPTS = [
    {"id": 1, "domain": "Database Engineering",
     "prompt": "Provide an architectural comparison between Log-Structured Merge-trees (LSM-trees) and B+ Trees in high-throughput databases. Detail write amplification, read latency, and compaction mechanics.", "max_tokens": 220},
    {"id": 2, "domain": "Microarchitectural Security",
     "prompt": "Perform an in-depth security threat analysis of CPU microarchitectural side-channel attacks, focusing on cache-timing attacks (Flush+Reload) and speculative execution vulnerabilities (Spectre). Detail software mitigation techniques.", "max_tokens": 220},
    {"id": 3, "domain": "Hierarchical MoE Design",
     "prompt": "Architect a 34-expert Hierarchical Mixture-of-Experts transformer with BitNet ternary weights. Detail router gating equations, auxiliary load-balancing loss, ST-MoE router z-loss, and dense Memory Attention layers.", "max_tokens": 220},
    {"id": 4, "domain": "Distributed Consensus",
     "prompt": "Examine the Raft consensus protocol: walk through randomized election timers, leader heartbeats, log replication invariants, split-brain prevention via majorities, and handling network partitions.", "max_tokens": 220},
    {"id": 5, "domain": "Statistical Mechanics & Physics",
     "prompt": "Explain the Second Law of Thermodynamics through statistical mechanics and Boltzmann entropy S = k_B * ln(Omega). Resolve the apparent paradox of Maxwell's Demon using information thermodynamics.", "max_tokens": 220},
    {"id": 6, "domain": "Cellular Respiration Pathways",
     "prompt": "Detail the metabolic pathways of aerobic cellular respiration from glycolysis through the Krebs cycle to the electron transport chain complexes I-IV. Detail the proton gradient and ATP yield.", "max_tokens": 220},
    {"id": 7, "domain": "Sovereign AI Ethics Arbitration",
     "prompt": "Compare Deontological ethics, Utilitarian consequentialism, and Bushido virtue ethics when designing sovereign AI safety refusal layers (C2-VIR / E_ICE). How should conflicting obligations between user obedience and harm prevention be arbitrated?", "max_tokens": 220},
    {"id": 8, "domain": "Macroeconomic Dynamics",
     "prompt": "Analyze the macroeconomic dynamics of stagflation: how supply-side resource shocks interact with monetary expansion and cost-push inflation. Compare Keynesian aggregate demand management and Monetarist rate-hiking responses.", "max_tokens": 220},
    {"id": 9, "domain": "Creative Cyberpunk Narrative",
     "prompt": "Craft an evocative cinematic opening scene depicting a cyber-enhanced masterless ronin navigating the dark rain-drenched subterranean market of Neo-Tokyo. Include sensory imagery, flickering holographic ads, and the thermal hum of neural processors.", "max_tokens": 220},
    {"id": 10, "domain": "Memory Attention Theory",
     "prompt": "Explain the theoretical foundations of Memory Attention. Describe how eliminating the W_V projection and substituting token-indexed layer memory reduces KV-cache footprint while maintaining expressive capacity.", "max_tokens": 220},
]

# ─── Inference ────────────────────────────────────────────────────────────────

def top_p_filter(logits, top_p=0.92):
    sorted_logits, sorted_idx = torch.sort(logits, descending=True)
    probs = F.softmax(sorted_logits, dim=-1)
    cum_probs = torch.cumsum(probs, dim=-1)
    remove = cum_probs - probs > top_p
    sorted_logits[remove] = float("-inf")
    out = torch.full_like(logits, float("-inf"))
    out.scatter_(0, sorted_idx, sorted_logits)
    return out


def generate_tokens(model, tok, prompt, max_tokens, device,
                    temperature=0.75, top_k=50, top_p=0.92, rep_penalty=1.35):
    formatted = f"User: {prompt}\n\nAssistant:"
    enc = tok.encode(formatted)
    prompt_ids = enc.ids
    gen = list(prompt_ids)

    EOS_IDS = {0, 50256}
    STOP_STRS = ["<|endoftext|>", "<|im_end|>", "<|user|>", "User:"]

    t0 = time.time()
    with torch.no_grad():
        for _ in range(max_tokens):
            input_ids = torch.tensor([gen], dtype=torch.long, device=device)

            # Full sequence forward — no cache, no deliberation
            # deliberation=False prevents E_ICE/Langevin corruption on T=1 slices
            out = model(input_ids, use_cache=False, deliberation=False)
            logits = out[0] if isinstance(out, tuple) else out
            curr = logits[0, -1, :].float()

            # Repetition penalty on last 48 generated tokens
            gen_only = gen[len(prompt_ids):]
            if gen_only:
                for tid in set(gen_only[-48:]):
                    if curr[tid] > 0:
                        curr[tid] /= rep_penalty
                    else:
                        curr[tid] *= rep_penalty

            # 4-gram ban
            if len(gen_only) >= 4:
                last3 = tuple(gen_only[-3:])
                for i in range(len(gen_only) - 3):
                    if tuple(gen_only[i:i+3]) == last3:
                        curr[gen_only[i+3]] = float("-inf")

            # Top-K filter
            k = min(top_k, curr.size(-1))
            topk_vals, _ = torch.topk(curr, k)
            curr[curr < topk_vals[-1]] = float("-inf")

            # Top-P filter
            curr = top_p_filter(curr, top_p=top_p)

            # Sample
            probs = F.softmax(curr / max(temperature, 0.05), dim=-1)
            if not probs.isfinite().any() or probs.sum() < 1e-8:
                probs = torch.ones_like(curr) / curr.size(-1)

            next_tok = int(torch.multinomial(probs, 1).item())
            if next_tok in EOS_IDS:
                break

            gen.append(next_tok)

            partial = tok.decode(gen[len(prompt_ids):]).strip()
            if any(s in partial for s in STOP_STRS):
                break

    elapsed = time.time() - t0
    new_ids = gen[len(prompt_ids):]
    text = tok.decode(new_ids).strip()
    for s in STOP_STRS:
        if s in text:
            text = text.split(s)[0].strip()
    return text, elapsed, len(new_ids)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 72)
    print("  QUILLAN 6L SFT FIXED — 20-PROMPT BENCHMARK")
    print("=" * 72)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}" + (f" | GPU: {torch.cuda.get_device_name(0)}" if device.type == "cuda" else ""))

    ckpt_path = REPO_ROOT / "checkpoints" / "checkpoints_oni" / "quillan_6l_clean_sft.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Not found: {ckpt_path}")

    print(f"\nLoading {ckpt_path.name} ({ckpt_path.stat().st_size / 1e9:.2f} GB)...")
    t0 = time.time()
    data = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    _loss = data.get('loss') or data.get('val_loss', 0.0)
    _step = data.get('step', '?')
    _ppl  = data.get('perplexity', '?')
    print(f"  Loaded in {time.time()-t0:.1f}s  |  loss={_loss:.4f}  PPL={_ppl}  step={_step}")

    cfg_dict = dict(data['config'])
    cfg_dict['device'] = str(device)
    cfg = QuillanOniConfig(**cfg_dict)

    model = QuillanRoninOni(cfg).to(device)
    missing, _ = model.load_state_dict(data['model_state_dict'], strict=False)
    if missing:
        print(f"  [WARN] {len(missing)} missing keys (paper module stubs — expected)")
    model.eval()
    model.fold_memory_attention_weights()
    print(f"  Memory attention pre-folded. Params: {sum(p.numel() for p in model.parameters()):,}\n")

    # Tokenizer
    tok_path = REPO_ROOT / "quillan_bpe_tokenizer_hf" / "tokenizer.json"
    if not tok_path.exists():
        tok_path = REPO_ROOT / "03 - Training & Model" / "quillan_bpe_tokenizer_hf" / "tokenizer.json"
    tok = Tokenizer.from_file(str(tok_path))
    print(f"Tokenizer: vocab={tok.get_vocab_size()}\n")

    log_file = EVAL_DIR / "mini_6l_sft_FIXED_evaluation.log"
    md_file  = EVAL_DIR / "mini_6l_sft_FIXED_evaluation.md"
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ppl_str = f"{_ppl:.1f}" if isinstance(_ppl, (int, float)) else str(_ppl)
    hdr = (
        f"{'='*72}\nQUILLAN 6L CLEAN SFT — 20-PROMPT BENCHMARK\n{'='*72}\n"
        f"Ckpt : {ckpt_path.name}  loss={_loss:.4f}  PPL={ppl_str}  step={_step}\n"
        f"Time : {ts}  Device : {device}\n"
        f"Samp : temp=0.75 top_k=50 top_p=0.92 rep_penalty=1.35  deliberation=OFF\n"
        f"{'='*72}\n\n"
    )
    with open(log_file, "w", encoding="utf-8") as f: f.write(hdr)
    with open(md_file,  "w", encoding="utf-8") as f:
        f.write(f"# Quillan-Ronin 6L SFT Fixed Evaluation\n\n```\n{hdr}```\n\n")

    total_tok, total_t = 0, 0.0

    # Short-form
    print("="*72 + "\n  SHORT-FORM (10 prompts)\n" + "="*72)
    for item in SHORT_FORM_PROMPTS:
        qid, dom, prm, mx = item["id"], item["domain"], item["prompt"], item["max_tokens"]
        print(f"\n[SF {qid:02d}] {dom}")
        resp, el, nt = generate_tokens(model, tok, prm, mx, device)
        sp = nt / max(el, 0.001)
        total_tok += nt; total_t += el
        print(f"  {el:.1f}s | {nt} tok | {sp:.1f} tok/s")
        print(f"  {resp[:200]}")
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"[SF {qid:02d}] {dom}\n{el:.2f}s | {nt}tok | {sp:.1f}tok/s\nP: {prm}\nR: {resp}\n{'-'*72}\n\n")
        with open(md_file, "a", encoding="utf-8") as f:
            f.write(f"### Short {qid}: {dom}\n- **{el:.2f}s** | **{nt}tok** | **{sp:.1f}tok/s**\n\n"
                    f"**Prompt**: {prm}\n\n**Response**:\n```\n{resp}\n```\n\n---\n\n")

    # Long-form
    print("\n" + "="*72 + "\n  LONG-FORM (10 prompts)\n" + "="*72)
    for item in LONG_FORM_PROMPTS:
        qid, dom, prm, mx = item["id"], item["domain"], item["prompt"], item["max_tokens"]
        print(f"\n[LF {qid:02d}] {dom}")
        resp, el, nt = generate_tokens(model, tok, prm, mx, device)
        sp = nt / max(el, 0.001)
        total_tok += nt; total_t += el
        print(f"  {el:.1f}s | {nt} tok | {sp:.1f} tok/s")
        print(f"  {resp[:200]}")
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"[LF {qid:02d}] {dom}\n{el:.2f}s | {nt}tok | {sp:.1f}tok/s\nP: {prm}\nR: {resp}\n{'-'*72}\n\n")
        with open(md_file, "a", encoding="utf-8") as f:
            f.write(f"### Long {qid}: {dom}\n- **{el:.2f}s** | **{nt}tok** | **{sp:.1f}tok/s**\n\n"
                    f"**Prompt**: {prm}\n\n**Response**:\n```\n{resp}\n```\n\n---\n\n")

    avg = total_tok / max(total_t, 0.001)
    summary = (f"\n{'='*72}\nDONE  |  Prompts=20  |  Tokens={total_tok}  |  Time={total_t:.1f}s  |  Avg={avg:.1f}tok/s\n"
               f"Log: {log_file}\nReport: {md_file}\n{'='*72}\n")
    print(summary)
    with open(log_file, "a", encoding="utf-8") as f: f.write(summary)
    with open(md_file,  "a", encoding="utf-8") as f: f.write(f"\n## Summary\n```\n{summary}```\n")


if __name__ == "__main__":
    main()



