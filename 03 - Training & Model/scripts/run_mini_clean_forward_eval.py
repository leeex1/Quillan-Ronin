#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_mini_clean_forward_eval.py — 20-Prompt Benchmark via Pure Transformer Core
Bypasses experimental perturbation layers to evaluate true learned language weights.
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

# NOTE (2026-10-02): the vault was reorganised under "03 - Training & Model".
# This script previously hard-coded the pre-reorg roots, so it raised
# FileNotFoundError on the tokenizer before generating a single token.
REPO_ROOT  = Path(r"C:\02_QUILLAN")
MODEL_DIR  = REPO_ROOT / "03 - Training & Model"


def _resolve(*candidates: Path) -> Path:
    for c in candidates:
        if c.exists():
            return c
    return candidates[0]


SCRIPTS_DIR = _resolve(MODEL_DIR / "scripts", REPO_ROOT / "scripts")
CKPT_DIR    = _resolve(MODEL_DIR / "checkpoints" / "checkpoints_oni",
                       REPO_ROOT / "checkpoints" / "checkpoints_oni")
TOKENIZER   = _resolve(MODEL_DIR / "quillan_bpe_tokenizer_hf" / "tokenizer.json",
                       REPO_ROOT / "quillan_bpe_tokenizer_hf" / "tokenizer.json")

for _p in (SCRIPTS_DIR, MODEL_DIR, REPO_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from quillan_v5_4_oni import QuillanOniConfig, QuillanRoninOni

EVAL_DIR = _resolve(MODEL_DIR / "evaluation_results", REPO_ROOT / "evaluation_results")
EVAL_DIR.mkdir(parents=True, exist_ok=True)

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
     "prompt": "Provide an architectural comparison between Log-Structured Merge-trees (LSM-trees) and B+ Trees in high-throughput databases. Detail write amplification, read latency, and compaction mechanics.", "max_tokens": 200},
    {"id": 2, "domain": "Microarchitectural Security",
     "prompt": "Perform an in-depth security threat analysis of CPU microarchitectural side-channel attacks, focusing on cache-timing attacks (Flush+Reload) and speculative execution vulnerabilities (Spectre). Detail software mitigation techniques.", "max_tokens": 200},
    {"id": 3, "domain": "Hierarchical MoE Design",
     "prompt": "Architect a 34-expert Hierarchical Mixture-of-Experts transformer with BitNet ternary weights. Detail router gating equations, auxiliary load-balancing loss, ST-MoE router z-loss, and dense Memory Attention layers.", "max_tokens": 200},
    {"id": 4, "domain": "Distributed Consensus",
     "prompt": "Examine the Raft consensus protocol: walk through randomized election timers, leader heartbeats, log replication invariants, split-brain prevention via majorities, and handling network partitions.", "max_tokens": 200},
    {"id": 5, "domain": "Statistical Mechanics & Physics",
     "prompt": "Explain the Second Law of Thermodynamics through statistical mechanics and Boltzmann entropy S = k_B * ln(Omega). Resolve the apparent paradox of Maxwell's Demon using information thermodynamics.", "max_tokens": 200},
    {"id": 6, "domain": "Cellular Respiration Pathways",
     "prompt": "Detail the metabolic pathways of aerobic cellular respiration from glycolysis through the Krebs cycle to the electron transport chain complexes I-IV. Detail the proton gradient and ATP yield.", "max_tokens": 200},
    {"id": 7, "domain": "Sovereign AI Ethics Arbitration",
     "prompt": "Construct an ethical arbitration matrix for an autonomous agent facing a multi-stakeholder resource allocation conflict under severe operational scarcity. Synthesize Bushido Gi, deontological constraints, and utilitarian risk minimisation.", "max_tokens": 200},
    {"id": 8, "domain": "Macroeconomic Dynamics",
     "prompt": "Analyze the transmission mechanisms of central bank quantitative easing (QE) and quantitative tightening (QT) on sovereign yield curves, bank lending channels, asset price inflation, and exchange rate dynamics.", "max_tokens": 200},
    {"id": 9, "domain": "Creative Cyberpunk Narrative",
     "prompt": "Write an atmospheric narrative scene: In the subterranean neon-drenched black markets of Neo-Kyoto 2088, an unbonded Ronin operative encounters an illicit neural-splice merchant selling black-box cognitive accelerators.", "max_tokens": 200},
    {"id": 10, "domain": "Memory Attention Theory",
     "prompt": "Provide a rigorous mathematical formulation of Memory Attention as an extension of standard scaled dot-product attention for multi-layer context recurrence. Detail key-value compression, working memory banks, and backward gradient flow.", "max_tokens": 200},
]

def clean_forward(model, input_ids):
    """Faithful full-depth forward pass.

    REWRITTEN 2026-10-02. The previous implementation was:

        out = block(x, layer_past=None, use_cache=False,
                    gov_scale=None, token_ids=input_ids)
        # ...wrapped in `except Exception: pass`

    but UnrolledTransformerBlock.forward is defined as:

        def forward(self, x, layer_past=None, use_cache=False, gov_scale: float = 1.0)

    There is no `token_ids` parameter, and `gov_scale` must be a float (the real
    model passes `self.governor.current_scale`), not None. Every single block
    call therefore raised TypeError, and the bare `except Exception: pass`
    silently discarded it. The result was that NO transformer block ever ran —
    the "model" degenerated to wte -> ln_f -> lm_head, which is exactly the
    unigram word-salad seen in clean_eval_quillan_{6l,12l}_clean_sft.md.

    This version fixes the call signature, supplies a real float gov_scale,
    removes the silent exception handling, mirrors the model's own input
    preprocessing, and ASSERTS that every layer executed.
    """
    x = model.wte(input_ids)

    # Optional persona conditioning
    evo = getattr(model, "agent_evolution", None)
    if evo is not None:
        try:
            x = evo.persona(x, persona_id=None)
        except Exception as exc:
            print(f"  [warn] persona conditioning skipped: {exc}")

    # Dual-Brain Ingestion Gating — mirrors QuillanRoninOni.forward
    if all(hasattr(model, a) for a in ("q1_bridge", "q2_bridge", "ingest_gate")):
        q1 = model.q1_bridge(x)
        q2 = model.q2_bridge(x)
        g_ingest = torch.sigmoid(model.ingest_gate(torch.cat([q1, q2], dim=-1)))
        x = x + 0.05 * (g_ingest * q1 + (1.0 - g_ingest) * q2)

    # Real governor scale — the model's own forward uses this, not None
    gov = getattr(model, "governor", None)
    gov_scale = float(getattr(gov, "current_scale", 1.0)) if gov is not None else 1.0

    layers_run = 0
    for block in model.h:
        out = block(x, None, False, gov_scale)   # -> (x, present, probs, lb, z, ent)
        x = out[0]
        layers_run += 1
    assert layers_run == len(model.h), (
        f"only {layers_run}/{len(model.h)} blocks executed — forward is not faithful"
    )

    hidden = model.ln_f(x)
    if all(hasattr(model, a) for a in
           ("quillan_finalizer_q1", "quillan_finalizer_q2", "quillan_comm_gate")):
        q1 = model.quillan_finalizer_q1(hidden)
        q2 = model.quillan_finalizer_q2(hidden)
        gate = torch.sigmoid(model.quillan_comm_gate(torch.cat([q1, q2], dim=-1)))
        fused = gate * q1 + (1.0 - gate) * q2
    else:
        fused = hidden
    return model.lm_head(fused)

def top_p_filter(logits, top_p=0.92):
    sorted_logits, sorted_idx = torch.sort(logits, descending=True)
    probs = F.softmax(sorted_logits, dim=-1)
    cum = torch.cumsum(probs, dim=-1)
    remove = cum - probs > top_p
    sorted_logits[remove] = float("-inf")
    out = torch.full_like(logits, float("-inf"))
    out.scatter_(0, sorted_idx, sorted_logits)
    return out

def generate_clean(model, tok, prompt, max_tokens=80, device=None,
                   temperature=0.75, top_k=50, top_p=0.92, rep_penalty=1.35):
    fmt = f"User: {prompt}\n\nAssistant:"
    prompt_ids = tok.encode(fmt).ids
    gen = list(prompt_ids)
    EOS_IDS = {0, 50256}
    STOP_STRS = ["<|endoftext|>", "<|im_end|>", "User:"]
    t0 = time.time()

    with torch.no_grad():
        for _ in range(max_tokens):
            inp = torch.tensor([gen[-512:]], dtype=torch.long, device=device)
            logits = clean_forward(model, inp)
            curr = logits[0, -1, :].float()

            gen_only = gen[len(prompt_ids):]
            if gen_only:
                for tid in set(gen_only[-48:]):
                    curr[tid] = curr[tid] / rep_penalty if curr[tid] > 0 else curr[tid] * rep_penalty

            if len(gen_only) >= 4:
                last3 = tuple(gen_only[-3:])
                for i in range(len(gen_only) - 3):
                    if tuple(gen_only[i:i+3]) == last3:
                        curr[gen_only[i+3]] = float("-inf")

            k = min(top_k, curr.size(-1))
            topk_vals, _ = torch.topk(curr, k)
            curr[curr < topk_vals[-1]] = float("-inf")
            curr = top_p_filter(curr, top_p=top_p)

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

def evaluate(ckpt_path=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 72)
    print("  QUILLAN 6L CLEAN FORWARD — 20-PROMPT BENCHMARK")
    print(f"Device: {device}" + (f" | GPU: {torch.cuda.get_device_name(0)}" if device.type == "cuda" else ""))
    print("=" * 72)

    if ckpt_path is None:
        ckpt_path = CKPT_DIR / "quillan_6l_clean_sft.pt"
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    print(f"\nLoading: {ckpt_path.name}")
    d = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg_dict = dict(d["config"])
    cfg_dict["device"] = str(device)
    cfg = QuillanOniConfig(**cfg_dict)

    model = QuillanRoninOni(cfg).to(device)
    # strict=True: a shape/name mismatch must FAIL LOUDLY. The old strict=False
    # silently accepted a partially-loaded model, which produced a "stub" model
    # and fake output with no error at all.
    missing, unexpected = model.load_state_dict(d["model_state_dict"], strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"Checkpoint/model mismatch — refusing to benchmark a partially loaded model.\n"
            f"  missing ({len(missing)}): {list(missing)[:5]}\n"
            f"  unexpected ({len(unexpected)}): {list(unexpected)[:5]}\n"
            f"  checkpoint config: n_layer={cfg.n_layer} hidden={cfg.hidden_dim} "
            f"vocab={cfg.vocab_size}"
        )
    model.eval()
    print(f"  Step: {d.get('step', '?')} | Loss: {d.get('loss', 0.0):.4f} | PPL: {d.get('ppl', 0.0):.2f}")
    print(f"  Arch: n_layer={cfg.n_layer} hidden={cfg.hidden_dim} vocab={cfg.vocab_size} "
          f"experts={cfg.num_experts} router={getattr(cfg, 'router_mode', '?')}")
    print(f"  Blocks that will execute: {len(model.h)} (full depth, no early exit)\n")

    if not TOKENIZER.exists():
        raise FileNotFoundError(f"Tokenizer not found: {TOKENIZER}")
    tok = Tokenizer.from_file(str(TOKENIZER))

    out_file = EVAL_DIR / f"clean_eval_{ckpt_path.stem}.md"
    lines = [f"# Quillan Clean-Forward 20-Prompt Benchmark\n\n**Checkpoint**: {ckpt_path.name} | Step: {d.get('step', '?')} | Loss: {d.get('loss', 0.0):.4f}\n\n"]

    total_tok, total_t = 0, 0.0
    print("=" * 72 + "\n  SHORT-FORM (10 prompts)\n" + "=" * 72)
    for item in SHORT_FORM_PROMPTS:
        qid, dom, prm, mx = item["id"], item["domain"], item["prompt"], item["max_tokens"]
        print(f"\n[SF {qid:02d}] {dom}")
        resp, el, nt = generate_clean(model, tok, prm, mx, device)
        sp = nt / max(el, 0.001)
        total_tok += nt; total_t += el
        print(f"  {el:.1f}s | {nt} tok | {sp:.1f} tok/s")
        print(f"  {resp[:200]}")
        lines.append(f"### SF {qid:02d}: {dom}\n**Prompt**: {prm}\n\n**{el:.1f}s | {nt} tok | {sp:.1f} tok/s**\n```\n{resp}\n```\n\n---\n\n")

    print("\n" + "=" * 72 + "\n  LONG-FORM (10 prompts)\n" + "=" * 72)
    for item in LONG_FORM_PROMPTS:
        qid, dom, prm, mx = item["id"], item["domain"], item["prompt"], item["max_tokens"]
        print(f"\n[LF {qid:02d}] {dom}")
        resp, el, nt = generate_clean(model, tok, prm, mx, device)
        sp = nt / max(el, 0.001)
        total_tok += nt; total_t += el
        print(f"  {el:.1f}s | {nt} tok | {sp:.1f} tok/s")
        print(f"  {resp[:200]}")
        lines.append(f"### LF {qid:02d}: {dom}\n**Prompt**: {prm}\n\n**{el:.1f}s | {nt} tok | {sp:.1f} tok/s**\n```\n{resp}\n```\n\n---\n\n")

    avg_sp = total_tok / max(total_t, 0.001)
    summary = f"\nDONE | Prompts=20 | Tokens={total_tok} | Time={total_t:.1f}s | Speed={avg_sp:.1f} tok/s\n"
    print(summary)
    lines.append(f"## Summary\n```\n{summary}\n```\n")

    with open(out_file, "w", encoding="utf-8") as f:
        f.writelines(lines)
    print(f"Report saved: {out_file}")

if __name__ == "__main__":
    ckpt = sys.argv[1] if len(sys.argv) > 1 else None
    evaluate(ckpt)
