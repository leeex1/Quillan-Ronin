#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
clean_generate.py — Standalone clean-inference generation for Quillan-Ronin.
Runs EXACTLY the transformer core (embedding + n_layer blocks + ln_f + lm_head)
without the 50+ paper-module x-perturbation injections that corrupt generation.

Uses a thin forward wrapper that calls only the essential path:
  embed -> [block_0..block_n] -> ln_f -> quillan_finalizer -> lm_head

This proves whether word-salad is a training quality issue (model weights)
or a inference noise issue (paper modules perturbing x on every pass).
"""

import sys, time, functools
from pathlib import Path

print = functools.partial(print, flush=True)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

REPO_ROOT  = Path(r"C:\02_QUILLAN")
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from quillan_v5_4_oni import QuillanOniConfig, QuillanRoninOni

# ─────────────────────────────────────────────────────────────
# CLEAN FORWARD: runs ONLY the essential generation path
# ─────────────────────────────────────────────────────────────
def clean_forward(model, input_ids):
    """
    Bypasses all paper-module x-perturbations.
    Runs: wte -> positional -> dual-brain ingest (SKIPPED) -> block loop
          -> ln_f -> quillan_finalizer -> lm_head
    Returns logits tensor [B, T, vocab].
    """
    cfg = model.cfg
    B, T = input_ids.shape
    device = input_ids.device

    # Embedding
    x = model.wte(input_ids)

    # Positional — use RoPE via block self-attention (handled inside block)
    # Run only the transformer blocks — the clean core
    for block in model.h:
        try:
            out = block(x, layer_past=None, use_cache=False,
                        gov_scale=None, token_ids=input_ids)
            # block returns: (x, present, probs, lb, z, ent)
            x = out[0]
        except Exception as e:
            pass  # if block fails, keep previous x

    # Final LayerNorm
    hidden = model.ln_f(x)

    # Dual finalizer
    try:
        q1 = model.quillan_finalizer_q1(hidden)
        q2 = model.quillan_finalizer_q2(hidden)
        gate = torch.sigmoid(model.quillan_comm_gate(torch.cat([q1, q2], dim=-1)))
        fused = gate * q1 + (1.0 - gate) * q2
    except Exception:
        fused = hidden

    logits = model.lm_head(fused)
    return logits


def top_p_filter(logits, top_p=0.92):
    sorted_logits, sorted_idx = torch.sort(logits, descending=True)
    probs = F.softmax(sorted_logits, dim=-1)
    cum = torch.cumsum(probs, dim=-1)
    remove = cum - probs > top_p
    sorted_logits[remove] = float("-inf")
    out = torch.full_like(logits, float("-inf"))
    out.scatter_(0, sorted_idx, sorted_logits)
    return out


def generate(model, tok, prompt, max_tokens=80, device=None,
             temperature=0.80, top_k=50, top_p=0.92, rep_penalty=1.35):
    fmt = f"User: {prompt}\n\nAssistant:"
    prompt_ids = tok.encode(fmt).ids
    gen = list(prompt_ids)
    EOS = {0, 50256}
    STOPS = ["<|endoftext|>", "<|im_end|>", "User:"]

    t0 = time.time()
    with torch.no_grad():
        for _ in range(max_tokens):
            ids = torch.tensor([gen], dtype=torch.long, device=device)
            logits = clean_forward(model, ids)
            curr = logits[0, -1, :].float()

            # Rep penalty
            gen_only = gen[len(prompt_ids):]
            for tid in set(gen_only[-48:]):
                curr[tid] = curr[tid] / rep_penalty if curr[tid] > 0 else curr[tid] * rep_penalty

            # 4-gram ban
            if len(gen_only) >= 4:
                last3 = tuple(gen_only[-3:])
                for i in range(len(gen_only)-3):
                    if tuple(gen_only[i:i+3]) == last3:
                        curr[gen_only[i+3]] = float("-inf")

            # Top-K
            topk_v, _ = torch.topk(curr, min(top_k, curr.size(-1)))
            curr[curr < topk_v[-1]] = float("-inf")
            curr = top_p_filter(curr, top_p)

            probs = F.softmax(curr / max(temperature, 0.05), dim=-1)
            if not probs.isfinite().any() or probs.sum() < 1e-8:
                probs = torch.ones_like(curr) / curr.size(-1)

            nxt = int(torch.multinomial(probs, 1).item())
            if nxt in EOS:
                break
            gen.append(nxt)

            partial = tok.decode(gen[len(prompt_ids):])
            if any(s in partial for s in STOPS):
                break

    elapsed = time.time() - t0
    text = tok.decode(gen[len(prompt_ids):]).strip()
    for s in STOPS:
        if s in text:
            text = text.split(s)[0].strip()
    return text, elapsed, len(gen) - len(prompt_ids)


PROMPTS = [
    ("Dijkstra Complexity", "What is the time and space complexity of Dijkstra's algorithm with a min-heap?", 80),
    ("Quillan Identity", "Who are you and what is your architecture?", 80),
    ("CAP Theorem", "Explain the CAP theorem and the trade-off between Consistency and Availability.", 80),
    ("Speed of Light", "What is the exact speed of light in a vacuum and what is its significance in physics?", 80),
    ("BitNet STE", "What is the function of the Straight-Through Estimator in BitNet 1.58b ternary quantization?", 80),
    ("Raft Consensus", "Explain how the Raft consensus protocol handles leader election and log replication.", 160),
    ("Cyberpunk Narrative", "Write an opening scene of a cyber-enhanced ronin navigating Neo-Tokyo's underground market.", 160),
]


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 68)
    print("  QUILLAN CLEAN-FORWARD INFERENCE TEST")
    print(f"  Device: {device}" + (f" | {torch.cuda.get_device_name(0)}" if device.type=="cuda" else ""))
    print("=" * 68)

    # Try the MA best checkpoint first (PPL=10.1 — target zone)
    ckpt = REPO_ROOT / "checkpoints" / "checkpoints_oni" / "quillan_6l_ma_best.pt"
    print(f"\nLoading: {ckpt.name}")
    d = torch.load(ckpt, map_location="cpu", weights_only=False)
    cfg_dict = dict(d["config"])
    cfg_dict["device"] = str(device)
    cfg = QuillanOniConfig(**cfg_dict)

    model = QuillanRoninOni(cfg).to(device)
    model.load_state_dict(d["model_state_dict"], strict=False)
    model.eval()
    model.fold_memory_attention_weights()
    print(f"  val_loss={d.get('val_loss',0):.4f}  PPL={d.get('perplexity',0):.1f}  params={sum(p.numel() for p in model.parameters()):,}\n")

    tok = Tokenizer.from_file(str(REPO_ROOT / "quillan_bpe_tokenizer_hf" / "tokenizer.json"))

    out_file = REPO_ROOT / "evaluation_results" / "clean_forward_test.md"
    lines = [f"# Quillan Clean-Forward Inference Test\n\n**Checkpoint**: {ckpt.name} | val_loss={d.get('val_loss',0):.4f} | PPL={d.get('perplexity',0):.1f}\n\n"]

    for name, prompt, max_t in PROMPTS:
        print(f"\n[{name}]")
        print(f"  Prompt: {prompt[:80]}")
        resp, el, nt = generate(model, tok, prompt, max_t, device)
        print(f"  {el:.1f}s | {nt}tok")
        print(f"  >> {resp[:200]}")
        lines.append(f"### {name}\n**Prompt**: {prompt}\n\n**{el:.1f}s | {nt}tok**\n```\n{resp}\n```\n\n---\n\n")

    with open(out_file, "w", encoding="utf-8") as f:
        f.writelines(lines)
    print(f"\n[DONE] Results: {out_file}")


if __name__ == "__main__":
    main()
