#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_dual_models.py — Side-by-Side Inference Test Harness
=========================================================
Loads and compares generations from:
  1. Quillan 6L Mini Model (quillan_6l_clean_sft.pt)
  2. Quillan 12L Main Model (quillan_12l_clean_sft.pt)

Supports both a default suite of test prompts and custom user input.
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
sys.path.insert(0, str(REPO_ROOT / "03 - Training & Model" / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from quillan_v5_4_oni import QuillanOniConfig, QuillanRoninOni

def resolve_path(rel_p):
    p1 = REPO_ROOT / "03 - Training & Model" / rel_p
    if p1.exists():
        return p1
    return REPO_ROOT / rel_p

CKPT_6L  = resolve_path("checkpoints/checkpoints_oni/quillan_6l_clean_sft.pt")
CKPT_12L = resolve_path("checkpoints/checkpoints_oni/quillan_12l_clean_sft.pt")
TOK_PATH = resolve_path("quillan_bpe_tokenizer_hf/tokenizer.json")


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


def generate(model, tok, prompt, max_tokens=100, device="cpu", temperature=0.7, top_p=0.9, rep_penalty=1.25):
    fmt = f"User: {prompt}\n\nAssistant:"
    prompt_ids = tok.encode(fmt).ids
    gen = list(prompt_ids)
    EOS = {0, 50256}
    STOP_STRS = ["User:", "Human:", "<|endoftext|>", "\n\nUser:"]

    t0 = time.time()
    with torch.no_grad():
        for _ in range(max_tokens):
            ids = torch.tensor([gen[-512:]], dtype=torch.long, device=device)
            logits = clean_forward(model, ids)
            curr = logits[0, -1, :].float()

            if temperature < 0.05:
                # Greedy
                nxt = int(curr.argmax().item())
            else:
                # Repetition penalty
                gen_only = gen[len(prompt_ids):]
                for tid in set(gen_only[-32:]):
                    curr[tid] = curr[tid] / rep_penalty if curr[tid] > 0 else curr[tid] * rep_penalty

                # Top-p
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


def load_model(ckpt_path, device):
    print(f"Loading {ckpt_path.name}...")
    d = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg_dict = dict(d["config"])
    cfg_dict["device"] = str(device)
    cfg = QuillanOniConfig(**cfg_dict)
    model = QuillanRoninOni(cfg).to(device)
    model.load_state_dict(d["model_state_dict"], strict=False)
    model.eval()
    print(f"  Loaded! (Step={d.get('step', '?')}, Loss={d.get('loss', 0.0):.4f})")
    return model, d


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default=None, help="Custom prompt to test")
    parser.add_argument("--model", type=str, choices=["6l", "12l", "both"], default="both")
    parser.add_argument("--max_tokens", type=int, default=80)
    parser.add_argument("--temp", type=float, default=0.4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 72)
    print("  QUILLAN DUAL-MODEL TEST HARNESS")
    print(f"  Device: {device}" + (f" | {torch.cuda.get_device_name(0)}" if device.type=="cuda" else ""))
    print("=" * 72)

    tok = Tokenizer.from_file(str(TOK_PATH))

    TEST_PROMPTS = [
        "What is the exact speed of light in a vacuum?",
        "Who are you and what is your architecture?",
        "What is the time complexity of Dijkstra's algorithm with a min-heap?",
        "Explain the Bushido virtue of Gi (Rectitude) in AI safety.",
    ]

    prompts = [args.prompt] if args.prompt else TEST_PROMPTS

    # Test 6L
    if args.model in ("6l", "both"):
        print("\n" + "=" * 72)
        print("  MODEL 1: QUILLAN 6-LAYER MINI (0.6B)")
        print("=" * 72)
        m6l, d6l = load_model(CKPT_6L, device)
        for i, p in enumerate(prompts, 1):
            print(f"\n[Prompt {i}] User: {p}")
            resp, el, tok_count = generate(m6l, tok, p, max_tokens=args.max_tokens, device=device, temperature=args.temp)
            print(f"Assistant ({el:.2f}s, {tok_count} tok, {tok_count/max(el,0.01):.1f} tok/s):")
            print(f"{resp}")
        del m6l
        torch.cuda.empty_cache()

    # Test 12L
    if args.model in ("12l", "both"):
        print("\n" + "=" * 72)
        print("  MODEL 2: QUILLAN 12-LAYER MAIN (1.0B)")
        print("=" * 72)
        m12l, d12l = load_model(CKPT_12L, device)
        for i, p in enumerate(prompts, 1):
            print(f"\n[Prompt {i}] User: {p}")
            resp, el, tok_count = generate(m12l, tok, p, max_tokens=args.max_tokens, device=device, temperature=args.temp)
            print(f"Assistant ({el:.2f}s, {tok_count} tok, {tok_count/max(el,0.01):.1f} tok/s):")
            print(f"{resp}")
        del m12l
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
