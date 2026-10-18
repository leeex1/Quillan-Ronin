#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_model_speech.py — Test speech, fluency, and correctness of 6L and 12L clean SFT models.
"""
import sys, os, time, math, dataclasses
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

REPO_ROOT = Path(r"C:\02_QUILLAN")
MODEL_DIR = REPO_ROOT / "03 - Training & Model"
SCRIPTS_DIR = MODEL_DIR / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(MODEL_DIR))

from quillan_v5_4_oni import QuillanOniConfig, QuillanRoninOni

TOKENIZER_PATH = MODEL_DIR / "quillan_bpe_tokenizer_hf" / "tokenizer.json"
tok = Tokenizer.from_file(str(TOKENIZER_PATH))

TEST_PROMPTS = [
    "Who are you and what is your architecture?",
    "What is the exact speed of light in a vacuum?",
    "What is the time complexity of binary search?",
    "Why must cryptographic MAC comparisons use constant-time functions?",
    "Explain the Bushido virtue of Gi (Rectitude).",
]

def load_checkpoint(ckpt_path: Path):
    d = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg_dict = dict(d["config"])
    cfg_dict["device"] = "cpu"
    cfg_dict["max_seq_len"] = 512
    valid_keys = {f.name for f in dataclasses.fields(QuillanOniConfig)}
    cfg = QuillanOniConfig(**{k: v for k, v in cfg_dict.items() if k in valid_keys})
    model = QuillanRoninOni(cfg).to("cpu")
    missing, unexpected = model.load_state_dict(d["model_state_dict"], strict=False)
    model.eval()
    return model, cfg, d, missing, unexpected

def generate_text(model, tok, prompt, mode="greedy", max_new=80, temp=0.7, top_p=0.9, rep_penalty=1.25):
    fmt = f"User: {prompt}\n\nAssistant:"
    prompt_ids = tok.encode(fmt).ids
    gen = list(prompt_ids)
    EOS_IDS = {0, 50256}
    
    with torch.no_grad():
        for _ in range(max_new):
            inp = torch.tensor([gen[-512:]], dtype=torch.long, device="cpu")
            out = model(inp, use_cache=False, deliberation=False)
            logits = out[0] if isinstance(out, tuple) else out
            curr = logits[0, -1, :].clone().float()

            if rep_penalty > 1.0:
                gen_only = gen[len(prompt_ids):]
                if gen_only:
                    for tid in set(gen_only[-48:]):
                        if curr[tid] > 0:
                            curr[tid] /= rep_penalty
                        else:
                            curr[tid] *= rep_penalty

            if mode == "greedy":
                nxt = int(curr.argmax().item())
            else:
                curr = curr / max(temp, 1e-4)
                sorted_logits, sorted_idx = torch.sort(curr, descending=True)
                probs = F.softmax(sorted_logits, dim=-1)
                cum = torch.cumsum(probs, dim=-1)
                remove = cum - probs > top_p
                sorted_logits[remove] = float("-inf")
                filtered = torch.full_like(curr, float("-inf"))
                filtered.scatter_(0, sorted_idx, sorted_logits)
                p = F.softmax(filtered, dim=-1)
                nxt = int(torch.multinomial(p, num_samples=1).item())

            if nxt in EOS_IDS:
                break
            gen.append(nxt)

    return tok.decode(gen[len(prompt_ids):]).strip()

def run_tests():
    ckpts = [
        ("6L Mini Model", MODEL_DIR / "checkpoints/checkpoints_oni/quillan_6l_clean_sft.pt"),
        ("12L Main Model", MODEL_DIR / "checkpoints/checkpoints_oni/quillan_12l_clean_sft.pt"),
    ]

    for label, path in ckpts:
        print("=" * 80)
        print(f"TESTING: {label} ({path.name})")
        print("=" * 80)
        if not path.exists():
            print(f"Error: {path} not found!")
            continue

        model, cfg, meta, missing, unexpected = load_checkpoint(path)
        print(f"  Step: {meta.get('step')} | Loss: {meta.get('loss', 0.0):.4f} | Val Loss: {meta.get('val_loss', 0.0):.4f}")
        print(f"  Missing Keys: {len(missing)} | Unexpected Keys: {len(unexpected)}")
        print()

        for idx, prompt in enumerate(TEST_PROMPTS, 1):
            print(f"[{idx}] Prompt: \"{prompt}\"")
            
            # 1. Greedy decoding
            t0 = time.time()
            resp_greedy = generate_text(model, tok, prompt, mode="greedy", max_new=75)
            dt_g = time.time() - t0
            print(f"  [Greedy] ({dt_g:.1f}s):")
            print(f"    {resp_greedy}")

            # 2. Sampled decoding (temp=0.7, top-p=0.9, rep_penalty=1.25)
            t0 = time.time()
            resp_sample = generate_text(model, tok, prompt, mode="sample", max_new=75, temp=0.7, top_p=0.9)
            dt_s = time.time() - t0
            print(f"  [Sampled temp=0.7] ({dt_s:.1f}s):")
            print(f"    {resp_sample}")
            print("-" * 60)
        print("\n")

if __name__ == "__main__":
    run_tests()
