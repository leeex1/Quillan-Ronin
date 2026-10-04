#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
👑 QUILLAN-RONIN v5.3.1 — OUT-OF-DOMAIN & LONG-FORM GENERATION AUDIT
Tests novel zero-shot reasoning, deep multi-paragraph explanations, and complex conceptual synthesis.
"""

import os
import sys
import time
import torch
from pathlib import Path

SCRIPTS_DIR = Path(r"C:\02_QUILLAN\scripts")
CKPT_PATH = Path(r"C:\02_QUILLAN\checkpoints\production_export\quillan_ronin_v531_sovereign_production.pt")

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from quillan_v10_unrolled_sovereign import QuillanUnrolledSovereign, QuillanUnrolledConfig
from sovereign_inference_engine import SovereignTokenizer

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

def run_ood_audit():
    print("==================================================================", flush=True)
    print("   👑 QUILLAN-RONIN v5.3.1 — OUT-OF-DOMAIN & LONG-FORM AUDIT", flush=True)
    print(f"   Target: {CKPT_PATH.name}", flush=True)
    print("==================================================================\n", flush=True)

    device = torch.device("cpu")
    tokenizer = SovereignTokenizer("gpt2")
    cfg = QuillanUnrolledConfig()

    model = QuillanUnrolledSovereign(cfg).to(device)
    print(f"[*] Loading production model from {CKPT_PATH}...", flush=True)
    data = torch.load(str(CKPT_PATH), map_location=device, weights_only=False)
    sd = data.get("model_state_dict", data.get("state_dict", data))
    model.load_state_dict(sd, strict=False)
    model.eval()
    print("[+] Model loaded and online.\n", flush=True)

    OOD_PROMPTS = [
        (
            "Bio-Cybernetics & Distributed Systems",
            "Question: How do mycorrhizal fungal networks in forests function like decentralized peer-to-peer computer networks? Detail how resources and signals are routed.\nAnswer:\n",
            220
        ),
        (
            "Relativistic Time Dilation & Thought Experiment",
            "Question: Explain the Twin Paradox in special relativity. Why does the traveling twin return younger, and how is the symmetry broken?\nAnswer:\n",
            220
        ),
        (
            "Distributed Consensus & Game Theory",
            "Question: Explain the Byzantine Generals Problem in distributed computing and how consensus algorithms achieve fault tolerance.\nAnswer:\n",
            220
        ),
        (
            "Philosophy of Mind (Qualia & Physicalism)",
            "Question: Explain Frank Jackson's 'Mary's Room' thought experiment and what it argues regarding physicalism and subjective experience (qualia).\nAnswer:\n",
            220
        ),
        (
            "Complex Algorithm Architecture (LRU Cache)",
            "Question: Write a complete Python implementation of an LRU (Least Recently Used) Cache with O(1) get and put operations using a Doubly Linked List and Hash Map.\nAnswer:\n",
            250
        )
    ]

    for idx, (domain, prompt, max_toks) in enumerate(OOD_PROMPTS, 1):
        toks = tokenizer.encode(prompt)
        print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━", flush=True)
        print(f"[{idx}/{len(OOD_PROMPTS)}] DOMAIN: {domain}", flush=True)
        print(f"PROMPT:\n{prompt.strip()}\n", flush=True)
        print("MODEL LONG-FORM SYNTHESIS:", flush=True)
        print("─" * 65, flush=True)

        t0 = time.time()
        out_toks = model.generate(
            toks,
            max_tokens=max_toks,
            temp=0.30,
            top_k=40,
            top_p=0.85,
            frequency_penalty=0.55,
            presence_penalty=0.35,
        )
        elapsed = time.time() - t0

        gen_toks = out_toks[len(toks):]
        gen_text = tokenizer.decode(gen_toks).strip()
        
        # Clean tail metadata if any
        if "<|im_end|>" in gen_text:
            gen_text = gen_text.split("<|im_end|>")[0].strip()
        if "<|endoftext|>" in gen_text:
            gen_text = gen_text.split("<|endoftext|>")[0].strip()

        tps = len(gen_toks) / max(0.001, elapsed)
        print(gen_text, flush=True)
        print("─" * 65, flush=True)
        print(f"Stats: {len(gen_toks)} tokens in {elapsed:.2f}s ({tps:.1f} tok/s)\n", flush=True)

    print("==================================================================", flush=True)
    print("   🏆 OUT-OF-DOMAIN AUDIT COMPLETE", flush=True)
    print("==================================================================", flush=True)

if __name__ == "__main__":
    run_ood_audit()
