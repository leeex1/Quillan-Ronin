#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
👑 QUILLAN MASTER SFT GOLD DATASET COMPILER
===========================================
Compiles all high-density gold instruction datasets into a pristine [N, 256] tensor:
  1. Quillan_Direct_Answers_Gold.jsonl (1,200 direct factual & logic Q&A)
  2. Quillan_Universal_Sovereign_Gold_1000.jsonl (1,100 multi-domain problem solving samples)
  3. sovereign_thinking_gold.jsonl (Chain-of-thought <think> reasoning samples)
  4. .github/agents/*.agent.md (34 Council domain expert personas)

Applies canonical format:
  User: {prompt}\n\nAssistant: {response}<|endoftext|>

Strict prompt masking: labels = -100 on prompt and padding.
"""

import json
import os
import sys
from pathlib import Path
import torch

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

REPO_ROOT = Path(r"C:\02_QUILLAN")
for p in [str(REPO_ROOT), str(REPO_ROOT / "scripts"), str(REPO_ROOT / "03 - Training & Model")]:
    if p not in sys.path:
        sys.path.insert(0, p)

from quillan_bpe_tokenizer import QuillanBPETokenizer

def main():
    print("=" * 72, flush=True)
    print("  👑 COMPILING MASTER SFT GOLD DATASET", flush=True)
    print("=" * 72, flush=True)

    tok = QuillanBPETokenizer()
    print(f"Tokenizer loaded (vocab={tok.vocab_size})", flush=True)

    all_pairs = []

    # 1. 34 Council Personas
    agents_dir = REPO_ROOT / ".github" / "agents"
    if agents_dir.exists():
        for f in sorted(agents_dir.glob("*.agent.md")):
            try:
                content = f.read_text(encoding="utf-8", errors="replace").strip()
                agent_name = f.stem.replace(".agent", "").upper()
                prompt = f"Consult Council Member {agent_name} on sovereign architecture."
                response = f"[{agent_name} Deliberation]\n{content[:600]}"
                all_pairs.append((prompt, response))
            except Exception:
                pass
        print(f"Added {len(all_pairs)} Council Personas", flush=True)

    # 2. JSONL Datasets
    jsonl_files = [
        REPO_ROOT / "training_data" / "Quillan_Direct_Answers_Gold.jsonl",
        REPO_ROOT / "training_data" / "Quillan_Universal_Sovereign_Gold_1000.jsonl",
        REPO_ROOT / "training_data" / "sovereign_thinking_gold.jsonl",
    ]

    for f in jsonl_files:
        if not f.exists():
            print(f"Skipping {f.name} (not found)", flush=True)
            continue
        c = 0
        with open(f, "r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    q = obj.get("question") or obj.get("prompt") or ""
                    a = obj.get("answer") or obj.get("response") or obj.get("final_output") or ""
                    # Clean tags
                    for tag in ["<|im_end|>", "<|endoftext|>", "Question:", "Answer:"]:
                        q = q.replace(tag, "").strip()
                        a = a.replace(tag, "").strip()
                    if q and a:
                        all_pairs.append((q, a))
                        c += 1
                except Exception:
                    continue
        print(f"Added {c} samples from {f.name}", flush=True)

    print(f"\nTotal collected Q&A pairs: {len(all_pairs)}", flush=True)

    seq_len = 256
    pad_id = 50256
    input_ids_list = []
    labels_list = []

    for prompt, resp in all_pairs:
        p_str = f"User: {prompt}\n\nAssistant:"
        r_str = f" {resp}<|endoftext|>"

        p_tokens = tok.encode(p_str)
        r_tokens = tok.encode(r_str)

        if not r_tokens:
            continue

        full_tokens = p_tokens + r_tokens
        if len(full_tokens) > seq_len:
            # Keep prompt, fit as much response as possible
            if len(p_tokens) < seq_len - 16:
                r_keep = seq_len - len(p_tokens)
                full_tokens = p_tokens + r_tokens[:r_keep]
            else:
                p_keep = seq_len // 3
                r_keep = seq_len - p_keep
                full_tokens = p_tokens[:p_keep] + r_tokens[:r_keep]
                p_tokens = p_tokens[:p_keep]

        pad_len = seq_len - len(full_tokens)
        input_ids = full_tokens + [pad_id] * pad_len
        labels = [-100] * len(p_tokens) + full_tokens[len(p_tokens):] + [-100] * pad_len

        if any(l != -100 for l in labels):
            input_ids_list.append(input_ids)
            labels_list.append(labels)

    input_ids_tensor = torch.tensor(input_ids_list, dtype=torch.long)
    labels_tensor = torch.tensor(labels_list, dtype=torch.long)

    out_file = REPO_ROOT / "training_data" / "canonical_standardized" / "quillan_gold_sft_master.pt"
    out_file.parent.mkdir(parents=True, exist_ok=True)

    torch.save({
        "input_ids": input_ids_tensor,
        "labels": labels_tensor,
        "total_samples": len(input_ids_list),
        "seq_len": seq_len,
    }, out_file)

    print(f"\n✅ Successfully compiled Master Gold SFT Dataset: {out_file}", flush=True)
    print(f"Shape: {input_ids_tensor.shape} ({len(input_ids_list)} pristine samples)", flush=True)

if __name__ == "__main__":
    main()
