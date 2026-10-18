#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
👑 QUILLAN PURE GOLD DATASET PACKER
===================================
Aggregates pure gold human/model responses from:
  - Quillan_Clean_Reasoning_Gold_Dataset.jsonl
  - Quillan_Direct_Answers_Gold.jsonl
  - Quillan_Ronin_v5.3.1_Samurai_Training_Seed_Dataset.jsonl
  - Quillan_Explanatory_Prose_Dataset.jsonl
  - Quillan_General_Knowledge_Dataset.jsonl

Applies canonical chat formatting without any artificial boilerplate:
  <|start|>
  <|user|>
  {prompt}
  <|assistant|>
  {clean_response}<|im_end|>

Masks prompt tokens with -100 so models train exclusively on pure gold answers.
"""

import json
import os
import sys
from pathlib import Path
import torch

REPO_ROOT = Path(r"C:\02_QUILLAN")
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "03 - Training & Model"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "09 - Projects" / "projects" / "oni"))

from quillan_bpe_tokenizer import QuillanBPETokenizer

def pack_dataset():
    tok = QuillanBPETokenizer()
    print(f"Tokenizer loaded (vocab={tok.vocab_size})")

    files = [
        REPO_ROOT / "training_data" / "Quillan_Clean_Reasoning_Gold_Dataset.jsonl",
        REPO_ROOT / "training_data" / "Quillan_Direct_Answers_Gold.jsonl",
        REPO_ROOT / "training_data" / "Quillan_Ronin_v5.3.1_Samurai_Training_Seed_Dataset.jsonl",
        REPO_ROOT / "training_data" / "Quillan_Explanatory_Prose_Dataset.jsonl",
        REPO_ROOT / "training_data" / "Quillan_General_Knowledge_Dataset.jsonl",
    ]

    all_samples = []
    for f in files:
        if not f.exists():
            print(f"Skipping missing file: {f}")
            continue
        count = 0
        with open(f, "r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue

                prompt = obj.get("question") or obj.get("prompt") or ""
                resp = obj.get("response") or obj.get("final_output") or obj.get("answer") or ""

                prompt = prompt.replace("<|im_end|>", "").strip()
                resp = resp.replace("<|im_end|>", "").strip()

                if not prompt or not resp:
                    continue

                all_samples.append((prompt, resp))
                count += 1
        print(f"Loaded {count} samples from {f.name}")

    print(f"\nTotal raw gold samples: {len(all_samples)}")

    seq_len = 256
    pad_id = 50256
    input_ids_list = []
    labels_list = []

    for prompt, resp in all_samples:
        prompt_formatted = f"<|start|>\n<|user|>\n{prompt}\n<|assistant|>\n"
        resp_formatted = f"{resp}<|im_end|>\n"

        p_tokens = tok.encode(prompt_formatted)
        r_tokens = tok.encode(resp_formatted)

        if not r_tokens:
            continue

        full_tokens = p_tokens + r_tokens
        if len(full_tokens) > seq_len:
            # truncate prompt if needed to fit at least 64 tokens of response
            if len(r_tokens) < seq_len:
                p_keep = seq_len - len(r_tokens)
                p_tokens = p_tokens[:p_keep]
                full_tokens = p_tokens + r_tokens
            else:
                full_tokens = full_tokens[:seq_len]

        pad_len = seq_len - len(full_tokens)
        input_ids = full_tokens + [pad_id] * pad_len

        # Label: -100 for prompt and padding, target tokens for response
        labels = [-100] * len(p_tokens) + full_tokens[len(p_tokens):] + [-100] * pad_len

        # Ensure at least one label is valid
        if any(l != -100 for l in labels):
            input_ids_list.append(input_ids)
            labels_list.append(labels)

    input_ids_tensor = torch.tensor(input_ids_list, dtype=torch.long)
    labels_tensor = torch.tensor(labels_list, dtype=torch.long)

    out_file = REPO_ROOT / "training_data" / "canonical_standardized" / "quillan_gold_alignment_clean.pt"
    out_file.parent.mkdir(parents=True, exist_ok=True)

    torch.save({
        "input_ids": input_ids_tensor,
        "labels": labels_tensor,
        "total_samples": len(input_ids_list),
        "seq_len": seq_len,
    }, out_file)

    print(f"\nSuccessfully saved clean gold dataset: {out_file}")
    print(f"Tensor shape: {input_ids_tensor.shape} (Total valid samples: {len(input_ids_list)})")

if __name__ == "__main__":
    pack_dataset()
