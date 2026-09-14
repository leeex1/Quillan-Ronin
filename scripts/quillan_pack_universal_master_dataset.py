#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
👑 QUILLAN UNIVERSAL MASTER GOLD DATASET COMPILER (v2 - Full 34-Expert & Multi-Domain)
====================================================================================
Aggregates pure gold multi-domain reasoning & instructions from:
  1. All 34 Council Expert files in training_data/experts_34/*.jsonl (~9,900 samples)
  2. Quillan_Universal_100_Percent_Master_Gold.jsonl (30,099 deep multi-domain samples)
  3. Quillan_Clean_Reasoning_Gold_Dataset.jsonl (2,279 deep reasoning traces)
  4. Quillan_Direct_Answers_Gold.jsonl (1,200 short-form precision answers)
  5. Quillan_Universal_Sovereign_Gold_1000.jsonl (1,100 sovereign capability samples)
  6. quillan_science_absolute.jsonl & quillan_science_additional.jsonl (physics, math, chemistry)
  7. Quillan_General_Knowledge_Dataset.jsonl & Quillan_Explanatory_Prose_Dataset.jsonl

Canonical chat format:
  <|start|>
  <|user|>
  {prompt}
  <|assistant|>
  {response}<|im_end|>

Labels:
  - Prompt tokens masked with -100 (zero gradient penalty)
  - Assistant response tokens are the true targets
  - Padding tokens masked with -100
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import List, Tuple
import torch

REPO_ROOT = Path(r"C:\02_QUILLAN")
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "03 - Training & Model"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from quillan_bpe_tokenizer import QuillanBPETokenizer

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

def clean_text(s: str) -> str:
    s = s.replace("<|im_end|>", "").replace("<|start|>", "").strip()
    for pre in ["# 🤖🧠 Quillan System Start 🧠🤖", "# 🤖🧠 Quillan System Start"]:
        if s.startswith(pre):
            s = s[len(pre):].strip()
    return s

def build_universal_master_pack(
    max_samples: int = 18000,
    seq_len: int = 256,
    output_path: Path = REPO_ROOT / "training_data" / "canonical_standardized" / "quillan_master_gold_training_v1.pt"
):
    print("=" * 75, flush=True)
    print("   👑 QUILLAN UNIVERSAL MASTER GOLD DATASET COMPILER (34-COUNCIL + MULTI-DOMAIN)", flush=True)
    print("=" * 75, flush=True)

    tok = QuillanBPETokenizer()
    print(f"[Tokenizer] Loaded vocab_size: {tok.vocab_size}", flush=True)

    all_pairs: List[Tuple[str, str]] = []

    # 1. Ingest all 34 Council Expert corpora
    experts_dir = REPO_ROOT / "training_data" / "experts_34"
    if experts_dir.exists():
        for exp_file in sorted(experts_dir.glob("*.jsonl")):
            cnt = 0
            with open(exp_file, "r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    if cnt >= 250:  # Take up to 250 samples per expert
                        break
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        p = clean_text(obj.get("question") or obj.get("prompt") or "")
                        r = clean_text(obj.get("response") or obj.get("answer") or "")
                        if p and r and len(r) >= 5:
                            all_pairs.append((p, r))
                            cnt += 1
                    except Exception:
                        continue
            print(f"[Council Expert] Ingested {cnt:4d} samples from {exp_file.name}", flush=True)

    # 2. Ingest Science Derivations
    science_files = [
        REPO_ROOT / "training_data" / "quillan_science_absolute.jsonl",
        REPO_ROOT / "training_data" / "quillan_science_additional.jsonl",
    ]
    for sf in science_files:
        if sf.exists():
            cnt = 0
            with open(sf, "r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    if cnt >= 800:
                        break
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        t = obj.get("text", "")
                        for delim in ["\nDERIVATION: ", "\nSOLUTION: ", "\nANSWER: ", "DERIVATION: ", "SOLUTION: "]:
                            if delim in t:
                                parts = t.split(delim, 1)
                                p = clean_text(parts[0])
                                r = clean_text(parts[1])
                                if p and r and len(r) >= 10:
                                    all_pairs.append((p, r))
                                    cnt += 1
                                break
                    except Exception:
                        continue
            print(f"[Science Corpus] Ingested {cnt:4d} samples from {sf.name}", flush=True)

    # 3. Ingest Standard Gold Datasets
    std_sources = [
        (REPO_ROOT / "training_data" / "Quillan_Clean_Reasoning_Gold_Dataset.jsonl", 2200),
        (REPO_ROOT / "training_data" / "Quillan_Direct_Answers_Gold.jsonl", 1200),
        (REPO_ROOT / "training_data" / "Quillan_Universal_Sovereign_Gold_1000.jsonl", 1000),
        (REPO_ROOT / "training_data" / "Quillan_General_Knowledge_Dataset.jsonl", 100),
        (REPO_ROOT / "training_data" / "Quillan_Explanatory_Prose_Dataset.jsonl", 100),
        (REPO_ROOT / "training_data" / "Quillan_Universal_100_Percent_Master_Gold.jsonl", 6000),
    ]

    for src_path, limit in std_sources:
        if not src_path.exists():
            continue
        cnt = 0
        with open(src_path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                if cnt >= limit or len(all_pairs) >= max_samples:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    p = clean_text(obj.get("question") or obj.get("prompt") or "")
                    r = clean_text(obj.get("response") or obj.get("answer") or obj.get("final_output") or "")
                    if p and r and len(r) >= 5:
                        all_pairs.append((p, r))
                        cnt += 1
                except Exception:
                    continue
        print(f"[Core Gold] Ingested {cnt:4d} samples from {src_path.name}", flush=True)

    print(f"\nTotal raw samples assembled across all sources: {len(all_pairs)}", flush=True)

    pad_id = 50256
    input_ids_list = []
    labels_list = []

    t0 = time.time()
    for idx, (prompt, resp) in enumerate(all_pairs):
        prompt_formatted = f"<|start|>\n<|user|>\n{prompt}\n<|assistant|>\n"
        resp_formatted = f"{resp}<|im_end|>\n"

        p_tokens = tok.encode(prompt_formatted)
        r_tokens = tok.encode(resp_formatted)

        if not r_tokens or len(r_tokens) < 2:
            continue

        # Balanced packing: preserve at least 64 tokens of response
        if len(p_tokens) + len(r_tokens) > seq_len:
            if len(r_tokens) < seq_len - 32:
                p_tokens = p_tokens[:seq_len - len(r_tokens)]
                full_tokens = p_tokens + r_tokens
            else:
                p_max = min(len(p_tokens), 64)
                p_tokens = p_tokens[:p_max]
                r_keep = seq_len - len(p_tokens)
                r_tokens = r_tokens[:r_keep]
                full_tokens = p_tokens + r_tokens
        else:
            full_tokens = p_tokens + r_tokens

        pad_len = seq_len - len(full_tokens)
        input_ids = full_tokens + [pad_id] * pad_len

        # Label: -100 for prompt and padding, target token IDs for response
        labels = [-100] * len(p_tokens) + r_tokens + [-100] * pad_len

        input_ids = input_ids[:seq_len]
        labels = labels[:seq_len]

        if any(l != -100 for l in labels):
            input_ids_list.append(input_ids)
            labels_list.append(labels)

    input_ids_tensor = torch.tensor(input_ids_list, dtype=torch.long)
    labels_tensor = torch.tensor(labels_list, dtype=torch.long)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "input_ids": input_ids_tensor,
        "labels": labels_tensor,
        "total_samples": len(input_ids_list),
        "seq_len": seq_len,
        "created_at": time.time(),
        "source": "quillan_universal_master_v2_all_34_experts",
    }
    torch.save(payload, output_path)

    elapsed = time.time() - t0
    print(f"\n[Success] Universal Master Dataset v2 successfully compiled in {elapsed:.2f}s!", flush=True)
    print(f"  Target File : {output_path}", flush=True)
    print(f"  Shape       : {input_ids_tensor.shape}", flush=True)
    print(f"  Total Tokens: {input_ids_tensor.numel():,} tokens", flush=True)
    print(f"  Target Ratio: {(labels_tensor != -100).sum().item() / labels_tensor.numel():.2%} supervised tokens", flush=True)

if __name__ == "__main__":
    build_universal_master_pack()
