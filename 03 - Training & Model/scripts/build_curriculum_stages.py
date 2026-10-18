#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
👑 QUILLAN-RONIN v5.4.0 — 3-STAGE CURRICULUM DATASET BUILDER
============================================================
Compiles the 10 master datasets into 3 distinct curriculum stages:
  - Stage 1: Identity & Foundation Pre-training
      (Quillan Seed, 12MB Sovereign, train.jsonl, full_train.jsonl)
  - Stage 2: Instruction Following & Conversational Alignment
      (instruct_train.jsonl, full_dataset.jsonl)
  - Stage 3: High-Tier Reasoning, Code, Math, Science & System Architecture
      (code_train.jsonl, science_absolute, science_additional, GPT_5.5_Distilled)
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import os
import re
import json
import time
import unicodedata
from pathlib import Path
from typing import List, Dict, Any, Tuple, Set
import torch
from huggingface_hub import hf_hub_download

# Path resolution
REPO_ROOT = Path(r"C:\02_QUILLAN")
MODEL_DIR = REPO_ROOT / "03 - Training & Model"
SCRIPTS_DIR = MODEL_DIR / "scripts"
DATA_DIR = MODEL_DIR / "training_data"
CANONICAL_DIR = DATA_DIR / "canonical_standardized"
CANONICAL_DIR.mkdir(parents=True, exist_ok=True)

for p in [str(SCRIPTS_DIR), str(MODEL_DIR), str(REPO_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from quillan_bpe_tokenizer import QuillanBPETokenizer

HF_DATASET_REPO = "CrashOverrideX/Quillan_Samurai_sets"

STAGES = {
    "stage1_identity_foundation": [
        "Quillan_Ronin_v5.3.1_Samurai_Training_Seed_Dataset.jsonl",
        "quillan_12mb_training_dataset.jsonl",
        "train.jsonl",
        "full_train.jsonl"
    ],
    "stage2_instructions_alignment": [
        "instruct_train.jsonl",
        "full_dataset.jsonl"
    ],
    "stage3_math_code_science": [
        "code_train.jsonl",
        "quillan_science_absolute.jsonl",
        "quillan_science_additional.jsonl",
        "GPT_5.5_Distilled.jsonl"
    ]
}

def fix_mojibake(text: str) -> str:
    if not isinstance(text, str) or not text:
        return ""
    if any(m in text for m in ("â€™", "â€œ", "â€", "â€”", "â€“", "Ã©", "Ã ", "Ã¨", "Ã§")):
        try:
            repaired = text.encode("latin1", errors="ignore").decode("utf-8", errors="ignore")
            if len(repaired) >= len(text) * 0.7:
                text = repaired
        except Exception:
            pass

    reps = {
        "â€™": "'", "â€œ": '"', "â€": '"', "â€”": "—", "â€“": "–", "â€¦": "...",
        "Â": "", "â€¢": "•", "Ã©": "é", "Ã¨": "è", "Ã ": "à", "Ã§": "ç",
        "\ufffd": "", "\u0000": "", "&amp;": "&", "&lt;": "<", "&gt;": ">", "&quot;": '"', "&#39;": "'"
    }
    for b, g in reps.items():
        if b in text:
            text = text.replace(b, g)
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
    return unicodedata.normalize("NFKC", text).strip()

def strip_thoughts_and_artifacts(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    text = re.sub(r'</?think>', '', text)
    text = re.sub(r'# 🤖🧠 Quillan System Start 🧠🤖.*?(?=\n\n|$)', '', text, flags=re.DOTALL)
    text = text.replace("<|im_start|>", "").replace("<|im_end|>", "").replace("<|endoftext|>", "")
    return text.strip()

def extract_pairs(data: Dict[str, Any], filename: str) -> List[Tuple[str, str]]:
    pairs = []
    if "messages" in data and isinstance(data["messages"], list):
        user_msg = None
        for msg in data["messages"]:
            if not isinstance(msg, dict):
                continue
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "user":
                user_msg = fix_mojibake(content)
            elif role == "assistant" and user_msg:
                ans = fix_mojibake(strip_thoughts_and_artifacts(content))
                if user_msg and ans and len(user_msg) >= 5 and len(ans) >= 10:
                    pairs.append((user_msg, ans))
                user_msg = None
        return pairs

    if "question" in data and "answer" in data:
        q = fix_mojibake(str(data["question"]))
        a = fix_mojibake(strip_thoughts_and_artifacts(str(data["answer"])))
        if len(q) >= 5 and len(a) >= 10:
            return [(q, a)]

    if "original_input" in data and "model_response" in data:
        raw_in = data["original_input"]
        if isinstance(raw_in, dict):
            task = raw_in.get("task", "") or raw_in.get("query", "") or raw_in.get("question", "")
            q = str(task)
        else:
            q = str(raw_in)
        q = fix_mojibake(q)
        a = fix_mojibake(strip_thoughts_and_artifacts(str(data["model_response"])))
        if len(q) >= 5 and len(a) >= 10:
            return [(q, a)]

    if "question" in data and "final_output" in data:
        q = fix_mojibake(str(data["question"]))
        a = fix_mojibake(strip_thoughts_and_artifacts(str(data["final_output"])))
        if len(q) >= 5 and len(a) >= 10:
            return [(q, a)]

    if "text" in data and isinstance(data["text"], str):
        raw_text = data["text"]
        if "<|user|>" in raw_text and "<|assistant|>" in raw_text:
            parts = raw_text.split("<|assistant|>")
            if len(parts) >= 2:
                q = fix_mojibake(parts[0].replace("<|user|>", "").strip())
                a = fix_mojibake(strip_thoughts_and_artifacts(parts[1]))
                if len(q) >= 5 and len(a) >= 10:
                    return [(q, a)]
        elif "PHYSICS PROBLEM:" in raw_text or "PROBLEM:" in raw_text:
            clean_sci = fix_mojibake(strip_thoughts_and_artifacts(raw_text))
            for split_token in ["SOLUTION:", "ANSWER:"]:
                if split_token in clean_sci:
                    p_parts = clean_sci.split(split_token)
                    q = p_parts[0].replace("PHYSICS PROBLEM:", "").replace("PROBLEM:", "").strip()
                    a = p_parts[1].strip()
                    if len(q) >= 5 and len(a) >= 10:
                        return [(q, a)]
            return [("Explain and solve this physics problem:\n" + clean_sci[:200], clean_sci)]

    return pairs

def build_stage(stage_key: str, filenames: List[str], tok: QuillanBPETokenizer, seq_len: int = 256, stride: int = 128):
    print(f"\n{'='*70}")
    print(f"  BUILDING {stage_key.upper()}")
    print(f"{'='*70}")

    stage_pairs: List[Tuple[str, str]] = []
    seen = set()

    for fname in filenames:
        print(f"[*] Harvesting {fname}...")
        try:
            local_path = hf_hub_download(repo_id=HF_DATASET_REPO, filename=fname, repo_type="dataset")
        except Exception as e:
            print(f"    [!] Error downloading {fname}: {e}")
            continue

        file_count = 0
        with open(local_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    pairs = extract_pairs(record, fname)
                    for q, a in pairs:
                        norm_q = q.lower().strip()
                        if norm_q not in seen:
                            seen.add(norm_q)
                            stage_pairs.append((q, a))
                            file_count += 1
                except Exception:
                    pass
        print(f"    -> Extracted {file_count:,} unique samples")

    print(f"\n[+] Total unique samples in {stage_key}: {len(stage_pairs):,}")

    # Tokenize into chunks
    print(f"[*] Tokenizing into {seq_len}-token sequences...")
    chunk_list: List[List[int]] = []
    t0 = time.time()

    for idx, (q, a) in enumerate(stage_pairs, 1):
        text = f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n{a}<|im_end|>"
        tokens = tok.encode(text)
        tokens.append(50256)

        if len(tokens) >= seq_len:
            for s in range(0, len(tokens) - seq_len + 1, stride):
                chunk_list.append(tokens[s : s + seq_len])
        else:
            padded = tokens + [50256] * (seq_len - len(tokens))
            chunk_list.append(padded)

        if idx % 5000 == 0:
            elapsed = max(1.0, time.time() - t0)
            toks = len(chunk_list) * seq_len
            print(f"    Processed {idx:,}/{len(stage_pairs):,} | {len(chunk_list):,} chunks ({toks/elapsed:.0f} tok/s)")

    tensor_data = torch.tensor(chunk_list, dtype=torch.long)
    total_tokens = tensor_data.numel()
    print(f"[+] {stage_key} Complete: shape={tensor_data.shape}, total_tokens={total_tokens:,} ({total_tokens/(1024*1024):.2f}M)")

    out_file = CANONICAL_DIR / f"{stage_key}.pt"
    payload = {
        "input_ids": tensor_data,
        "labels": tensor_data.clone(),
        "meta": {
            "stage": stage_key,
            "num_samples": len(stage_pairs),
            "total_tokens": total_tokens,
            "seq_len": seq_len,
            "sources": filenames
        }
    }
    torch.save(payload, out_file)
    print(f"[✓] Saved stage tensor to: {out_file}")
    return out_file

def main():
    print("======================================================================")
    print("  👑 QUILLAN-RONIN 3-STAGE PROGRESSIVE CURRICULUM COMPILER")
    print("======================================================================")

    tok = QuillanBPETokenizer()
    print(f"[*] Verified Tokenizer Vocab Size: {tok.vocab_size:,}")

    built_files = {}
    for stage_key, file_list in STAGES.items():
        out_p = build_stage(stage_key, file_list, tok)
        built_files[stage_key] = out_p

    print("\n" + "=" * 70)
    print("  👑 ALL 3 CURRICULUM STAGES SUCCESSFULLY COMPILED & SERIALIZED")
    print("=" * 70)
    for k, p in built_files.items():
        sz = p.stat().st_size / (1024 * 1024)
        print(f"  - {k:<35} : {sz:6.2f} MB -> {p.name}")

if __name__ == "__main__":
    main()
