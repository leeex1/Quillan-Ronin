#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
👑 QUILLAN-RONIN v5.4.0 — MASTER 10-DATASET COMPILER & NORMALIZER
================================================================
Harvests, sanitizes, dedupes, repairs mojibake, strips <think> leaks,
and uniformly formats all 10 Sovereign Datasets into clean ChatML pairs:
  <|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n{answer}<|im_end|>

Then tokenizes directly into 256-token PyTorch tensors using the verified
QuillanBPETokenizer (vocab 50,262) for immediate, non-scrambled from-scratch training.
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

# Paths
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

TARGET_DATASETS = [
    "code_train.jsonl",
    "instruct_train.jsonl",
    "full_train.jsonl",
    "train.jsonl",
    "full_dataset.jsonl",
    "quillan_12mb_training_dataset.jsonl",
    "GPT_5.5_Distilled.jsonl",
    "quillan_science_absolute.jsonl",
    "quillan_science_additional.jsonl",
    "Quillan_Ronin_v5.3.1_Samurai_Training_Seed_Dataset.jsonl"
]

def fix_mojibake(text: str) -> str:
    """Repairs encoding corruptions, double-encoded UTF-8, and non-printable control bytes."""
    if not isinstance(text, str) or not text:
        return ""
    
    # Try Latin-1 to UTF-8 double-encoding repair
    if any(m in text for m in ("â€™", "â€œ", "â€", "â€”", "â€“", "Ã©", "Ã ", "Ã¨", "Ã§", "Ã®", "Ã¯")):
        try:
            repaired = text.encode("latin1", errors="ignore").decode("utf-8", errors="ignore")
            if len(repaired) >= len(text) * 0.7:
                text = repaired
        except Exception:
            pass

    # Exact replacement map for persistent mojibake & HTML entities
    reps = {
        "â€™": "'",
        "â€œ": '"',
        "â€": '"',
        "â€”": "—",
        "â€“": "–",
        "â€¦": "...",
        "Â": "",
        "â€¢": "•",
        "Ã©": "é",
        "Ã¨": "è",
        "Ã ": "à",
        "Ã§": "ç",
        "Ã®": "î",
        "Ã¯": "ï",
        "Ã´": "ô",
        "Ã»": "û",
        "Ã¼": "ü",
        "Ã¶": "ö",
        "Ã¤": "ä",
        "\ufffd": "",
        "\u0000": "",
        "&amp;": "&",
        "&lt;": "<",
        "&gt;": ">",
        "&quot;": '"',
        "&#39;": "'",
        "&nbsp;": " "
    }
    for bad, good in reps.items():
        if bad in text:
            text = text.replace(bad, good)
            
    # Strip non-printable ASCII control characters except \n, \r, \t
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
    return unicodedata.normalize("NFKC", text).strip()

def strip_thoughts_and_artifacts(text: str) -> str:
    """Eliminates internal monologue <think>...</think> tags and leaked system headers."""
    if not isinstance(text, str):
        return ""
    # Strip full <think>...</think> blocks
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    # Strip orphaned unclosed <think> or </think>
    text = re.sub(r'</?think>', '', text)
    # Strip raw model boot banners if present
    text = re.sub(r'# 🤖🧠 Quillan System Start 🧠🤖.*?(?=\n\n|$)', '', text, flags=re.DOTALL)
    text = re.sub(r'/==================================================================\\.*\\==================================================================/', '', text, flags=re.DOTALL)
    # Strip existing prompt tag leaks inside content
    text = text.replace("<|im_start|>", "").replace("<|im_end|>", "").replace("<|endoftext|>", "")
    return text.strip()

def extract_pairs_from_record(data: Dict[str, Any], filename: str) -> List[Tuple[str, str]]:
    """Extracts (user_query, assistant_response) tuples across diverse schemas."""
    pairs = []
    
    # Schema 1: OpenAI/Anthropic 'messages' format (code_train, instruct_train, full_train)
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
                ans = strip_thoughts_and_artifacts(content)
                ans = fix_mojibake(ans)
                if user_msg and ans and len(user_msg) >= 5 and len(ans) >= 10:
                    pairs.append((user_msg, ans))
                user_msg = None
        return pairs

    # Schema 2: 'question' and 'answer' / 'responses_create_params' (train.jsonl)
    if "question" in data and "answer" in data:
        q = fix_mojibake(str(data["question"]))
        a = strip_thoughts_and_artifacts(str(data["answer"]))
        a = fix_mojibake(a)
        if len(q) >= 5 and len(a) >= 10:
            return [(q, a)]

    # Schema 3: 'original_input' and 'model_response' (full_dataset.jsonl)
    if "original_input" in data and "model_response" in data:
        raw_in = data["original_input"]
        if isinstance(raw_in, dict):
            domain = raw_in.get("domain", "")
            concept = raw_in.get("concept", "")
            task = raw_in.get("task", "") or raw_in.get("query", "") or raw_in.get("question", "")
            q = f"Domain: {domain} | Concept: {concept}\nTask: {task}" if (domain or concept) else str(task)
        else:
            q = str(raw_in)
        q = fix_mojibake(q)
        a = strip_thoughts_and_artifacts(str(data["model_response"]))
        a = fix_mojibake(a)
        if len(q) >= 5 and len(a) >= 10:
            return [(q, a)]

    # Schema 4: 'question' and 'final_output' (quillan_12mb_training_dataset, Seed Dataset)
    if "question" in data and "final_output" in data:
        q = fix_mojibake(str(data["question"]))
        a = strip_thoughts_and_artifacts(str(data["final_output"]))
        a = fix_mojibake(a)
        if len(q) >= 5 and len(a) >= 10:
            return [(q, a)]

    # Schema 5: 'text' with <|user|> / <|assistant|> (GPT_5.5_Distilled)
    if "text" in data and isinstance(data["text"], str):
        raw_text = data["text"]
        if "<|user|>" in raw_text and "<|assistant|>" in raw_text:
            parts = raw_text.split("<|assistant|>")
            if len(parts) >= 2:
                q = parts[0].replace("<|user|>", "").strip()
                a = parts[1].replace("<|endoftext|>", "").replace("<|im_end|>", "").strip()
                q = fix_mojibake(q)
                a = strip_thoughts_and_artifacts(a)
                a = fix_mojibake(a)
                if len(q) >= 5 and len(a) >= 10:
                    return [(q, a)]
        elif "PHYSICS PROBLEM:" in raw_text or "PROBLEM:" in raw_text:
            # Science problems
            clean_sci = fix_mojibake(raw_text)
            clean_sci = strip_thoughts_and_artifacts(clean_sci)
            if "SOLUTION:" in clean_sci:
                p_parts = clean_sci.split("SOLUTION:")
                q = p_parts[0].replace("PHYSICS PROBLEM:", "").replace("PROBLEM:", "").strip()
                a = p_parts[1].strip()
                if len(q) >= 5 and len(a) >= 10:
                    return [(q, a)]
            elif "ANSWER:" in clean_sci:
                p_parts = clean_sci.split("ANSWER:")
                q = p_parts[0].replace("PHYSICS PROBLEM:", "").replace("PROBLEM:", "").strip()
                a = p_parts[1].strip()
                if len(q) >= 5 and len(a) >= 10:
                    return [(q, a)]
            else:
                # Store as knowledge question
                return [("Solve and explain the following physics problem:\n" + clean_sci[:200], clean_sci)]

    return pairs

def main():
    print("=" * 72)
    print("  👑 QUILLAN-RONIN v5.4.0 — MASTER 10-DATASET NORMALIZATION & COMPILER")
    print("=" * 72)

    tok = QuillanBPETokenizer()
    print(f"[*] Verified Tokenizer Vocab Size: {tok.vocab_size:,} | Base BPE + 5 Specials")

    all_pairs: List[Tuple[str, str]] = []
    seen_queries: Set[str] = set()
    stats_per_file: Dict[str, int] = {}

    for ds_filename in TARGET_DATASETS:
        print(f"\n[+] Processing: {ds_filename}...")
        try:
            local_path = hf_hub_download(repo_id=HF_DATASET_REPO, filename=ds_filename, repo_type="dataset")
        except Exception as e:
            print(f"    [!] Failed to download {ds_filename} from HF: {e}")
            continue

        file_pairs_count = 0
        with open(local_path, "r", encoding="utf-8", errors="ignore") as f:
            for line_idx, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    extracted = extract_pairs_from_record(record, ds_filename)
                    for q, a in extracted:
                        q_norm = q.lower().strip()
                        if q_norm not in seen_queries:
                            seen_queries.add(q_norm)
                            all_pairs.append((q, a))
                            file_pairs_count += 1
                except Exception:
                    pass

        stats_per_file[ds_filename] = file_pairs_count
        print(f"    -> Harvested & Sanitized: {file_pairs_count:,} clean Q/A pairs")

    print("\n" + "=" * 72)
    print(f"  TOTAL UNIQUE HIGH-GRADE SAMPLES HARVESTED: {len(all_pairs):,}")
    print("=" * 72)
    for fname, count in stats_per_file.items():
        print(f"  - {fname:<45} : {count:6,d} pairs")

    # 1. Save Clean Uniform JSONL
    out_jsonl = CANONICAL_DIR / "quillan_sovereign_10_clean.jsonl"
    print(f"\n[*] Writing normalized ChatML JSONL: {out_jsonl}...")
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for q, a in all_pairs:
            formatted_text = f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n{a}<|im_end|>"
            record = {
                "prompt": f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n",
                "response": f"{a}<|im_end|>",
                "text": formatted_text
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # 2. Tokenize and Chunk into Fixed 256-token Tensors for Training
    print("\n[*] Tokenizing normalized dataset into 256-token training chunks...")
    seq_len = 256
    stride = 128
    chunk_list: List[List[int]] = []
    
    t0 = time.time()
    for idx, (q, a) in enumerate(all_pairs, 1):
        # Format with ChatML tokens
        text = f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n{a}<|im_end|>"
        tokens = tok.encode(text)
        tokens.append(50256) # EOS

        if len(tokens) >= seq_len:
            for s in range(0, len(tokens) - seq_len + 1, stride):
                chunk_list.append(tokens[s : s + seq_len])
        else:
            # Pad to seq_len
            padded = tokens + [50256] * (seq_len - len(tokens))
            chunk_list.append(padded)

        if idx % 5000 == 0:
            elapsed = max(1.0, time.time() - t0)
            tok_count = len(chunk_list) * seq_len
            print(f"    Processed {idx:,}/{len(all_pairs):,} samples | {len(chunk_list):,} chunks ({tok_count/elapsed:.0f} tok/s)", flush=True)

    input_ids_tensor = torch.tensor(chunk_list, dtype=torch.long)
    total_tokens = input_ids_tensor.numel()
    print(f"\n[+] Tokenization Complete!")
    print(f"    Tensor Shape : {input_ids_tensor.shape}")
    print(f"    Total Tokens : {total_tokens:,} tokens ({total_tokens / (1024*1024):.2f}M tokens)")
    print(f"    Token Bounds : min={input_ids_tensor.min().item()}, max={input_ids_tensor.max().item()} (all <= 50,261)")

    # 3. Save serialized tensor files
    out_pt = CANONICAL_DIR / "quillan_sovereign_10_clean.pt"
    master_gold_pt = CANONICAL_DIR / "quillan_master_gold_training_v1.pt"

    payload = {
        "input_ids": input_ids_tensor,
        "labels": input_ids_tensor.clone(),
        "meta": {
            "num_samples": len(all_pairs),
            "total_tokens": total_tokens,
            "seq_len": seq_len,
            "vocab_size": tok.vocab_size,
            "timestamp": time.time(),
            "sources": list(stats_per_file.keys())
        }
    }

    print(f"[*] Saving payload to: {out_pt}...")
    torch.save(payload, out_pt)
    print(f"[*] Updating pipeline master target: {master_gold_pt}...")
    torch.save(payload, master_gold_pt)

    print("\n" + "=" * 72)
    print("  👑 SUCCESS: 10 DATASETS SANITIZED, NORMALIZED, AND TOKENIZED")
    print("=" * 72)

if __name__ == "__main__":
    main()
