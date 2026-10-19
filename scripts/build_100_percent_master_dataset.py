#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
👑 QUILLAN-RONIN v5.3.1 — 100% EXHAUSTIVE MASTER DATASET INGESTION
Ingests 100% of every available training corpus in c:\\02_QUILLAN\\training_data:
  1. full_train.jsonl (68.1 MB)
  2. instruct_train.jsonl (57.8 MB)
  3. code_train.jsonl (11.9 MB)
  4. train.jsonl (29.3 MB)
  5. full_dataset.jsonl (27.0 MB)
  6. quillan_12mb_training_dataset.jsonl (12.0 MB)
  7. GPT_5.5_Distilled.jsonl (44.5 MB)
  8. pdf_papers_corpus.jsonl (9.0 MB)
  9. quillan_science_absolute.jsonl (2.5 MB)
  10. quillan_science_additional.jsonl (4.2 MB)
  11. Quillan_Clean_Reasoning_Gold_Dataset.jsonl (2.2 MB)
  12. All Quillan_* Gold datasets
  13. All 34 Council Expert files in experts_34/

Strips <think> tags to preserve direct, high-grade answers without internal monologue artifacts.
Outputs to: Quillan_Universal_100_Percent_Master_Gold.jsonl
"""

import os
import sys
import re
import json
from pathlib import Path
from typing import List, Dict, Any, Set

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

DATA_DIR = Path(r"C:\02_QUILLAN\training_data")
OUTPUT_FILE = DATA_DIR / "Quillan_Universal_100_Percent_Master_Gold.jsonl"

def clean_text(text: str) -> str:
    """Removes <think>...</think> reasoning traces to train crisp direct responses."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    text = re.sub(r'# 🤖🧠 Quillan System Start 🧠🤖.*?(?=\n\n|$)', '', text, flags=re.DOTALL)
    return text.strip()

def process_file_messages(file_path: Path, master_samples: List[Dict[str, str]], seen_prompts: Set[str]):
    count = 0
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                messages = data.get("messages", [])
                user_q = None
                for msg in messages:
                    if msg.get("role") == "user":
                        user_q = msg.get("content", "").strip()
                    elif msg.get("role") == "assistant" and user_q:
                        raw_ans = msg.get("content", "").strip()
                        clean_ans = clean_text(raw_ans)
                        if user_q and clean_ans and len(clean_ans) > 5 and user_q not in seen_prompts:
                            seen_prompts.add(user_q)
                            prompt = f"Question: {user_q}\nAnswer:\n"
                            response = f"{clean_ans}<|im_end|>"
                            master_samples.append({"prompt": prompt, "response": response})
                            count += 1
                        user_q = None
            except Exception:
                continue
    print(f"    -> Extracted {count:,} samples from {file_path.name}", flush=True)

def process_file_qa(file_path: Path, q_key: str, a_key: str, master_samples: List[Dict[str, str]], seen_prompts: Set[str]):
    count = 0
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                q_val = data.get(q_key, "")
                if isinstance(q_val, dict):
                    q = q_val.get("text") or q_val.get("question") or q_val.get("prompt") or ""
                else:
                    q = str(q_val)
                    
                a_val = data.get(a_key, "")
                if isinstance(a_val, dict):
                    a = a_val.get("text") or a_val.get("content") or a_val.get("response") or ""
                else:
                    a = str(a_val)
                    
                if q and a:
                    q = q.replace("Question:", "").replace("Answer:", "").strip()
                    clean_a = clean_text(a)
                    if q and clean_a and len(clean_a) > 5 and q not in seen_prompts:
                        seen_prompts.add(q)
                        prompt = f"Question: {q}\nAnswer:\n"
                        response = f"{clean_a}<|im_end|>"
                        master_samples.append({"prompt": prompt, "response": response})
                        count += 1
            except Exception:
                continue
    print(f"    -> Extracted {count:,} samples from {file_path.name}", flush=True)

def process_file_science_text(file_path: Path, master_samples: List[Dict[str, str]], seen_prompts: Set[str]):
    count = 0
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                text = data.get("text", "")
                if not isinstance(text, str):
                    continue
                if "PHYSICS PROBLEM:" in text and "DERIVATION:" in text:
                    parts = text.split("DERIVATION:", 1)
                    q = parts[0].replace("PHYSICS PROBLEM:", "").strip()
                    ans = clean_text(parts[1])
                    if q and ans and q not in seen_prompts:
                        seen_prompts.add(q)
                        prompt = f"Question: {q}\nAnswer:\n"
                        response = f"{ans}<|im_end|>"
                        master_samples.append({"prompt": prompt, "response": response})
                        count += 1
                elif "PROBLEM:" in text and "SOLUTION:" in text:
                    parts = text.split("SOLUTION:", 1)
                    p_part = parts[0].split("PROBLEM:", 1)[1] if "PROBLEM:" in parts[0] else parts[0]
                    if "REASONING:" in p_part:
                        p_part = p_part.split("REASONING:")[0]
                    q = p_part.strip()
                    ans = clean_text(parts[1])
                    if q and ans and q not in seen_prompts:
                        seen_prompts.add(q)
                        prompt = f"Question: {q}\nAnswer:\n"
                        response = f"{ans}<|im_end|>"
                        master_samples.append({"prompt": prompt, "response": response})
                        count += 1
                elif "Question:" in text and "Answer:" in text:
                    parts = text.split("Answer:", 1)
                    q = parts[0].replace("Question:", "").strip()
                    ans = clean_text(parts[1])
                    if q and ans and q not in seen_prompts:
                        seen_prompts.add(q)
                        prompt = f"Question: {q}\nAnswer:\n"
                        response = f"{ans}<|im_end|>"
                        master_samples.append({"prompt": prompt, "response": response})
                        count += 1
            except Exception:
                continue
    print(f"    -> Extracted {count:,} science derivations from {file_path.name}", flush=True)

def harvest_100_percent():
    print("==================================================================", flush=True)
    print("   👑 INGESTING 100% OF ALL AVAILABLE TRAINING DATASETS", flush=True)
    print("==================================================================", flush=True)

    master_samples: List[Dict[str, str]] = []
    seen_prompts: Set[str] = set()

    # 1. Message-based Datasets
    print("[*] Processing message-based conversational datasets...", flush=True)
    for fn in ["full_train.jsonl", "instruct_train.jsonl", "code_train.jsonl"]:
        fp = DATA_DIR / fn
        if fp.exists():
            process_file_messages(fp, master_samples, seen_prompts)

    # 2. Q&A and Thought/Response Datasets
    print("\n[*] Processing structured Q&A / Thought corpora...", flush=True)
    qa_configs = [
        ("train.jsonl", "question", "answer"),
        ("full_dataset.jsonl", "original_input", "model_response"),
        ("quillan_12mb_training_dataset.jsonl", "question", "final_output"),
        ("Quillan_Ronin_v5.3.1_Samurai_Training_Seed_Dataset.jsonl", "question", "final_output"),
        ("Quillan_Clean_Reasoning_Gold_Dataset.jsonl", "question", "response"),
        ("Quillan_Explanatory_Prose_Dataset.jsonl", "question", "response"),
        ("Quillan_General_Knowledge_Dataset.jsonl", "question", "response"),
        ("Quillan_Hyper_Tune_Gold_Dataset.jsonl", "question", "response"),
        ("Quillan_70B_Teacher_Distilled_Gold.jsonl", "prompt", "response"),
        ("Quillan_Direct_Answers_Gold.jsonl", "prompt", "response"),
        ("Quillan_Refined_Thought_Corpus.jsonl", "prompt", "refined_reasoning"),
    ]

    for fn, qk, ak in qa_configs:
        fp = DATA_DIR / fn
        if fp.exists():
            process_file_qa(fp, qk, ak, master_samples, seen_prompts)

    # 3. Science, Physics, Papers & Distilled Datasets
    print("\n[*] Processing science, physics, derivations & paper corpora...", flush=True)
    science_files = [
        "quillan_science_absolute.jsonl",
        "quillan_science_additional.jsonl",
        "GPT_5.5_Distilled.jsonl",
        "pdf_papers_corpus.jsonl"
    ]
    for fn in science_files:
        fp = DATA_DIR / fn
        if fp.exists():
            process_file_science_text(fp, master_samples, seen_prompts)

    # 4. All 34 Council Expert files in experts_34/
    experts_dir = DATA_DIR / "experts_34"
    if experts_dir.exists():
        print(f"\n[*] Processing all 34 Council Expert domain banks in {experts_dir.name}/...", flush=True)
        count = 0
        for ef in sorted(experts_dir.glob("*.jsonl")):
            with open(ef, "r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        p = data.get("prompt") or data.get("instruction") or data.get("question")
                        r = data.get("response") or data.get("output") or data.get("answer")
                        if p and r:
                            clean_q = p.replace("Question:", "").replace("Answer:", "").strip()
                            clean_r = clean_text(r)
                            if clean_q and clean_r and len(clean_r) > 5 and clean_q not in seen_prompts:
                                seen_prompts.add(clean_q)
                                prompt = f"Question: {clean_q}\nAnswer:\n"
                                response = f"{clean_r}<|im_end|>"
                                master_samples.append({"prompt": prompt, "response": response})
                                count += 1
                    except Exception:
                        continue
        print(f"    -> Extracted {count:,} domain-expert samples from {experts_dir.name}/", flush=True)

    # Save 100% Unified Master Dataset
    print(f"\n[*] Writing {len(master_samples):,} total unified samples to {OUTPUT_FILE.name}...", flush=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as out_f:
        for s in master_samples:
            out_f.write(json.dumps(s, ensure_ascii=False) + "\n")

    size_mb = OUTPUT_FILE.stat().st_size / (1024 * 1024)
    print("==================================================================", flush=True)
    print(f"   🏆 100% MASTER DATASET CREATED: {len(master_samples):,} SAMPLES ({size_mb:.2f} MB)", flush=True)
    print("==================================================================", flush=True)

if __name__ == "__main__":
    harvest_100_percent()
