#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
👑 QUILLAN-RONIN v5.3.1 — MASTER DATASET HARVESTER
Extracts and unifies all existing rich datasets in c:\\02_QUILLAN\\training_data:
  - code_train.jsonl (1,841 samples)
  - quillan_science_absolute.jsonl (1,141 samples)
  - instruct_train.jsonl (7,218 samples)
  - experts_34/*.jsonl (34 domain banks)
  - Quillan_Direct_Answers_Gold.jsonl (core seed)

Strips <think> tags to preserve direct, high-grade answers without internal monologue artifacts.
Standardizes on format:
  {"prompt": "Question: {q}\\nAnswer:\\n", "response": "{clean_answer}<|im_end|>"}
"""

import os
import sys
import re
import json
from pathlib import Path
from typing import List, Dict, Any

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

DATA_DIR = Path(r"C:\02_QUILLAN\training_data")
OUTPUT_FILE = DATA_DIR / "Quillan_Master_Combined_Gold.jsonl"

def clean_thought_tags(text: str) -> str:
    """Removes <think>...</think> reasoning traces to train crisp direct responses."""
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    text = re.sub(r'# 🤖🧠 Quillan System Start 🧠🤖.*?(?=\n\n|$)', '', text, flags=re.DOTALL)
    return text.strip()

def harvest_datasets():
    print("==================================================================", flush=True)
    print("   👑 HARVESTING ALL AVAILABLE QUILLAN TRAINING CORPORA", flush=True)
    print("==================================================================", flush=True)

    master_samples: List[Dict[str, str]] = []
    seen_prompts = set()

    # 1. Harvest code_train.jsonl & instruct_train.jsonl (Messages format)
    message_files = [
        DATA_DIR / "code_train.jsonl",
        DATA_DIR / "instruct_train.jsonl"
    ]

    for mf in message_files:
        if not mf.exists():
            continue
        print(f"[*] Harvesting {mf.name}...", flush=True)
        count = 0
        with open(mf, "r", encoding="utf-8", errors="replace") as f:
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
                            clean_ans = clean_thought_tags(raw_ans)
                            if user_q and clean_ans and user_q not in seen_prompts:
                                seen_prompts.add(user_q)
                                prompt = f"Question: {user_q}\nAnswer:\n"
                                response = f"{clean_ans}<|im_end|>"
                                master_samples.append({"prompt": prompt, "response": response})
                                count += 1
                            user_q = None
                except Exception:
                    continue
        print(f"    -> Extracted {count} clean Q&A pairs from {mf.name}", flush=True)

    # 2. Harvest quillan_science_absolute.jsonl (Physics/Science Derivations)
    sci_file = DATA_DIR / "quillan_science_absolute.jsonl"
    if sci_file.exists():
        print(f"[*] Harvesting {sci_file.name}...", flush=True)
        count = 0
        with open(sci_file, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    text = data.get("text", "")
                    if "PHYSICS PROBLEM:" in text and "DERIVATION:" in text:
                        parts = text.split("DERIVATION:", 1)
                        q = parts[0].replace("PHYSICS PROBLEM:", "").strip()
                        ans = parts[1].strip()
                        if q and ans and q not in seen_prompts:
                            seen_prompts.add(q)
                            prompt = f"Question: {q}\nAnswer:\n"
                            response = f"{ans}<|im_end|>"
                            master_samples.append({"prompt": prompt, "response": response})
                            count += 1
                except Exception:
                    continue
        print(f"    -> Extracted {count} science derivations from {sci_file.name}", flush=True)

    # 3. Harvest experts_34 directory
    experts_dir = DATA_DIR / "experts_34"
    if experts_dir.exists():
        print(f"[*] Harvesting Council Expert domain files from {experts_dir.name}/...", flush=True)
        count = 0
        for ef in experts_dir.glob("*.jsonl"):
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
                            clean_r = clean_thought_tags(r)
                            if clean_q and clean_r and clean_q not in seen_prompts:
                                seen_prompts.add(clean_q)
                                prompt = f"Question: {clean_q}\nAnswer:\n"
                                response = f"{clean_r}<|im_end|>"
                                master_samples.append({"prompt": prompt, "response": response})
                                count += 1
                    except Exception:
                        continue
        print(f"    -> Extracted {count} domain-expert samples from {experts_dir.name}/", flush=True)

    # 4. Include verified core Gold seed
    core_file = DATA_DIR / "Quillan_Direct_Answers_Gold.jsonl"
    if core_file.exists():
        count = 0
        with open(core_file, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    p = data.get("prompt", "").strip()
                    r = data.get("response", "").strip()
                    if p and r:
                        master_samples.append({"prompt": p, "response": r})
                        count += 1
                except Exception:
                    continue
        print(f"    -> Included {count} core gold precision samples", flush=True)

    # Write unified master dataset
    print(f"\n[*] Writing {len(master_samples)} combined master samples to {OUTPUT_FILE}...", flush=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as out_f:
        for s in master_samples:
            out_f.write(json.dumps(s, ensure_ascii=False) + "\n")

    size_mb = OUTPUT_FILE.stat().st_size / (1024 * 1024)
    print("==================================================================", flush=True)
    print(f"   🏆 MASTER HARVEST COMPLETE: {len(master_samples)} SAMPLES ({size_mb:.2f} MB)", flush=True)
    print("==================================================================", flush=True)

if __name__ == "__main__":
    harvest_datasets()
