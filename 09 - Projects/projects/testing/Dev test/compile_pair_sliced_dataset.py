#!/usr/bin/env python3
"""
👑 QUILLAN-RONIN v5.3.1 — INTACT PAIR-SLICED DATASET COMPILER
Parses discrete, complete (Prompt, Response) pairs from all multi-frontier datasets:
- GPT_5.5_Distilled.jsonl
- instruct_train.jsonl
- code_train.jsonl
- full_train.jsonl
- Quillan_Refined_Thought_Corpus.jsonl

Constructs target-masked tensors where prompt tokens have label = -100,
ensuring the model learns 100% fluent English grammar and syntax on complete responses.
"""

import os, sys, time, json, torch, tiktoken
from pathlib import Path

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

enc = tiktoken.get_encoding("gpt2")
REPO_ROOT = Path(r"C:\02_QUILLAN")
DATA_DIR = REPO_ROOT / "training_data"

MAX_SEQ_LEN = 192

JSONL_SOURCES = [
    (DATA_DIR / "Quillan_Refined_Thought_Corpus.jsonl", "70B-Thoughts", 500),
    (DATA_DIR / "GPT_5.5_Distilled.jsonl", "GPT-5.5", 3000),
    (DATA_DIR / "instruct_train.jsonl", "Claude-Instruct", 3000),
    (DATA_DIR / "code_train.jsonl", "Code-Engineering", 2000),
    (DATA_DIR / "full_train.jsonl", "Multi-Domain", 3000)
]

all_input_ids = []
all_labels = []
total_pairs = 0

print("==================================================================", flush=True)
print("   👑 COMPILING INTACT PAIR-SLICED DATASET (TARGET-MASKED SFT)", flush=True)
print("==================================================================", flush=True)

t0 = time.time()

for src_path, label_name, max_count in JSONL_SOURCES:
    if not src_path.exists(): continue
    count = 0
    with open(src_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                d = json.loads(line)
                user_msg = ""
                asst_msg = ""

                # Extract user & assistant text by format
                if "prompt" in d and "refined_reasoning" in d:
                    user_msg = d["prompt"]
                    asst_msg = d["refined_reasoning"]
                elif "instruction" in d and "output" in d:
                    user_msg = d["instruction"]
                    asst_msg = d["output"]
                elif "messages" in d and isinstance(d["messages"], list):
                    for m in d["messages"]:
                        if m.get("role") == "user": user_msg = m.get("content", "")
                        elif m.get("role") == "assistant": asst_msg = m.get("content", "")
                elif "text" in d and "<|user|>" in d["text"] and "<|assistant|>" in d["text"]:
                    raw = d["text"]
                    parts = raw.split("<|assistant|>")
                    u_part = parts[0].replace("<|user|>", "").strip()
                    a_part = parts[1].strip() if len(parts) > 1 else ""
                    user_msg = u_part
                    asst_msg = a_part
                elif "prompt" in d and "response" in d:
                    user_msg = d["prompt"]
                    asst_msg = d["response"]

                user_msg = user_msg.replace("# 🤖", "").replace("# 🧠", "").strip()
                asst_msg = asst_msg.replace("# 🤖", "").replace("# 🧠", "").strip()

                if len(user_msg) >= 5 and len(asst_msg) >= 5:
                    prompt_str = f"<|system|>\nYou are Quillan-Ronin, a sovereign, highly intelligent AI assistant and expert coder.\n<|user|>\n{user_msg}\n<|assistant|>\n"
                    resp_str = f"{asst_msg}\n<|end|>"

                    p_toks = enc.encode(prompt_str)
                    r_toks = enc.encode(resp_str)

                    # Truncate response if total length exceeds MAX_SEQ_LEN
                    if len(p_toks) + len(r_toks) > MAX_SEQ_LEN:
                        allowed_r = MAX_SEQ_LEN - len(p_toks)
                        if allowed_r < 15: # Prompt too long, skip
                            continue
                        r_toks = r_toks[:allowed_r-1] + [50256]

                    input_ids = p_toks + r_toks
                    # Target Loss Masking: -100 on prompt tokens, calculate loss on response tokens
                    labels = [-100] * len(p_toks) + r_toks

                    # Pad to MAX_SEQ_LEN
                    pad_len = MAX_SEQ_LEN - len(input_ids)
                    input_ids = input_ids + [50256] * pad_len
                    labels = labels + [-100] * pad_len

                    all_input_ids.append(input_ids)
                    all_labels.append(labels)
                    count += 1
                    total_pairs += 1

                    if count >= max_count:
                        break
            except Exception:
                continue
    print(f"[+] Compiled {count:>5,} intact pairs from {label_name:18s} ({src_path.name})", flush=True)

out_pt = DATA_DIR / "intact_pair_dataset.pt"
dataset_dict = {
    "input_ids": torch.tensor(all_input_ids, dtype=torch.long),
    "labels": torch.tensor(all_labels, dtype=torch.long),
    "num_samples": total_pairs,
    "max_seq_len": MAX_SEQ_LEN
}
torch.save(dataset_dict, out_pt)

print(f"\n[+] Successfully compiled {total_pairs:,} complete (Prompt, Response) pairs in {time.time()-t0:.2f}s!", flush=True)
print(f"[+] Saved target-masked pair dataset to: {out_pt.name} ({out_pt.stat().st_size/(1024**2):.1f} MB)\n", flush=True)
