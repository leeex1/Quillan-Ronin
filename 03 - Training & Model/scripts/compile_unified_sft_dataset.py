#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compile_unified_sft_dataset.py
==============================
Compiles and re-tokenizes all authentic multi-model datasets into the canonical
Quillan format with zero data corruption and zero synthetic repetition:

1. Samples with Thinking (Claude Opus Code, Claude Opus Instruct, GPT-5.5 Think, Sovereign Think):
   Prompt (Masked with -100):
     User: {input}

     Assistant:
     <think>

   Target (Learned Model Output):
     {think}
     </think>
     <output>
     {output}
     </output><|endoftext|>

2. Direct QA Samples (GPT-5.5 Direct, Direct Answers Gold, Universal Sovereign Gold):
   Prompt (Masked with -100):
     User: {input}

     Assistant:
     <output>

   Target (Learned Model Output):
     {output}
     </output><|endoftext|>

EXCLUSIONS:
  - train.jsonl (Haystack riddles with single-token answers) is 100% EXCLUDED.
  - Zero hardcoded dummy thinking strings.

GUARANTEES:
  - 100% INTACT samples: Fits completely within seq_len=512 with proper closing tags.
  - Correct EOS token: 50256 (<|endoftext|>).
  - Target-loss masking: Prompt and padding tokens are strictly -100.
  - Pre-tokenized tensor cache saved to E:\\quillan_datasets\\sft_master_tokenized_cache.pt.
"""

import os
import sys
import json
import time
import hashlib
from pathlib import Path

import torch
from tokenizers import Tokenizer

MAX_SEQ_LEN = 512
REPO_ROOT   = Path(r"C:\02_QUILLAN")
DATA_DIR    = Path(r"E:\quillan_datasets")
GOLD_DIR    = REPO_ROOT / "03 - Training & Model" / "training_data"
OUT_JSONL   = DATA_DIR / "unified_sft_intact.jsonl"
OUT_CACHE   = DATA_DIR / "sft_master_tokenized_cache.pt"
TMP_CACHE   = DATA_DIR / "sft_master_tokenized_cache.tmp"

tok_path = REPO_ROOT / "03 - Training & Model" / "quillan_bpe_tokenizer_hf" / "tokenizer.json"
tok = Tokenizer.from_file(str(tok_path))

print("=" * 70, flush=True)
print("  QUILLAN-RONIN — COMPILING CLEAN SFT DATASET (RETOKENIZATION)", flush=True)
print(f"  Target Sequence Length: {MAX_SEQ_LEN} tokens", flush=True)
print("=" * 70, flush=True)

seen_prompts = set()
all_inps = []
all_lbls = []
all_keys = []
jsonl_records = []
stats = {}

def process_sample(inp_text, out_text, think_text=None, source_tag=""):
    inp_text = str(inp_text).strip()
    out_text = str(out_text).strip()
    if len(inp_text) < 4 or len(out_text) < 1:
        return False

    # Deduplicate on prompt
    p_hash = hashlib.sha256(inp_text.lower().encode("utf-8")).digest()
    if p_hash in seen_prompts:
        return False
    seen_prompts.add(p_hash)

    if think_text and str(think_text).strip():
        th = str(think_text).strip()
        prompt_str = f"User: {inp_text}\n\nAssistant:\n<think>\n"
        resp_str   = f"{th}\n</think>\n<output>\n{out_text}\n</output><|endoftext|>"
    else:
        th = ""
        prompt_str = f"User: {inp_text}\n\nAssistant:\n<output>\n"
        resp_str   = f"{out_text}\n</output><|endoftext|>"

    p_ids = tok.encode(prompt_str).ids
    r_ids = tok.encode(resp_str).ids
    full_ids = p_ids + r_ids

    # Strict zero-truncation filter: must fit 100% complete
    if len(full_ids) > MAX_SEQ_LEN:
        return False

    labels = [-100] * len(p_ids) + r_ids
    pad_len = MAX_SEQ_LEN - len(full_ids)
    if pad_len > 0:
        full_ids = full_ids + [50256] * pad_len
        labels   = labels + [-100] * pad_len

    k = f"{source_tag}_{hashlib.md5(inp_text.encode('utf-8')).hexdigest()[:12]}"

    all_inps.append(full_ids)
    all_lbls.append(labels)
    all_keys.append(k)

    jsonl_records.append({
        "input": inp_text,
        "think": th,
        "output": out_text,
        "source": source_tag
    })
    return True

t0 = time.time()

# 1. Sovereign Thinking Gold (13 samples)
p_sov = GOLD_DIR / "sovereign_thinking_gold.jsonl"
if p_sov.exists():
    c = 0
    with open(p_sov, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            d = json.loads(line)
            q = d["question"]
            r = d["response"]
            if "<think>" in r and "</think>" in r:
                th = r.split("</think>")[0].replace("<think>", "").strip()
                out = r.split("</think>")[1].replace("<output>", "").replace("</output>", "").strip()
            else:
                th = None
                out = r.strip()
            if process_sample(q, out, th, "sov_gold"): c += 1
    stats["sovereign_gold"] = c
    print(f"  [+] Sovereign Gold: {c} intact samples", flush=True)

# 2. Direct Answers Gold (1,200 samples)
p_da = GOLD_DIR / "Quillan_Direct_Answers_Gold.jsonl"
if p_da.exists():
    c = 0
    with open(p_da, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            d = json.loads(line)
            q = (d.get("prompt") or d.get("question", "")).replace("Question:", "").replace("Answer:", "").strip()
            out = (d.get("response") or d.get("answer", "")).replace("<output>", "").replace("</output>", "").strip()
            if process_sample(q, out, None, "direct_ans"): c += 1
    stats["direct_answers"] = c
    print(f"  [+] Direct Answers Gold: {c} intact samples", flush=True)

# 3. Universal Sovereign Gold (1,100 samples)
p_uni = GOLD_DIR / "Quillan_Universal_Sovereign_Gold_1000.jsonl"
if p_uni.exists():
    c = 0
    with open(p_uni, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            d = json.loads(line)
            q = d.get("question", "").strip()
            out = d.get("response", "").replace("<output>", "").replace("</output>", "").strip()
            if process_sample(q, out, None, "uni_gold"): c += 1
    stats["universal_gold"] = c
    print(f"  [+] Universal Sovereign Gold: {c} intact samples", flush=True)

# 4. Claude Opus Code (1,840 samples)
p_code = DATA_DIR / "code_train.jsonl"
if p_code.exists():
    c = 0
    with open(p_code, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            d = json.loads(line)
            msgs = d.get("messages", [])
            u = next((m["content"] for m in msgs if m.get("role") == "user"), "")
            a = next((m["content"] for m in msgs if m.get("role") == "assistant"), "")
            if "<think>" in a and "</think>" in a:
                th = a.split("</think>")[0].replace("<think>", "").strip()
                out = a.split("</think>")[1].replace("<output>", "").replace("</output>", "").strip()
                if process_sample(u, out, th, "claude_code"): c += 1
    stats["claude_code"] = c
    print(f"  [+] Claude Opus Code: {c} intact samples", flush=True)

# 5. Claude Opus Instruct (7,217 samples)
p_inst = DATA_DIR / "instruct_train.jsonl"
if p_inst.exists():
    c = 0
    with open(p_inst, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            d = json.loads(line)
            msgs = d.get("messages", [])
            u = next((m["content"] for m in msgs if m.get("role") == "user"), "")
            a = next((m["content"] for m in msgs if m.get("role") == "assistant"), "")
            if "<think>" in a and "</think>" in a:
                th = a.split("</think>")[0].replace("<think>", "").strip()
                out = a.split("</think>")[1].replace("<output>", "").replace("</output>", "").strip()
                if process_sample(u, out, th, "claude_inst"): c += 1
    stats["claude_instruct"] = c
    print(f"  [+] Claude Opus Instruct: {c} intact samples", flush=True)

# 6. GPT-5.5 Distilled (18,197 samples)
p_gpt = DATA_DIR / "GPT_5.5_Distilled.jsonl"
if p_gpt.exists():
    c_th = 0
    c_dir = 0
    with open(p_gpt, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            d = json.loads(line)
            t = d.get("text", "")
            if "<|assistant|>" not in t: continue
            parts = t.split("<|assistant|>")
            u = parts[0].replace("<|user|>", "").strip()
            a = parts[1].strip()
            if "<think>" in a and "</think>" in a:
                th = a.split("</think>")[0].replace("<think>", "").strip()
                out = a.split("</think>")[1].replace("<output>", "").replace("</output>", "").strip()
                if process_sample(u, out, th, "gpt_think"): c_th += 1
            else:
                out = a.replace("<output>", "").replace("</output>", "").strip()
                if process_sample(u, out, None, "gpt_direct"): c_dir += 1
    stats["gpt_think"] = c_th
    stats["gpt_direct"] = c_dir
    print(f"  [+] GPT-5.5 Distilled: {c_th} think + {c_dir} direct intact samples", flush=True)

total_samples = len(all_inps)
print("\n" + "=" * 70, flush=True)
print(f"  TOTAL INTACT UNIFIED SAMPLES: {total_samples:,}", flush=True)
print(f"  Extraction time: {time.time()-t0:.1f}s", flush=True)
print("=" * 70, flush=True)

# Write unified JSONL
print(f"\n[*] Writing clean unified JSONL to {OUT_JSONL.name}...", flush=True)
with open(OUT_JSONL, "w", encoding="utf-8") as f:
    for rec in jsonl_records:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
print(f"[+] Written {len(jsonl_records):,} records to {OUT_JSONL} ({OUT_JSONL.stat().st_size/1e6:.2f} MB)", flush=True)

# Stack contiguous tensors
print("[*] Stacking contiguous tensors (inputs and labels)...", flush=True)
inputs_tensor = torch.tensor(all_inps, dtype=torch.long)
labels_tensor = torch.tensor(all_lbls, dtype=torch.long)
print(f"[+] Shapes: inputs={inputs_tensor.shape}, labels={labels_tensor.shape}", flush=True)

# Save fast tokenized cache
print(f"[*] Saving fast cache to {TMP_CACHE.name}...", flush=True)
torch.save({
    "inputs": inputs_tensor,
    "labels": labels_tensor,
    "keys": all_keys,
    "count": total_samples,
    "max_seq_len": MAX_SEQ_LEN,
    "stats": stats,
    "timestamp": time.time(),
}, TMP_CACHE)

if OUT_CACHE.exists():
    OUT_CACHE.unlink()
TMP_CACHE.rename(OUT_CACHE)
print(f"[+] Successfully wrote {OUT_CACHE.name} ({OUT_CACHE.stat().st_size/1e6:.2f} MB) in {time.time()-t0:.1f}s total!", flush=True)
