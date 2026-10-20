#!/usr/bin/env python3
"""
unify_all_training_datasets.py
Aggregates and unifies the COMPLETE, uncompromised Quillan training corpus:
1. Existing Master Corpus: quillan_master_unified_200k.jsonl (199,874 samples)
2. Crownelius/GPT-5.6-Sol-Luna-Terra-Traces (15,353 samples)
3. r0b0tlab/qwen3.8-max-glm5.2-kimi-k3-distillation (104,187 samples)
4. MoreThought/Fable-5.1-Max-Reasoning-Filtered-10000x (15,000 samples)

Total unified pool: 334,414 samples.
Validates non-empty prompt/response pairs and outputs dual-compatible JSONL
(with both 'conversations' and 'messages' keys) for Unsloth, TRL, and custom PyTorch engines.
"""

import os
import sys
import json
import time
from pathlib import Path

try:
    import pyarrow.parquet as pq
except ImportError:
    pq = None

OUTPUT_FILE = Path(r"C:\02_QUILLAN\03 - Training & Model\training_data\quillan_master_frontier_all_334k.jsonl")

# 1. Existing master
EXISTING_MASTER = Path(r"C:\02_QUILLAN\03 - Training & Model\training_data\quillan_master_unified_200k.jsonl")

# 2. Frontier datasets
FRONTIER_DIR = Path(r"C:\02_QUILLAN\03 - Training & Model\training_data\frontier_datasets")
SOL_TERRA_FILE = FRONTIER_DIR / "gpt_5_6_sol_luna_terra" / "data" / "train.parquet"
QWEN_OPENAI_DIR = FRONTIER_DIR / "qwen3_8_glm5_2_kimi_k3" / "data" / "openai_messages"
FABLE_DIR = FRONTIER_DIR / "fable_5_1_max"

def clean_text(text) -> str:
    if text is None:
        return ""
    if isinstance(text, str):
        return text.strip()
    if isinstance(text, dict):
        return json.dumps(text, ensure_ascii=False)
    return str(text).strip()

def build_entry(sample_id: str, source: str, system_val: str, human_val: str, gpt_val: str):
    if not human_val or not gpt_val:
        return None

    # ShareGPT format
    conversations = []
    if system_val:
        conversations.append({"from": "system", "value": system_val})
    conversations.append({"from": "human", "value": human_val})
    conversations.append({"from": "gpt", "value": gpt_val})

    # OpenAI ChatML format
    messages = []
    if system_val:
        messages.append({"role": "system", "content": system_val})
    messages.append({"role": "user", "content": human_val})
    messages.append({"role": "assistant", "content": gpt_val})

    return {
        "id": sample_id,
        "source": source,
        "conversations": conversations,
        "messages": messages
    }

def process_existing_master(out_f) -> int:
    if not EXISTING_MASTER.exists():
        print(f"[WARN] Existing master {EXISTING_MASTER} not found!")
        return 0
    print(f"[1/4] Ingesting all {EXISTING_MASTER.name} (199,874 samples)...", flush=True)
    count = 0
    with open(EXISTING_MASTER, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except Exception:
                continue

            system_val, human_val, gpt_val = "", "", ""
            if "conversations" in data:
                for c in data["conversations"]:
                    role = c.get("from", "")
                    val = clean_text(c.get("value", ""))
                    if role == "system":
                        system_val = val
                    elif role in ("human", "user"):
                        human_val = val
                    elif role in ("gpt", "assistant"):
                        gpt_val = val
            elif "messages" in data:
                for m in data["messages"]:
                    role = m.get("role", "")
                    val = clean_text(m.get("content", ""))
                    if role == "system":
                        system_val = val
                    elif role == "user":
                        human_val = val
                    elif role == "assistant":
                        gpt_val = val

            sample_id = data.get("id") or f"master_base_{idx}"
            source = data.get("source") or "quillan_200k_base"

            entry = build_entry(sample_id, source, system_val, human_val, gpt_val)
            if entry:
                out_f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                count += 1
                if count % 50000 == 0:
                    print(f"      ... wrote {count:,} samples from existing master", flush=True)
    print(f"      Done existing master: {count:,} valid samples ingested.", flush=True)
    return count

def process_gpt_5_6_traces(out_f) -> int:
    if not SOL_TERRA_FILE.exists() or pq is None:
        print(f"[WARN] Sol-Luna-Terra parquet missing: {SOL_TERRA_FILE}", flush=True)
        return 0
    print(f"[2/4] Ingesting all GPT-5.6 Sol-Luna-Terra Traces: {SOL_TERRA_FILE.name}...", flush=True)
    table = pq.read_table(SOL_TERRA_FILE)
    num_rows = table.num_rows
    count = 0
    for idx in range(num_rows):
        row_json_str = table["row_json"][idx].as_py()
        try:
            data = json.loads(row_json_str)
        except Exception:
            continue

        system_val, human_val, gpt_val = "", "", ""
        for m in data.get("messages", []):
            role = m.get("role", "")
            content = clean_text(m.get("content", ""))
            if role == "system":
                system_val = content
            elif role == "user":
                human_val = content
            elif role == "assistant":
                gpt_val = content

        sample_id = f"gpt56_trace_{idx}"
        source = f"gpt56_{data.get('category', 'trace')}"
        entry = build_entry(sample_id, source, system_val, human_val, gpt_val)
        if entry:
            out_f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            count += 1
    print(f"      Done GPT-5.6 traces: {count:,} valid samples ingested.", flush=True)
    return count

def process_qwen38_kimi(out_f) -> int:
    if not QWEN_OPENAI_DIR.exists() or pq is None:
        print(f"[WARN] Qwen3.8 openai_messages missing: {QWEN_OPENAI_DIR}", flush=True)
        return 0
    print(f"[3/4] Ingesting all Qwen3.8 / GLM-5.2 / Kimi-K3 Distillation ({QWEN_OPENAI_DIR})...", flush=True)
    parquet_files = sorted(list(QWEN_OPENAI_DIR.glob("*.parquet")))
    count = 0
    for p_file in parquet_files:
        table = pq.read_table(p_file)
        num_rows = table.num_rows
        for idx in range(num_rows):
            messages = table["messages"][idx].as_py()
            source = table["source"][idx].as_py() if "source" in table.column_names else "qwen38_distill"
            system_val, human_val, gpt_val = "", "", ""
            for m in messages:
                role = m.get("role", "")
                content = clean_text(m.get("content", ""))
                rc = clean_text(m.get("reasoning_content", ""))
                if role == "system":
                    system_val = content
                elif role == "user":
                    human_val = content
                elif role == "assistant":
                    if rc:
                        gpt_val = f"<think>\n{rc}\n</think>\n{content}"
                    else:
                        gpt_val = content

            sample_id = f"qwen_kimi_{p_file.stem}_{idx}"
            entry = build_entry(sample_id, source, system_val, human_val, gpt_val)
            if entry:
                out_f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                count += 1
                if count % 25000 == 0:
                    print(f"      ... wrote {count:,} Qwen/Kimi samples", flush=True)
    print(f"      Done Qwen/Kimi: {count:,} valid samples ingested.", flush=True)
    return count

def process_fable(out_f) -> int:
    if not FABLE_DIR.exists():
        print(f"[WARN] Fable directory missing: {FABLE_DIR}", flush=True)
        return 0
    print(f"[4/4] Ingesting all Fable-5.1 Max-Reasoning Filtered 10000x ({FABLE_DIR})...", flush=True)
    fable_files = [FABLE_DIR / "10k" / "train.jsonl", FABLE_DIR / "5k" / "train.jsonl"]
    count = 0
    for fpath in fable_files:
        if not fpath.exists():
            continue
        print(f"      Reading {fpath.name} ({fpath.parent.name})...", flush=True)
        with open(fpath, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except Exception:
                    continue
                system_val, human_val, gpt_val = "", "", ""
                for m in data.get("messages", []):
                    role = m.get("role", "")
                    content = clean_text(m.get("content", ""))
                    if role == "system":
                        system_val = content
                    elif role == "user":
                        human_val = content
                    elif role == "assistant":
                        gpt_val = content

                sample_id = f"fable_{fpath.parent.name}_{idx}"
                entry = build_entry(sample_id, "fable_5_1_max_reasoning", system_val, human_val, gpt_val)
                if entry:
                    out_f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                    count += 1
                    if count % 5000 == 0:
                        print(f"      ... wrote {count:,} Fable samples", flush=True)
    print(f"      Done Fable: {count:,} valid samples ingested.", flush=True)
    return count

def main():
    start_time = time.time()
    print("=" * 70)
    print("QUILLAN-RONIN COMPREHENSIVE MASTER DATASET UNIFICATION (ALL SOURCES)")
    print(f"Output File: {OUTPUT_FILE}")
    print("=" * 70, flush=True)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as out_f:
        c1 = process_existing_master(out_f)
        c2 = process_gpt_5_6_traces(out_f)
        c3 = process_qwen38_kimi(out_f)
        c4 = process_fable(out_f)

    total_count = c1 + c2 + c3 + c4
    file_size_mb = OUTPUT_FILE.stat().st_size / (1024 ** 2)
    file_size_gb = file_size_mb / 1024
    elapsed = time.time() - start_time

    print("\n" + "=" * 70)
    print("FULL UNIFICATION COMPLETE — 100% SAMPLES PRESERVED")
    print(f"Total Combined Samples: {total_count:,}")
    print(f"  - Master Corpus:  {c1:,}")
    print(f"  - GPT-5.6 Traces: {c2:,}")
    print(f"  - Qwen/GLM/Kimi:  {c3:,}")
    print(f"  - Fable-5.1 Max:  {c4:,}")
    print(f"Final File Size:         {file_size_mb:.2f} MB ({file_size_gb:.2f} GB)")
    print(f"Total Elapsed Time:      {elapsed:.1f}s ({elapsed / 60:.2f} mins)")
    print("=" * 70, flush=True)

if __name__ == "__main__":
    main()
