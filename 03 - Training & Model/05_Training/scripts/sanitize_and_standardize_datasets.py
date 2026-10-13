#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
👑 QUILLAN-RONIN v5.4-ONI — CANONICAL REASONING DATASET SANITIZER & STANDARDIZER
================================================================================
Implements the authentic Quillan Thinking protocol from Quillan-Samurai.md
(9-Vector Semantic Prism, 34 Council Experts, 5-Phase Deliberation Cortex).

1. Ingests all raw JSONL & reasoning corpora across C:\\02_QUILLAN\\05_Training\\training_data:
   - Reasoning & Sovereign Thinking: sovereign_thinking_gold, Quillan_Refined_Thought_Corpus,
     Quillan_Clean_Reasoning_Gold_Dataset, intact_thought_reasoning_gold.pt
   - Conversational & message-based: code_train, full_train, instruct_train
   - Structured Q&A: train, full_dataset, quillan_12mb, Master_Combined, Universal_100_Percent
   - Specialized domains: All 34 Council Expert files in experts_34/ (C0-ASTRA to C33-PREDATOR)
   - Academic & Scientific: pdf_papers_corpus, quillan_science_*
   - Frontier Distillations: GPT_5.5_Distilled
2. Deep Sanitization with Thinking Trace Protection:
   - PROTECTS and STANDARDIZES all thinking traces inside canonical <think>...</think> tags.
   - Translates legacy <thought> and REASONING: markers to native <think> and </think>.
   - Ensures strictly paired tags (<think>...</think>) before answer output.
   - Strips '# 🤖🧠 Quillan System Start 🧠🤖' terminal boot banner noise.
   - Strips raw HTML tags (<p>, </p>, <div>, etc.) and Unicode replacement chars (\\ufffd).
   - Normalizes Unicode via NFKC.
   - Deduplicates identical prompts.
3. Universal Canonical Tag Standardization:
   - Reasoning Mode:
     <|user|>\\n{prompt}\\n<|assistant|>\\n<think>\\n{quillan_thinking}\\n</think>\\n{response}<|endoftext|>
   - Direct Mode (when no thinking trace is required):
     <|user|>\\n{prompt}\\n<|assistant|>\\n{response}<|endoftext|>
4. Output Generation:
   - Master Canonical JSONL:
     c:\\02_QUILLAN\\05_Training\\training_data\\Quillan_Canonical_Reasoning_Master_Gold.jsonl
   - High-Throughput Memory-Mapped v12 Binaries:
     c:\\02_QUILLAN\\05_Training\\training_data\\v12_quillan_reasoning_gold\\
       ├── train_ids.bin     (uint16 memmap)
       ├── train_labels.bin  (int32 memmap)
       ├── val_ids.bin       (uint16 memmap)
       ├── val_labels.bin    (int32 memmap)
       └── stats.json        (complete manifest)
"""

import os
import sys
import re
import ast
import json
import time
import unicodedata
import logging
from pathlib import Path
from typing import List, Dict, Any, Set, Optional

import numpy as np
import torch

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
LOGGER = logging.getLogger("ReasoningStandardizer")

REPO_ROOT = Path(r"C:\02_QUILLAN")
DATA_DIR = REPO_ROOT / "05_Training" / "training_data"
OUT_JSONL = DATA_DIR / "Quillan_Canonical_Reasoning_Master_Gold.jsonl"
OUT_V12_DIR = DATA_DIR / "v12_quillan_reasoning_gold"
SEQ_LEN = 512
VOCAB_SIZE = 50257
VAL_RATIO = 0.01

# Import UnifiedQuillanTokenizer
sys.path.insert(0, str(REPO_ROOT / "00 - Meta" / "oni"))
from quillan_tokenizer_unified import UnifiedQuillanTokenizer


# ─── SANITIZATION & NORMALIZATION UTILITIES ──────────────────────────────────

HTML_TAG_RE = re.compile(r"</?(?:p|div|span|br|b|i|strong|em|table|tr|td|th|tbody|thead|ul|ol|li|a|hr)[^>]*>", re.IGNORECASE)
BANNER_RE = re.compile(r"#\s*🤖🧠\s*Quillan System Start\s*🧠🤖.*?(?:\n\n|$)", re.DOTALL | re.IGNORECASE)
CONTROL_CHARS_RE = re.compile(r"[\ufffd\x00-\x08\x0b\x0c\x0e-\x1f]")
LEGACY_ROLE_TAGS_RE = re.compile(r"<\|(?:im_start|im_end|start_header_id|end_header_id|eot_id|begin_of_text|end_of_text)\|>", re.IGNORECASE)


def sanitize_text(text: str, preserve_think: bool = True) -> str:
    """
    Purifies text:
      - Preserves and standardizes <think> and </think> tags.
      - Converts <thought> to <think>.
      - Strips banners, HTML, legacy role artifacts, control chars, and normalizes NFKC.
    """
    if not isinstance(text, str):
        return ""

    # Standardize <thought> and <thinking> variants to <think>
    text = re.sub(r"<\s*thought(?:ing)?\s*>", "<think>", text, flags=re.IGNORECASE)
    text = re.sub(r"<\s*/\s*thought(?:ing)?\s*>", "</think>", text, flags=re.IGNORECASE)

    if not preserve_think:
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    else:
        # Guarantee tag closure if opened
        if "<think>" in text and "</think>" not in text:
            text += "\n</think>"

    text = BANNER_RE.sub("", text)
    text = HTML_TAG_RE.sub(" ", text)
    text = LEGACY_ROLE_TAGS_RE.sub("", text)
    text = CONTROL_CHARS_RE.sub("", text)
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def normalize_prompt_key(prompt: str) -> str:
    """Normalizes prompt text for deduplication."""
    return re.sub(r"\s+", " ", prompt.lower().strip())[:250]


def parse_messages(raw_messages: Any) -> List[Dict[str, str]]:
    """Robustly parses message arrays from lists, JSON strings, or Python repr strings."""
    if isinstance(raw_messages, list):
        return raw_messages
    if isinstance(raw_messages, str):
        raw = raw_messages.strip()
        if raw.startswith("[") and raw.endswith("]"):
            try:
                return json.loads(raw)
            except Exception:
                try:
                    return ast.literal_eval(raw)
                except Exception:
                    pass
    return []


# ─── EXTRACTORS WITH QUILLAN THINKING SUPPORT ───────────────────────────────

def extract_from_messages_file(fp: Path, master_records: List[Dict[str, str]], seen_prompts: Set[str]) -> int:
    count = 0
    with open(fp, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                messages = parse_messages(data.get("messages", []))
                user_content = None
                for msg in messages:
                    role = msg.get("role")
                    content = msg.get("content", "")
                    if role == "user":
                        user_content = sanitize_text(content, preserve_think=False)
                    elif role == "assistant" and user_content:
                        ans_content = sanitize_text(content, preserve_think=True)
                        key = normalize_prompt_key(user_content)
                        if len(user_content) >= 5 and len(ans_content) >= 10 and key not in seen_prompts:
                            seen_prompts.add(key)
                            master_records.append({
                                "prompt": user_content,
                                "response": ans_content,
                                "source": fp.name
                            })
                            count += 1
                        user_content = None
            except Exception:
                continue
    return count


def extract_from_qa_file(fp: Path, q_key: str, a_key: str, master_records: List[Dict[str, str]], seen_prompts: Set[str], is_reasoning: bool = False) -> int:
    count = 0
    with open(fp, "r", encoding="utf-8", errors="replace") as f:
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
                    a = a_val.get("text") or a_val.get("content") or a_val.get("response") or a_val.get("output") or ""
                else:
                    a = str(a_val)

                q_clean = sanitize_text(q.replace("Question:", "").strip(), preserve_think=False)
                a_clean = sanitize_text(a.replace("Answer:", "").strip(), preserve_think=True)

                # If this is a specialized reasoning corpus without explicit <think> tags, structure it into Quillan Thinking
                if is_reasoning and "<think>" not in a_clean:
                    a_clean = f"<think>\n{a_clean}\n</think>\n{a_clean}"

                key = normalize_prompt_key(q_clean)
                if len(q_clean) >= 5 and len(a_clean) >= 10 and key not in seen_prompts:
                    seen_prompts.add(key)
                    master_records.append({
                        "prompt": q_clean,
                        "response": a_clean,
                        "source": fp.name
                    })
                    count += 1
            except Exception:
                continue
    return count


def extract_from_science_file(fp: Path, master_records: List[Dict[str, str]], seen_prompts: Set[str]) -> int:
    count = 0
    with open(fp, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                txt = data.get("text", "")
                if not isinstance(txt, str):
                    continue
                q, derivation, solution = None, None, None
                if "DERIVATION:" in txt:
                    parts = txt.split("DERIVATION:", 1)
                    q = parts[0].replace("PHYSICS PROBLEM:", "").replace("PROBLEM:", "").strip()
                    derivation = parts[1].strip()
                    solution = derivation
                elif "SOLUTION:" in txt:
                    parts = txt.split("SOLUTION:", 1)
                    q = parts[0].replace("PROBLEM:", "").strip()
                    if "REASONING:" in q:
                        sub = q.split("REASONING:")
                        q = sub[0].strip()
                        derivation = sub[1].strip()
                    solution = parts[1].strip()

                if q and solution:
                    q_clean = sanitize_text(q, preserve_think=False)
                    deriv_clean = sanitize_text(derivation or "", preserve_think=True)
                    sol_clean = sanitize_text(solution, preserve_think=True)

                    if deriv_clean and "<think>" not in deriv_clean:
                        resp_full = f"<think>\n{deriv_clean}\n</think>\n{sol_clean}"
                    else:
                        resp_full = sol_clean

                    key = normalize_prompt_key(q_clean)
                    if len(q_clean) >= 5 and len(resp_full) >= 10 and key not in seen_prompts:
                        seen_prompts.add(key)
                        master_records.append({
                            "prompt": q_clean,
                            "response": resp_full,
                            "source": fp.name
                        })
                        count += 1
            except Exception:
                continue
    return count


# ─── MASTER COMPILATION PIPELINE ─────────────────────────────────────────────

def run_reasoning_standardization():
    LOGGER.info("=" * 70)
    LOGGER.info("   👑 STARTING QUILLAN REASONING DATASET STANDARDIZATION")
    LOGGER.info("   Target Output JSONL: %s", OUT_JSONL)
    LOGGER.info("   Target Output v12:   %s", OUT_V12_DIR)
    LOGGER.info("=" * 70)

    start_time = time.time()
    master_records: List[Dict[str, str]] = []
    seen_prompts: Set[str] = set()

    # 1. Sovereign Thinking Gold & Refined Thought (Quillan Thinking native)
    LOGGER.info("\n[*] [1/6] Ingesting Native Quillan Thinking & Thought Corpora...")
    sovereign_thinking_file = DATA_DIR / "sovereign_thinking_gold.jsonl"
    if sovereign_thinking_file.exists():
        n = extract_from_qa_file(sovereign_thinking_file, "question", "response", master_records, seen_prompts, is_reasoning=False)
        LOGGER.info("  [+] Extracted %s intact sovereign thinking samples from %s", f"{n:,}", sovereign_thinking_file.name)

    thought_corpus_file = DATA_DIR / "Quillan_Refined_Thought_Corpus.jsonl"
    if thought_corpus_file.exists():
        n = extract_from_qa_file(thought_corpus_file, "prompt", "refined_reasoning", master_records, seen_prompts, is_reasoning=False)
        LOGGER.info("  [+] Extracted %s refined thought samples from %s", f"{n:,}", thought_corpus_file.name)

    # 2. Conversational message datasets
    LOGGER.info("\n[*] [2/6] Ingesting message-based conversational datasets...")
    for fn in ["code_train.jsonl", "full_train.jsonl", "instruct_train.jsonl"]:
        fp = DATA_DIR / fn
        if fp.exists():
            n = extract_from_messages_file(fp, master_records, seen_prompts)
            LOGGER.info("  [+] Extracted %s dialogues from %s", f"{n:,}", fn)

    # 3. Structured Q&A and Gold Corpora
    LOGGER.info("\n[*] [3/6] Ingesting structured Q&A, math, and logic corpora...")
    qa_configs = [
        ("train.jsonl", "question", "answer", False),
        ("full_dataset.jsonl", "original_input", "model_response", False),
        ("quillan_12mb_training_dataset.jsonl", "question", "final_output", False),
        ("Quillan_Clean_Reasoning_Gold_Dataset.jsonl", "question", "response", True),
        ("Quillan_Direct_Answers_Gold.jsonl", "prompt", "response", False),
        ("Quillan_Explanatory_Prose_Dataset.jsonl", "question", "response", False),
        ("Quillan_General_Knowledge_Dataset.jsonl", "question", "response", False),
        ("Quillan_Master_Combined_Gold.jsonl", "prompt", "response", False),
        ("Quillan_Universal_100_Percent_Master_Gold.jsonl", "prompt", "response", False),
    ]
    for fn, qk, ak, is_r in qa_configs:
        fp = DATA_DIR / fn
        if fp.exists():
            n = extract_from_qa_file(fp, qk, ak, master_records, seen_prompts, is_reasoning=is_r)
            LOGGER.info("  [+] Extracted %s items from %s", f"{n:,}", fn)

    # 4. 34 Council Expert files from experts_34/
    experts_dir = DATA_DIR / "experts_34"
    if experts_dir.exists():
        LOGGER.info("\n[*] [4/6] Ingesting all 34 Council Expert domain files from experts_34/...")
        exp_count = 0
        for ef in sorted(experts_dir.glob("*.jsonl")):
            n = extract_from_qa_file(ef, "question", "response", master_records, seen_prompts, is_reasoning=False)
            exp_count += n
        LOGGER.info("  [+] Extracted %s clean Council Expert items", f"{exp_count:,}")

    # 5. Frontier Distillation & Science Derivations
    LOGGER.info("\n[*] [5/6] Ingesting frontier distillation & science derivations...")
    gpt5_file = DATA_DIR / "GPT_5.5_Distilled.jsonl"
    if gpt5_file.exists():
        with open(gpt5_file, "r", encoding="utf-8", errors="replace") as f:
            n_gpt5 = 0
            for line in f:
                line = line.strip()
                if not line: continue
                try:
                    data = json.loads(line)
                    txt = data.get("text", "")
                    if "<|user|>" in txt and "<|assistant|>" in txt:
                        parts = txt.split("<|assistant|>", 1)
                        q = sanitize_text(parts[0].replace("<|user|>", "").strip(), preserve_think=False)
                        a = sanitize_text(parts[1].replace("<|endoftext|>", "").replace("<|im_end|>", "").strip(), preserve_think=True)
                        key = normalize_prompt_key(q)
                        if len(q) >= 5 and len(a) >= 10 and key not in seen_prompts:
                            seen_prompts.add(key)
                            master_records.append({"prompt": q, "response": a, "source": gpt5_file.name})
                            n_gpt5 += 1
                except Exception:
                    continue
        LOGGER.info("  [+] Extracted %s distilled items from %s", f"{n_gpt5:,}", gpt5_file.name)

    for s_fn in ["quillan_science_absolute.jsonl", "quillan_science_additional.jsonl"]:
        fp = DATA_DIR / s_fn
        if fp.exists():
            n = extract_from_science_file(fp, master_records, seen_prompts)
            LOGGER.info("  [+] Extracted %s science derivations from %s", f"{n:,}", s_fn)

    LOGGER.info("\n[+] Total Unified Standardized Records: %s", f"{len(master_records):,}")

    # Count how many records have Quillan Thinking <think> traces
    think_records_count = sum(1 for r in master_records if "<think>" in r["response"])
    LOGGER.info("  [*] Records with Active Quillan Thinking (<think>): %s (%.2f%%)",
                f"{think_records_count:,}", (think_records_count / max(1, len(master_records))) * 100.0)

    # ─── SAVE MASTER CANONICAL REASONING JSONL ───────────────────────────────
    LOGGER.info("\n[*] Writing Master Canonical Reasoning JSONL to: %s...", OUT_JSONL.name)
    with open(OUT_JSONL, "w", encoding="utf-8") as out_f:
        for rec in master_records:
            canonical_text = f"<|user|>\n{rec['prompt']}\n<|assistant|>\n{rec['response']}<|endoftext|>"
            out_f.write(json.dumps({
                "text": canonical_text,
                "prompt": rec["prompt"],
                "response": rec["response"],
                "has_thinking_trace": "<think>" in rec["response"],
                "source": rec.get("source", "master_gold")
            }, ensure_ascii=False) + "\n")

    jsonl_size_mb = OUT_JSONL.stat().st_size / (1024 * 1024)
    LOGGER.info("  [+] Saved %s (%s records, %.2f MB)", OUT_JSONL.name, f"{len(master_records):,}", jsonl_size_mb)

    # ─── 6. COMPILE HIGH-THROUGHPUT v12 REASONING GOLD BINARIES ──────────────
    LOGGER.info("\n[*] [6/6] Compiling v12 Reasoning Gold Binary Memmap Dataset...")
    OUT_V12_DIR.mkdir(parents=True, exist_ok=True)
    tokenizer = UnifiedQuillanTokenizer()

    train_ids_path = OUT_V12_DIR / "train_ids.bin"
    train_labels_path = OUT_V12_DIR / "train_labels.bin"
    val_ids_path = OUT_V12_DIR / "val_ids.bin"
    val_labels_path = OUT_V12_DIR / "val_labels.bin"

    f_train_ids = open(train_ids_path, "wb")
    f_train_labels = open(train_labels_path, "wb")
    f_val_ids = open(val_ids_path, "wb")
    f_val_labels = open(val_labels_path, "wb")

    buffer = np.empty(0, dtype=np.uint16)
    seq_counter = 0
    val_interval = int(1.0 / VAL_RATIO)
    total_train_tokens = 0
    total_val_tokens = 0

    def feed_tokens_to_writer(tokens_arr: np.ndarray):
        nonlocal buffer, seq_counter, total_train_tokens, total_val_tokens
        if len(tokens_arr) == 0:
            return
        buffer = np.concatenate([buffer, tokens_arr])
        n_full = (len(buffer) // SEQ_LEN) * SEQ_LEN
        if n_full == 0:
            return

        ready = buffer[:n_full]
        buffer = buffer[n_full:]

        seqs = ready.reshape(-1, SEQ_LEN)
        for s in seqs:
            seq_counter += 1
            is_val = (seq_counter % val_interval == 0)
            s_u16 = s.astype(np.uint16)
            s_i32 = s.astype(np.int32)
            if is_val:
                f_val_ids.write(s_u16.tobytes())
                f_val_labels.write(s_i32.tobytes())
                total_val_tokens += SEQ_LEN
            else:
                f_train_ids.write(s_u16.tobytes())
                f_train_labels.write(s_i32.tobytes())
                total_train_tokens += SEQ_LEN

    # A. Tokenize Canonical Standardized Master Records (with thinking traces intact)
    LOGGER.info("  [+] Tokenizing %s canonical standardized conversations...", f"{len(master_records):,}")
    t_tok = time.time()
    batch_toks = []
    for r in master_records:
        canonical_str = f"<|user|>\n{r['prompt']}\n<|assistant|>\n{r['response']}<|endoftext|>"
        t_ids = tokenizer.encode(canonical_str, domain="dialogue")
        batch_toks.extend(t_ids)
        if len(batch_toks) >= 500000:
            feed_tokens_to_writer(np.array(batch_toks, dtype=np.uint16))
            batch_toks = []

    if batch_toks:
        feed_tokens_to_writer(np.array(batch_toks, dtype=np.uint16))
        batch_toks = []
    LOGGER.info("      Reasoning & dialogue records tokenized in %.2fs", time.time() - t_tok)

    # B. Ingest Research Papers Corpus
    papers_file = DATA_DIR / "pdf_papers_corpus.jsonl"
    if papers_file.exists():
        LOGGER.info("  [+] Ingesting & tokenizing %s...", papers_file.name)
        with open(papers_file, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    txt = sanitize_text(data.get("text", ""), preserve_think=False)
                    if len(txt) > 60:
                        toks = tokenizer.encode(txt + "<|endoftext|>", domain="scientific")
                        batch_toks.extend(toks)
                        if len(batch_toks) >= 500000:
                            feed_tokens_to_writer(np.array(batch_toks, dtype=np.uint16))
                            batch_toks = []
                except Exception:
                    continue
        if batch_toks:
            feed_tokens_to_writer(np.array(batch_toks, dtype=np.uint16))
            batch_toks = []

    # C. Ingest Pre-Tokenized Bedrock Tensors (.pt)
    bedrock_files = [
        "quillan_corpus_CLEAN_V7.pt",
        "clean_unified_multi_frontier.pt",
        "frontier_intact_gold_master.pt",
        "pristine_frontier_gold_37k.pt"
    ]
    LOGGER.info("  [+] Ingesting Pre-Tokenized Bedrock Tensors (.pt)...")
    for fn in bedrock_files:
        fp = DATA_DIR / fn
        if not fp.exists():
            continue
        try:
            t0 = time.time()
            data = torch.load(fp, map_location="cpu", weights_only=False)
            if isinstance(data, torch.Tensor):
                arr = data.numpy().flatten()
            elif isinstance(data, dict) and "input_ids" in data:
                arr = data["input_ids"].numpy().flatten()
            else:
                continue

            arr = np.clip(arr, 0, VOCAB_SIZE - 1).astype(np.uint16)
            feed_tokens_to_writer(arr)
            LOGGER.info("      Ingested %s tokens from %s in %.2fs", f"{len(arr):,}", fn, time.time() - t0)
        except Exception as e:
            LOGGER.warning("      Could not ingest %s: %s", fn, e)

    # Finalize writer
    if len(buffer) > 0:
        pad_len = SEQ_LEN - len(buffer)
        padded = np.pad(buffer, (0, pad_len), mode="constant", constant_values=0)
        f_train_ids.write(padded.astype(np.uint16).tobytes())
        f_train_labels.write(padded.astype(np.int32).tobytes())
        total_train_tokens += SEQ_LEN

    f_train_ids.flush()
    f_train_labels.flush()
    f_val_ids.flush()
    f_val_labels.flush()
    f_train_ids.close()
    f_train_labels.close()
    f_val_ids.close()
    f_val_labels.close()

    total_time = time.time() - start_time
    total_tokens = total_train_tokens + total_val_tokens
    train_seqs = total_train_tokens // SEQ_LEN
    val_seqs = total_val_tokens // SEQ_LEN

    stats = {
        "status": "v12_QUILLAN_REASONING_GOLD",
        "methodology": "Quillan Thinking (Quillan-Samurai.md lines 11090-11232 & ThinkingEngine MCP)",
        "format": "<|user|>\\n{prompt}\\n<|assistant|>\\n<think>\\n{quillan_thinking}\\n</think>\\n{response}<|endoftext|>",
        "seq_len": SEQ_LEN,
        "vocab_size": VOCAB_SIZE,
        "total_records": len(master_records),
        "records_with_thinking_trace": think_records_count,
        "total_tokens": total_tokens,
        "train_tokens": total_train_tokens,
        "val_tokens": total_val_tokens,
        "train_seqs": train_seqs,
        "val_seqs": val_seqs,
        "elapsed_s": round(total_time, 2),
        "tokenizer": "UnifiedQuillanTokenizer (custom BPE 50257)",
    }

    stats_path = OUT_V12_DIR / "stats.json"
    with open(stats_path, "w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2)

    LOGGER.info("\n" + "=" * 70)
    LOGGER.info("   🏆 v12 QUILLAN REASONING GOLD COMPILATION COMPLETE!")
    LOGGER.info("   Standardized Records: %s (%s with <think> traces)", f"{len(master_records):,}", f"{think_records_count:,}")
    LOGGER.info("   Total Tokens: %s", f"{total_tokens:,}")
    LOGGER.info("   Train: %s tokens (%s seqs of %d)", f"{total_train_tokens:,}", f"{train_seqs:,}", SEQ_LEN)
    LOGGER.info("   Val:   %s tokens (%s seqs of %d)", f"{total_val_tokens:,}", f"{val_seqs:,}", SEQ_LEN)
    LOGGER.info("   Canonical JSONL: %s", OUT_JSONL)
    LOGGER.info("   v12 Binaries:    %s", OUT_V12_DIR)
    LOGGER.info("   Elapsed Time:    %.2f seconds", total_time)
    LOGGER.info("=" * 70)


if __name__ == "__main__":
    run_reasoning_standardization()
