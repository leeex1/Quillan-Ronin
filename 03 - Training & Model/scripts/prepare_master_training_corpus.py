#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QUILLAN-RONIN: Master Multi-Stage Dataset Standardizer & Verifier
================================================================
Unifies and validates the entire 200,000+ sample corpus across:
  - CrashOverrideX--Quillan_Samurai_sets (10 JSONL files)
  - canonical_standardized (quillan_sovereign_10_clean.jsonl)

Standardizes every sample into:
  1. Standard ChatML messages: [{'role': 'system'}, {'role': 'user'}, {'role': 'assistant'}]
  2. Full ChatML text string: <|im_start|>system...<|im_end|><|im_start|>user...<|im_end|><|im_start|>assistant...<|im_end|>
  3. Seamlessly compatible with Unsloth Desktop and native SFT trainers.
"""

import json
import os
import sys
import time
from pathlib import Path

# Force UTF-8 on Windows console
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

SYSTEM_PROMPT = (
    "You are Quillan-Ronin v5.4.0-ONI, a sovereign 34-expert Mixture-of-Experts AI assistant "
    "operating under the Bushido code of rectitude, honor, and rigorous technical truth."
)

BASE_HF = Path(r"C:\Users\Admin\.cache\huggingface\hub\datasets--CrashOverrideX--Quillan_Samurai_sets\snapshots\bd7cbfb5cfdfc7e65a6e580cad6383303739a4f6")
BASE_LOCAL = Path(r"C:\02_QUILLAN\03 - Training & Model\training_data\canonical_standardized")
OUT_DIR = Path(r"C:\02_QUILLAN\03 - Training & Model\training_data")
OUT_FILE = OUT_DIR / "quillan_master_unified_200k.jsonl"


def format_chatml(system_text: str, user_text: str, assistant_text: str) -> str:
    parts = []
    if system_text:
        parts.append(f"<|im_start|>system\n{system_text.strip()}<|im_end|>")
    parts.append(f"<|im_start|>user\n{user_text.strip()}<|im_end|>")
    parts.append(f"<|im_start|>assistant\n{assistant_text.strip()}<|im_end|>")
    return "\n".join(parts)


def parse_sample(source_name: str, d: dict):
    """Normalizes any source dictionary into (messages, text) or None if invalid."""
    system = SYSTEM_PROMPT
    user = ""
    assistant = ""

    # 1. Direct 'messages' list (code_train, full_train, instruct_train)
    if "messages" in d and isinstance(d["messages"], list):
        msgs = d["messages"]
        extracted_sys = None
        extracted_user = None
        extracted_asst = None
        for m in msgs:
            role = m.get("role", "")
            content = m.get("content", "")
            if role == "system" and not extracted_sys:
                extracted_sys = content
            elif role == "user" and not extracted_user:
                extracted_user = content
            elif role == "assistant" and not extracted_asst:
                extracted_asst = content
        if extracted_user and extracted_asst:
            system = extracted_sys or SYSTEM_PROMPT
            user = extracted_user
            assistant = extracted_asst
            chatml_text = format_chatml(system, user, assistant)
            return {
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                    {"role": "assistant", "content": assistant},
                ],
                "text": chatml_text,
                "source": source_name,
            }

    # 2. 'original_input' + 'model_thoughts' + 'model_response' (full_dataset.jsonl)
    if "original_input" in d and "model_response" in d:
        raw_in = d.get("original_input", "")
        if isinstance(raw_in, dict):
            parts = []
            for k, v in raw_in.items():
                parts.append(f"{k.capitalize()}: {v}")
            user = "\n".join(parts)
        elif isinstance(raw_in, str):
            user = raw_in.strip()
        else:
            user = str(raw_in).strip()
        resp = d.get("model_response", "")
        resp = resp.strip() if isinstance(resp, str) else str(resp).strip()
        thoughts = d.get("model_thoughts", "")
        thoughts = thoughts.strip() if isinstance(thoughts, str) else str(thoughts).strip()
        if user and resp:
            if thoughts:
                assistant = f"<think>\n{thoughts}\n</think>\n\n{resp}"
            else:
                assistant = resp
            return {
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                    {"role": "assistant", "content": assistant},
                ],
                "text": format_chatml(system, user, assistant),
                "source": source_name,
            }

    # 3. 'question' + 'reasoning_trace' / 'answer' (quillan_12mb, seed_dataset, train.jsonl)
    if "question" in d:
        user = d.get("question", "").strip()
        trace = d.get("reasoning_trace", "").strip()
        ans = d.get("answer", "") or d.get("response", "")
        if isinstance(ans, str):
            ans = ans.strip()
        else:
            ans = str(ans).strip()
        if user and (ans or trace):
            if trace and ans:
                assistant = f"<think>\n{trace}\n</think>\n\n{ans}"
            elif trace:
                assistant = f"<think>\n{trace}\n</think>"
            else:
                assistant = ans
            return {
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                    {"role": "assistant", "content": assistant},
                ],
                "text": format_chatml(system, user, assistant),
                "source": source_name,
            }

    # 4. 'prompt' + 'response' (quillan_sovereign_10_clean.jsonl)
    if "prompt" in d and "response" in d:
        raw_p = d.get("prompt", "")
        raw_r = d.get("response", "")
        if "<|im_start|>user" in raw_p:
            # Strip ChatML wrappers if already embedded in prompt
            clean_user = raw_p.replace("<|im_start|>user\n", "").replace("<|im_end|>\n<|im_start|>assistant\n", "").strip()
        else:
            clean_user = raw_p.strip()
        clean_resp = raw_r.replace("<|im_end|>", "").strip()
        if clean_user and clean_resp:
            return {
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": clean_user},
                    {"role": "assistant", "content": clean_resp},
                ],
                "text": format_chatml(system, clean_user, clean_resp),
                "source": source_name,
            }

    # 5. Direct 'text' corpus (quillan_corpus_CLEAN_V7, GPT_5.5_Distilled, science)
    if "text" in d and isinstance(d["text"], str):
        full_text = d["text"].strip()
        if len(full_text) > 30:
            # Check if it already has ChatML structure
            if "<|im_start|>user" in full_text and "<|im_start|>assistant" in full_text:
                return {
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": "Analyze and solve:"},
                        {"role": "assistant", "content": full_text},
                    ],
                    "text": full_text,
                    "source": source_name,
                }
            # Natural text pretraining chunk
            return {
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": "Explain the following technical knowledge comprehensively:"},
                    {"role": "assistant", "content": full_text},
                ],
                "text": format_chatml(system, "Explain the following technical knowledge comprehensively:", full_text),
                "source": source_name,
            }

    return None


def main():
    print("=" * 80)
    print("⚔️ QUILLAN-RONIN: FULL 200,000+ SAMPLE CORPUS UNIFICATION")
    print("=" * 80)
    t0 = time.time()

    sources = [
        # CrashOverrideX HF Cache
        BASE_HF / "quillan_corpus_CLEAN_V7.jsonl",
        BASE_HF / "train.jsonl",
        BASE_HF / "GPT_5.5_Distilled.jsonl",
        BASE_HF / "quillan_12mb_training_dataset.jsonl",
        BASE_HF / "full_train.jsonl",
        BASE_HF / "instruct_train.jsonl",
        BASE_HF / "full_dataset.jsonl",
        BASE_HF / "quillan_science_additional.jsonl",
        BASE_HF / "code_train.jsonl",
        BASE_HF / "quillan_science_absolute.jsonl",
        BASE_HF / "Quillan_Ronin_v5.3.1_Samurai_Training_Seed_Dataset.jsonl",
        # Local Standardized
        BASE_LOCAL / "quillan_sovereign_10_clean.jsonl",
    ]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    total_valid = 0
    total_skipped = 0

    with open(OUT_FILE, "w", encoding="utf-8") as out_fp:
        for src in sources:
            if not src.exists():
                print(f"[-] Missing: {src.name}")
                continue

            src_valid = 0
            src_skipped = 0
            file_size_mb = src.stat().st_size / (1024 * 1024)

            with open(src, "r", encoding="utf-8", errors="ignore") as in_fp:
                for line in in_fp:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        parsed = parse_sample(src.name, data)
                        if parsed:
                            out_fp.write(json.dumps(parsed, ensure_ascii=False) + "\n")
                            src_valid += 1
                        else:
                            src_skipped += 1
                    except Exception:
                        src_skipped += 1

            total_valid += src_valid
            total_skipped += src_skipped
            print(f"[+] {src.name:<45} | Valid: {src_valid:>8,} | Skipped: {src_skipped:>5,} | ({file_size_mb:>6.1f} MB)")

    elapsed = time.time() - t0
    out_size_mb = OUT_FILE.stat().st_size / (1024 * 1024)
    print("=" * 80)
    print(f"✓ Master Unified Dataset Created: {OUT_FILE}")
    print(f"  Total Valid Samples: {total_valid:,}")
    print(f"  Total Skipped      : {total_skipped:,}")
    print(f"  Total Output Size  : {out_size_mb:.2f} MB")
    print(f"  Elapsed Time       : {elapsed:.2f}s")
    print("=" * 80)


if __name__ == "__main__":
    main()
