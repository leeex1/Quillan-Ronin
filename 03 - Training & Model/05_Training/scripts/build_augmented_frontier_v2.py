#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
👑 QUILLAN-RONIN v5.3.1 — COMPREHENSIVE AUGMENTED DATASET BUILDER
=================================================================
Mines ALL available data sources on this machine and builds a
combined augmented 384-token pre-tokenized training corpus:

Sources:
  1. Quillan_Universal_100_Percent_Master_Gold.jsonl  (30,099 diverse Q&A samples)
  2. quillan_corpus_CLEAN_V7.jsonl                    (525 MB full clean corpus)
  3. pdf_papers_corpus.jsonl                          (academic papers)
  4. code_train.jsonl                                 (code samples)
  5. experts_34/*.jsonl                               (34 expert-specific datasets)
  6. Quillan_Refined_Thought_Corpus.jsonl             (prompt/refined_reasoning)
  7. full_train.jsonl                                 (68 MB general training)

Output: training_data/augmented_frontier_v2.pt
"""

import json
import sys
import random
import logging
from pathlib import Path

import torch

# ── Setup ─────────────────────────────────────────────────────────────────────
REPO_ROOT = Path(r"C:\02_QUILLAN")
DATA_DIR  = REPO_ROOT / "training_data"
OUT_PATH  = DATA_DIR / "augmented_frontier_v2.pt"

sys.path.insert(0, str(REPO_ROOT / "scripts"))

from sovereign_inference_engine import SovereignTokenizer

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
LOGGER = logging.getLogger("quillan.augmented_builder")

MAX_SEQ_LEN = 384
MIN_RESP_CHARS = 60
CAP_PER_FILE = 8000  # Max samples per source to prevent any single source dominating


def encode_pair(tokenizer: SovereignTokenizer, prompt: str, response: str, thinking: str = ""):
    """
    Encode a prompt/response pair into the canonical Quillan Sovereign Flow:
    <|start|>
    <|user|>
    {prompt}
    <|assistant|>
    <assistant_thinking>
    {thinking}
    </assistant_thinking>
    <assistant_response>
    {response}
    </assistant_response>
    <|end|>
    """
    p_text = f"<|start|>\n<|user|>\n{prompt.strip()}\n<|assistant|>\n"
    p_ids = tokenizer.encode(p_text)

    if thinking.strip():
        r_text = (
            f"<assistant_thinking>\n{thinking.strip()}\n</assistant_thinking>\n"
            f"<assistant_response>\n{response.strip()}\n</assistant_response>\n<|end|>"
        )
    else:
        r_text = f"<assistant_response>\n{response.strip()}\n</assistant_response>\n<|end|>"

    r_ids = tokenizer.encode(r_text)
    seq = p_ids + r_ids
    labels = [-100] * len(p_ids) + list(r_ids)

    # Clip if too long — preserve response portion
    if len(seq) > MAX_SEQ_LEN:
        resp_space = MAX_SEQ_LEN - len(p_ids)
        if resp_space < 15:
            return None  # Prompt alone exceeds budget
        seq = seq[:MAX_SEQ_LEN]
        labels = labels[:MAX_SEQ_LEN]

    # Pad
    pad = MAX_SEQ_LEN - len(seq)
    inp = seq + [50256] * pad
    lbl = labels + [-100] * pad

    # Require meaningful response content
    resp_count = sum(1 for l in lbl if l != -100)
    if resp_count < 8:
        return None

    return (
        torch.tensor(inp, dtype=torch.long),
        torch.tensor(lbl, dtype=torch.long),
    )


def load_jsonl_source(
    tokenizer: SovereignTokenizer,
    path: Path,
    prompt_keys: list,
    resp_keys: list,
    label: str,
    cap: int = CAP_PER_FILE,
    skip_thinking: bool = False,
) -> list:
    """Generic JSONL loader with configurable key priority and cap."""
    if not path.exists():
        LOGGER.warning("[SKIP] Not found: %s", path)
        return []

    samples = []
    seen: set = set()

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if len(samples) >= cap:
                break
            if not line.strip():
                continue
            try:
                d = json.loads(line)
                # Extract prompt
                q = ""
                for pk in prompt_keys:
                    if pk in d and d[pk]:
                        q = str(d[pk]).strip()
                        break
                # Extract response — check multiple keys
                r = ""
                for rk in resp_keys:
                    if rk in d and d[rk]:
                        r = str(d[rk]).strip()
                        break

                if not q or len(r) < MIN_RESP_CHARS:
                    continue

                # Skip Quillan-system-start headers in prompts (those are already in the JSONL datasets)
                if skip_thinking and "<think>" in r and "</think>" not in r:
                    continue  # Incomplete thinking block

                # Dedup by prompt prefix
                key = q[:80]
                if key in seen:
                    continue
                seen.add(key)

                pair = encode_pair(tokenizer, q, r)
                if pair:
                    samples.append(pair)
            except Exception:
                pass

    LOGGER.info("[+] %s: %d samples", label, len(samples))
    return samples


def build():
    tokenizer = SovereignTokenizer("gpt2")
    all_samples = []

    LOGGER.info("Building augmented frontier v2 dataset (MAX_SEQ_LEN=%d)...", MAX_SEQ_LEN)

    # ── Source 1: Universal 100% Master Gold (30k diverse Q&A) ───────────────
    s = load_jsonl_source(
        tokenizer,
        DATA_DIR / "Quillan_Universal_100_Percent_Master_Gold.jsonl",
        prompt_keys=["prompt", "question", "input", "instruction"],
        resp_keys=["response", "output", "answer", "text"],
        label="Universal 100% Master Gold",
        cap=8000,
    )
    all_samples.extend(s)

    # ── Source 2: quillan_corpus_CLEAN_V7 (large general corpus, sample) ─────
    # Too large to fully load — sample 6k from it
    LOGGER.info("[2] Sampling quillan_corpus_CLEAN_V7.jsonl (capped at 6k)...")
    v7_path = DATA_DIR / "quillan_corpus_CLEAN_V7.jsonl"
    if v7_path.exists():
        v7_samples = []
        seen_v7: set = set()
        with open(v7_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if len(v7_samples) >= 6000:
                    break
                if not line.strip():
                    continue
                try:
                    d = json.loads(line)
                    text = ""
                    for k in ["text", "content", "prompt", "response", "output"]:
                        if k in d and len(str(d[k])) > 200:
                            text = str(d[k]).strip()
                            break
                    if not text:
                        # Try single text field
                        if isinstance(d, str) and len(d) > 200:
                            text = d.strip()
                    if not text or len(text) < 200:
                        continue
                    key = text[:60]
                    if key in seen_v7:
                        continue
                    seen_v7.add(key)
                    # Split into prompt/response at natural boundary
                    # Look for "Question:"/"Answer:" or "User:"/"Assistant:" splits
                    if "\nAnswer:" in text or "\nA:" in text:
                        parts = text.split("\nAnswer:", 1) if "\nAnswer:" in text else text.split("\nA:", 1)
                        q_part = parts[0].replace("Question:", "").replace("Q:", "").strip()
                        r_part = parts[1].strip() if len(parts) > 1 else ""
                    elif "<|assistant|>" in text:
                        parts = text.split("<|assistant|>", 1)
                        q_part = parts[0].replace("<|user|>", "").strip()
                        r_part = parts[1].strip() if len(parts) > 1 else ""
                    else:
                        # Use as raw continuations — split at ~1/3 mark
                        split_point = max(60, len(text) // 3)
                        q_part = text[:split_point].strip()
                        r_part = text[split_point:split_point + 300].strip()

                    if not q_part or len(r_part) < MIN_RESP_CHARS:
                        continue
                    pair = encode_pair(tokenizer, q_part, r_part)
                    if pair:
                        v7_samples.append(pair)
                except Exception:
                    pass
        LOGGER.info("[+] quillan_corpus_CLEAN_V7: %d samples", len(v7_samples))
        all_samples.extend(v7_samples)

    # ── Source 3: PDF papers corpus (academic reasoning) ─────────────────────
    s = load_jsonl_source(
        tokenizer,
        DATA_DIR / "pdf_papers_corpus.jsonl",
        prompt_keys=["prompt", "question", "title", "abstract", "input"],
        resp_keys=["response", "text", "content", "output", "body"],
        label="PDF Papers Corpus",
        cap=2000,
    )
    all_samples.extend(s)

    # ── Source 4: Code training data ─────────────────────────────────────────
    s = load_jsonl_source(
        tokenizer,
        DATA_DIR / "code_train.jsonl",
        prompt_keys=["prompt", "instruction", "question", "input"],
        resp_keys=["output", "response", "completion", "code"],
        label="Code Training Data",
        cap=3000,
    )
    all_samples.extend(s)

    # ── Source 5: All 34 Expert-Specific Datasets ─────────────────────────────
    experts_dir = DATA_DIR / "experts_34"
    expert_total = 0
    if experts_dir.exists():
        for expert_file in sorted(experts_dir.glob("*.jsonl")):
            s = load_jsonl_source(
                tokenizer,
                expert_file,
                prompt_keys=["prompt", "question", "input", "instruction"],
                resp_keys=["response", "output", "answer", "completion"],
                label=f"Expert: {expert_file.stem}",
                cap=200,  # Small cap per expert to balance domains
            )
            all_samples.extend(s)
            expert_total += len(s)
    LOGGER.info("[+] All 34 expert datasets: %d total samples", expert_total)

    # ── Source 6: Refined Thought Corpus (prompt/refined_reasoning) ───────────
    refined_path = DATA_DIR / "Quillan_Refined_Thought_Corpus.jsonl"
    if refined_path.exists():
        r_samples = []
        with open(refined_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    d = json.loads(line)
                    q = d.get("prompt", "").strip()
                    r = d.get("refined_reasoning", d.get("teacher", "")).strip()
                    if not q or len(r) < MIN_RESP_CHARS:
                        continue
                    pair = encode_pair(tokenizer, q, r)
                    if pair:
                        r_samples.append(pair)
                except Exception:
                    pass
        LOGGER.info("[+] Refined Thought Corpus: %d samples", len(r_samples))
        all_samples.extend(r_samples)

    # ── Source 7: General full_train.jsonl ────────────────────────────────────
    s = load_jsonl_source(
        tokenizer,
        DATA_DIR / "full_train.jsonl",
        prompt_keys=["prompt", "question", "input", "instruction"],
        resp_keys=["response", "output", "answer", "completion"],
        label="Full Train General",
        cap=4000,
    )
    all_samples.extend(s)

    # ── Source 8: Router training data (teaches expert routing) ──────────────
    s = load_jsonl_source(
        tokenizer,
        DATA_DIR / "router_training_dataset.jsonl",
        prompt_keys=["prompt", "question", "input"],
        resp_keys=["response", "output", "answer"],
        label="Router Training Dataset",
        cap=1000,
    )
    all_samples.extend(s)

    # ── Deduplicate across all sources ────────────────────────────────────────
    LOGGER.info("Deduplicating across all sources...")
    seen_tensors: set = set()
    unique = []
    for inp, lbl in all_samples:
        h = hash(tuple(inp[:16].tolist()))
        if h not in seen_tensors:
            seen_tensors.add(h)
            unique.append((inp, lbl))

    random.shuffle(unique)
    LOGGER.info("[✓] Total unique augmented samples: %d", len(unique))

    # ── Save as .pt ────────────────────────────────────────────────────────────
    inp_all = torch.stack([s[0] for s in unique])
    lbl_all = torch.stack([s[1] for s in unique])

    torch.save(
        {
            "input_ids": inp_all,
            "labels": lbl_all,
            "num_samples": len(unique),
            "max_seq_len": MAX_SEQ_LEN,
        },
        str(OUT_PATH),
    )
    size_mb = OUT_PATH.stat().st_size / (1024 * 1024)
    LOGGER.info("[✓] Saved augmented_frontier_v2.pt: %d samples, %.1f MB", len(unique), size_mb)

    # Quality check
    avg_resp = (lbl_all != -100).sum(dim=1).float().mean().item()
    LOGGER.info("[✓] Avg response tokens per sample: %.1f", avg_resp)


if __name__ == "__main__":
    build()
