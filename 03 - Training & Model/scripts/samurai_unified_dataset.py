#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
samurai_unified_dataset.py  -  Unified Multi-Dataset Pipeline for Quillan-Ronin (~204.92M Tokens)
==================================================================================================
Streams and batches all canonical datasets from `CrashOverrideX/Quillan_Samurai_sets`:
  1. GPT_5.5_Distilled.jsonl         (GPT Sol 5.6 / 5.5 Distilled)           ~11.63M tokens
  2. full_train.jsonl                (Claude Fable 5 / Opus multi-turn)       ~17.86M tokens
  3. instruct_train.jsonl            (Claude Fable 5 / Opus instructions)     ~15.15M tokens
  4. code_train.jsonl                (Claude Fable 5 / Opus coding)           ~3.11M tokens
  5. quillan_corpus_CLEAN_V7.jsonl   (Master Sovereign Pretrain/Cyber/Media)  ~137.69M tokens
  6. train.jsonl                     (Formal Reasoning Gym)                   ~7.52M tokens
  7. full_dataset.jsonl              (Extended instruction pairs)             ~7.08M tokens
  8. quillan_12mb_training_dataset   (Core seed corpus)                       ~3.12M tokens
  9. quillan_science_*.jsonl         (CAMEL-AI Physics & Deep Science)        ~1.76M tokens
==================================================================================================
"""

import os
import sys
import json
import glob
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Iterator

import torch
from torch.utils.data import IterableDataset, Dataset
from tokenizers import Tokenizer

USER_TOKEN_STR = "<|user|>"
ASSISTANT_TOKEN_STR = "<|assistant|>"
EOS_TOKEN_ID = 0


def find_dataset_dir() -> Path:
    """Locate the samurai datasets in HF cache or local repository directories."""
    # 1. HuggingFace Hub cache snapshot
    hf_cache_pattern = os.path.expanduser("~/.cache/huggingface/hub/datasets--CrashOverrideX--Quillan_Samurai_sets/snapshots/*")
    snapshots = glob.glob(hf_cache_pattern)
    if snapshots:
        return Path(snapshots[0])

    # 2. Local repository training_data
    local_repo = Path(r"C:\02_QUILLAN\03 - Training & Model\training_data")
    if local_repo.exists():
        return local_repo

    # 3. Fallback
    return Path(r"C:\02_QUILLAN\training_data")


class SamuraiDatasetItem:
    """Standardized representation of a single training example."""
    __slots__ = ("input_ids", "labels", "source", "token_count")

    def __init__(self, input_ids: List[int], labels: List[int], source: str):
        self.input_ids = input_ids
        self.labels = labels
        self.source = source
        self.token_count = len(input_ids)


class SamuraiDataProcessor:
    """Tokenizes and formats heterogeneous dataset schemas into unified causal LM pairs."""

    def __init__(self, tokenizer: Tokenizer, seq_len: int = 512):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.pad_id = EOS_TOKEN_ID

    def process_record(self, raw_item: dict, source_name: str) -> Optional[Tuple[List[int], List[int]]]:
        """
        Parse raw json record and return (input_ids, labels) padded/truncated to seq_len.
        Labels mask non-assistant tokens with -100 for SFT/instruction data.
        """
        # Case A: Messages format (Claude Fable 5: full_train, instruct_train, code_train)
        if "messages" in raw_item and isinstance(raw_item["messages"], list):
            return self._process_messages(raw_item["messages"])

        # Case B: Standard Q&A format (train.jsonl, Reasoning Gym)
        if "question" in raw_item and "answer" in raw_item:
            q = str(raw_item["question"]).strip()
            a = str(raw_item["answer"]).strip()
            return self._process_qa(q, a)

        # Case C: GPT Sol 5.6 / 5.5 Distilled format (already tagged with <|user|>, <|assistant|>)
        if "text" in raw_item:
            text = str(raw_item["text"]).strip()
            if USER_TOKEN_STR in text and ASSISTANT_TOKEN_STR in text:
                return self._process_tagged_text(text)
            else:
                # Pretraining text (quillan_corpus_CLEAN_V7, science)
                return self._process_pretrain_text(text)

        # Case D: prompt/response pair
        if "prompt" in raw_item and "response" in raw_item:
            p = str(raw_item["prompt"]).strip()
            r = str(raw_item["response"]).strip()
            return self._process_qa(p, r)

        return None

    def _process_qa(self, question: str, answer: str) -> Optional[Tuple[List[int], List[int]]]:
        if not question or not answer:
            return None
        prompt_text = f"{USER_TOKEN_STR}\n{question}\n{ASSISTANT_TOKEN_STR}\n"
        full_text = f"{prompt_text}{answer}"

        prompt_ids = self.tokenizer.encode(prompt_text).ids
        full_ids = self.tokenizer.encode(full_text).ids
        if len(full_ids) < 4:
            return None

        # Truncate to seq_len
        if len(full_ids) > self.seq_len:
            full_ids = full_ids[: self.seq_len]

        # Labels: mask prompt prefix with -100
        prefix_len = min(len(prompt_ids), len(full_ids))
        labels = [-100] * prefix_len + full_ids[prefix_len:]

        # Pad to seq_len
        pad_len = self.seq_len - len(full_ids)
        if pad_len > 0:
            full_ids = full_ids + [self.pad_id] * pad_len
            labels = labels + [-100] * pad_len

        return full_ids, labels

    def _process_tagged_text(self, text: str) -> Optional[Tuple[List[int], List[int]]]:
        parts = text.split(ASSISTANT_TOKEN_STR, 1)
        if len(parts) != 2:
            return self._process_pretrain_text(text)

        prompt_part = parts[0] + ASSISTANT_TOKEN_STR + "\n"
        response_part = parts[1].strip()

        prompt_ids = self.tokenizer.encode(prompt_part).ids
        full_ids = self.tokenizer.encode(text).ids
        if len(full_ids) < 4:
            return None

        if len(full_ids) > self.seq_len:
            full_ids = full_ids[: self.seq_len]

        prefix_len = min(len(prompt_ids), len(full_ids))
        labels = [-100] * prefix_len + full_ids[prefix_len:]

        pad_len = self.seq_len - len(full_ids)
        if pad_len > 0:
            full_ids = full_ids + [self.pad_id] * pad_len
            labels = labels + [-100] * pad_len

        return full_ids, labels

    def _process_messages(self, messages: List[dict]) -> Optional[Tuple[List[int], List[int]]]:
        full_ids = []
        labels = []

        for msg in messages:
            role = msg.get("role", "user")
            content = str(msg.get("content", "")).strip()
            if not content:
                continue

            if role in ("system", "user"):
                tag = f"{USER_TOKEN_STR}\n" if role == "user" else f"<|start|>\n"
                seg_text = f"{tag}{content}\n"
                seg_ids = self.tokenizer.encode(seg_text).ids
                full_ids.extend(seg_ids)
                labels.extend([-100] * len(seg_ids))
            elif role == "assistant":
                tag = f"{ASSISTANT_TOKEN_STR}\n"
                tag_ids = self.tokenizer.encode(tag).ids
                seg_ids = self.tokenizer.encode(f"{content}\n").ids
                full_ids.extend(tag_ids)
                labels.extend([-100] * len(tag_ids))
                full_ids.extend(seg_ids)
                labels.extend(seg_ids)

        if len(full_ids) < 4:
            return None

        if len(full_ids) > self.seq_len:
            full_ids = full_ids[: self.seq_len]
            labels = labels[: self.seq_len]

        pad_len = self.seq_len - len(full_ids)
        if pad_len > 0:
            full_ids = full_ids + [self.pad_id] * pad_len
            labels = labels + [-100] * pad_len

        return full_ids, labels

    def _process_pretrain_text(self, text: str) -> Optional[Tuple[List[int], List[int]]]:
        if not text:
            return None
        full_ids = self.tokenizer.encode(text).ids
        if len(full_ids) < 4:
            return None

        if len(full_ids) > self.seq_len:
            full_ids = full_ids[: self.seq_len]

        labels = list(full_ids)  # standard causal next-token prediction

        pad_len = self.seq_len - len(full_ids)
        if pad_len > 0:
            full_ids = full_ids + [self.pad_id] * pad_len
            labels = labels + [-100] * pad_len

        return full_ids, labels


class SamuraiStreamingDataset(IterableDataset):
    """
    Streaming multi-dataset reader that iterates through all Samurai dataset files
    in round-robin or randomized interleaving with zero OOM risk.
    """

    def __init__(
        self,
        dataset_dir: Optional[Path] = None,
        tokenizer: Optional[Tokenizer] = None,
        seq_len: int = 512,
        shuffle_files: bool = True,
        max_samples_per_epoch: Optional[int] = None,
    ):
        super().__init__()
        self.dataset_dir = Path(dataset_dir) if dataset_dir else find_dataset_dir()
        if tokenizer is None:
            tok_path = Path(r"C:\02_QUILLAN\03 - Training & Model\scripts\tokenizer.json")
            if not tok_path.exists():
                tok_path = Path(r"C:\02_QUILLAN\03 - Training & Model\quillan_bpe_tokenizer_hf\tokenizer.json")
            self.tokenizer = Tokenizer.from_file(str(tok_path))
        else:
            self.tokenizer = tokenizer

        self.seq_len = seq_len
        self.shuffle_files = shuffle_files
        self.max_samples_per_epoch = max_samples_per_epoch
        self.processor = SamuraiDataProcessor(self.tokenizer, seq_len=seq_len)

        # Discover all available jsonl files in the target directory
        self.files = sorted(list(self.dataset_dir.glob("*.jsonl")))
        if not self.files:
            print(f"[WARN] No .jsonl files found in {self.dataset_dir}")

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        file_list = list(self.files)
        if self.shuffle_files:
            random.shuffle(file_list)

        yielded = 0
        # Stream lines from all files
        for fpath in file_list:
            fname = fpath.name
            try:
                with open(fpath, "r", encoding="utf-8", errors="ignore") as fp:
                    for line in fp:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            record = json.loads(line)
                        except Exception:
                            continue

                        processed = self.processor.process_record(record, source_name=fname)
                        if processed is not None:
                            input_ids, labels = processed
                            yield (
                                torch.tensor(input_ids, dtype=torch.long),
                                torch.tensor(labels, dtype=torch.long),
                            )
                            yielded += 1
                            if self.max_samples_per_epoch and yielded >= self.max_samples_per_epoch:
                                return
            except Exception as e:
                print(f"[ERROR] Error reading {fname}: {e}")
                continue


def get_samurai_stats() -> Dict[str, dict]:
    """Audit and return row counts and estimated tokens across all Samurai dataset files."""
    d_dir = find_dataset_dir()
    stats = {}
    total_tokens = 0
    total_rows = 0

    for fpath in sorted(d_dir.glob("*.jsonl")):
        rows = 0
        chars = 0
        with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
            for l in f:
                rows += 1
                chars += len(l)
        tokens = chars // 4
        stats[fpath.name] = {
            "size_mb": os.path.getsize(fpath) / (1024 * 1024),
            "rows": rows,
            "est_tokens_m": tokens / 1e6,
        }
        total_tokens += tokens
        total_rows += rows

    stats["TOTAL"] = {
        "size_mb": sum(s["size_mb"] for k, s in stats.items() if k != "TOTAL"),
        "rows": total_rows,
        "est_tokens_m": total_tokens / 1e6,
    }
    return stats


if __name__ == "__main__":
    print("=" * 70)
    print("SAMURAI UNIFIED DATASET AUDIT (~204.92M TOKENS)")
    print("=" * 70)
    st = get_samurai_stats()
    for name, s in st.items():
        if name == "TOTAL":
            print("-" * 70)
        print(f"{name:45s} | {s['size_mb']:7.2f} MB | {s['rows']:7d} rows | ~{s['est_tokens_m']:6.2f}M tokens")
    print("=" * 70)
