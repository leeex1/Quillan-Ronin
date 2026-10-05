#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sft_common.py - shared SFT data-hygiene helpers for run_6l_sft_clean.py and
run_12l_sft_clean.py.

Why this exists
---------------
The gold SFT files hold ~2.3k rows but only ~51 UNIQUE (question, response)
pairs (each repeated up to ~100x). A row-level random split therefore puts the
same example in both train and val, so "val loss" measured memorisation and
"best checkpoint" meant "most memorised".

These helpers make evaluation honest:
  * group_split()  - holds out whole *questions*, never rows, so no held-out
                     question is ever seen in training. The seed is shared so
                     the 6L and 12L models are scored on the same held-out set.
  * data_health_report() - states plainly when the corpus is too small.
  * check_full_unfreeze_fits() - refuses a CPU full-unfreeze that would not fit
                     in available RAM instead of thrashing/pagefile-stalling.

No new runtime dependencies (psutil is optional and already used by the
callers, with a stdlib fallback).
"""
from __future__ import annotations

import random
from typing import Sequence

# Below this many unique (question, response) pairs the model can only memorise.
MIN_UNIQUE_PAIRS_WARN = 500
SPLIT_SEED = 1337


def group_split(
    keys: Sequence[str],
    val_fraction: float = 0.10,
    min_val_groups: int = 3,
    seed: int = SPLIT_SEED,
) -> tuple[list[int], list[int], set[str]]:
    """Split sample indices by *group key* (the question text).

    Every sample whose key is in the held-out set goes to validation; none of
    those keys appear in train. Deterministic for a given (keys, seed).

    Returns (train_indices, val_indices, held_out_keys). If val_fraction <= 0,
    or there are too few groups to hold any out, val is empty.
    """
    groups = sorted(set(keys))
    if val_fraction <= 0 or len(groups) < 2:
        return list(range(len(keys))), [], set()

    rng = random.Random(seed)
    rng.shuffle(groups)
    n_val = max(min_val_groups, int(round(len(groups) * val_fraction)))
    n_val = min(n_val, len(groups) - 1)  # always leave at least one train group
    held_out = set(groups[:n_val])

    train_idx = [i for i, k in enumerate(keys) if k not in held_out]
    val_idx = [i for i, k in enumerate(keys) if k in held_out]
    return train_idx, val_idx, held_out


def data_health_report(
    rows_read: int, unique_pairs: int, unique_questions: int, dup_skipped: int
) -> list[str]:
    """Human-readable data-quality lines (caller prints them)."""
    lines = [
        f"  Data health: {rows_read} rows read -> {unique_pairs} unique pairs "
        f"({unique_questions} unique questions), {dup_skipped} exact duplicates dropped",
    ]
    if unique_pairs < MIN_UNIQUE_PAIRS_WARN:
        lines.append(
            f"  [WARN] DATA-STARVED: only {unique_pairs} unique pairs "
            f"(< {MIN_UNIQUE_PAIRS_WARN}). Training can memorise these but cannot "
            f"generalise. More steps or a bigger unfreeze will NOT fix this - "
            f"add genuinely distinct training data."
        )
    return lines


def check_full_unfreeze_fits(estimated_gb: float, safety: float = 0.85) -> None:
    """Raise if a full-unfreeze footprint will not fit in available RAM.

    No-op if psutil is unavailable (cannot measure -> do not block).
    """
    try:
        import psutil
    except ImportError:
        return
    available_gb = psutil.virtual_memory().available / 1e9
    if estimated_gb > available_gb * safety:
        raise RuntimeError(
            f"Full unfreeze needs ~{estimated_gb:.1f} GB but only "
            f"{available_gb:.1f} GB RAM is available (limit {safety:.0%}). "
            f"Close other work / stop other training, or use "
            f"QUILLAN_UNFREEZE_SET=wide."
        )


class MasterCleanSFTDataset:
    """Unified SFT + Master Gold Domain Alignment Dataset for Quillan-Ronin.

    Loads and deduplicates:
      1. quillan_gold_sft_master.pt (2,355 SFT samples)
      2. quillan_master_gold_training_v1.pt (15,631 Council & Domain knowledge samples)
      3. JSONL sources (Quillan_Universal_Sovereign_Gold_1000, Quillan_Direct_Answers_Gold, sovereign_thinking_gold)

    Indexes group keys by prompt hash for zero train/val leakage with group_split().
    """
    def __init__(self, tok=None, seq_len: int = 256, repo_root=None, include_pretrain: bool = True):
        import hashlib, json
        from pathlib import Path
        import torch

        if repo_root is None:
            self.repo_root = Path(r"C:\02_QUILLAN")
        else:
            self.repo_root = Path(repo_root)

        self.train_dir = self.repo_root / "03 - Training & Model"
        self.seq_len = seq_len
        self.samples = []
        self.keys = []
        seen_hashes = set()
        dup_skipped = 0

        def resolve_path(rel_p):
            p1 = self.train_dir / rel_p
            if p1.exists():
                return p1
            return self.repo_root / rel_p

        # 1. Load SFT Master Tensors (2,355 samples)
        p_sft = resolve_path("training_data/canonical_standardized/quillan_gold_sft_master.pt")
        if p_sft.exists():
            d_sft = torch.load(p_sft, map_location="cpu", weights_only=False)
            sft_inps = d_sft["input_ids"]
            sft_lbls = d_sft["labels"]
            sft_added = 0
            for i in range(len(sft_inps)):
                inp_t = sft_inps[i]
                lbl_t = sft_lbls[i]
                h = hashlib.sha256(inp_t.numpy().tobytes()).digest()
                if h in seen_hashes:
                    dup_skipped += 1
                    continue
                seen_hashes.add(h)

                v = (lbl_t != -100).nonzero(as_tuple=True)[0]
                fv = v[0].item() if len(v) > 0 else 0
                k = f"p_{hashlib.md5(inp_t[:fv].numpy().tobytes()).hexdigest()}" if fv > 0 else f"sft_{i}"

                self.samples.append((inp_t, lbl_t))
                self.keys.append(k)
                sft_added += 1
            print(f"  [DATA] SFT Master: {sft_added} unique samples loaded from {p_sft.name}")

        # 2. Load Master Gold Training Tensors (15,631 samples)
        p_master = resolve_path("training_data/canonical_standardized/quillan_master_gold_training_v1.pt")
        if p_master.exists():
            d_m = torch.load(p_master, map_location="cpu", weights_only=False)
            m_inps = d_m["input_ids"]
            m_lbls = d_m["labels"]
            m_added = 0
            for i in range(len(m_inps)):
                inp_t = m_inps[i]
                lbl_t = m_lbls[i]
                h = hashlib.sha256(inp_t.numpy().tobytes()).digest()
                if h in seen_hashes:
                    dup_skipped += 1
                    continue
                seen_hashes.add(h)

                v = (lbl_t != -100).nonzero(as_tuple=True)[0]
                fv = v[0].item() if len(v) > 0 else 0
                k = f"p_{hashlib.md5(inp_t[:fv].numpy().tobytes()).hexdigest()}" if fv > 0 else f"doc_{hashlib.md5(inp_t[:32].numpy().tobytes()).hexdigest()}"

                self.samples.append((inp_t, lbl_t))
                self.keys.append(k)
                m_added += 1
            print(f"  [DATA] Master Gold: {m_added} unique samples loaded from {p_master.name}")

        # 3. Load Master Distilled & Samurai Tensors from E:\quillan_datasets (48.1M tokens)
        samurai_dir = Path(r"E:\quillan_datasets")
        if samurai_dir.exists():
            master_files = [
                ("GPT_5.5_Distilled.pt", "gpt5"),
                ("instruct_train.pt", "instruct"),
                ("code_train.pt", "code"),
                ("train.pt", "train"),
                ("quillan_science_absolute.pt", "sci_abs"),
                ("quillan_science_additional.pt", "sci_add"),
                ("full_train.pt", "full"),
            ]
            for m_file, prefix in master_files:
                p_m = samurai_dir / m_file
                if not p_m.exists():
                    continue
                try:
                    tensor = torch.load(p_m, map_location="cpu", weights_only=False)
                    if hasattr(tensor, "shape") and tensor.dim() == 1:
                        n_tokens = tensor.numel()
                        n_samples = n_tokens // self.seq_len
                        added_m = 0
                        for si in range(n_samples):
                            inp_t = tensor[si * self.seq_len : (si + 1) * self.seq_len]
                            lbl_t = inp_t.clone()
                            lbl_t[lbl_t == 0] = -100

                            h = hashlib.sha256(inp_t.numpy().tobytes()).digest()
                            if h in seen_hashes:
                                dup_skipped += 1
                                continue
                            seen_hashes.add(h)

                            k = f"{prefix}_{hashlib.md5(inp_t[:32].numpy().tobytes()).hexdigest()}"
                            self.samples.append((inp_t, lbl_t))
                            self.keys.append(k)
                            added_m += 1
                        print(f"  [DATA] Master Samurai: {added_m} unique samples ({added_m * self.seq_len:,} tokens) loaded from {m_file}")
                except Exception as e:
                    print(f"  [WARN] Failed to load {m_file}: {e}")

        # 4. Load Pretrain Foundation Corpus Tensors if requested
        if include_pretrain and len(self.samples) < 50000:
            p_pretrain = resolve_path("training_data/quillan_pretrain_corpus_343mb.pt")
            if p_pretrain.exists():
                d_p = torch.load(p_pretrain, map_location="cpu", weights_only=False)
                p_inps = d_p.get("input_ids", None)
                if p_inps is not None:
                    p_added = 0
                    for i in range(len(p_inps)):
                        inp_t = p_inps[i]
                        if inp_t.dtype != torch.long:
                            inp_t = inp_t.long()
                        lbl_t = inp_t.clone()
                        lbl_t[lbl_t == 0] = -100

                        h = hashlib.sha256(inp_t.numpy().tobytes()).digest()
                        if h in seen_hashes:
                            dup_skipped += 1
                            continue
                        seen_hashes.add(h)

                        k = f"pre_{hashlib.md5(inp_t[:32].numpy().tobytes()).hexdigest()}"
                        self.samples.append((inp_t, lbl_t))
                        self.keys.append(k)
                        p_added += 1
                    print(f"  [DATA] Pretrain Corpus: {p_added} unique samples loaded from {p_pretrain.name}")

        # 5. Load JSONL Sources (if tokenizer provided)
        jsonl_added = 0
        if tok is not None:
            jsonl_sources = [
                (resolve_path("training_data/Quillan_Universal_Sovereign_Gold_1000.jsonl"), "prompt", "response"),
                (resolve_path("training_data/Quillan_Direct_Answers_Gold.jsonl"), "prompt", "response"),
                (resolve_path("training_data/sovereign_thinking_gold.jsonl"), "question", "response"),
            ]
            for path, q_k, a_k in jsonl_sources:
                if not path.exists():
                    continue
                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            item = json.loads(line)
                            q = item.get(q_k, "").strip()
                            a = item.get(a_k, "").strip()
                            if not q or not a:
                                continue
                            prompt_prefix = f"User: {q}\n\nAssistant:"
                            full_text     = f"{prompt_prefix} {a}"

                            p_ids = tok.encode(prompt_prefix).ids
                            f_ids = tok.encode(full_text).ids
                            if len(f_ids) + 1 > self.seq_len:
                                continue
                            if f_ids[:len(p_ids)] != p_ids:
                                continue

                            inp_list = (f_ids + [0] * (self.seq_len - len(f_ids)))[:self.seq_len]
                            lbl_list = ([-100] * len(p_ids) + f_ids[len(p_ids):] + [0] + [-100] * (self.seq_len - len(f_ids) - 1))[:self.seq_len]

                            inp_t = torch.tensor(inp_list, dtype=torch.long)
                            lbl_t = torch.tensor(lbl_list, dtype=torch.long)
                            h = hashlib.sha256(inp_t.numpy().tobytes()).digest()
                            if h in seen_hashes:
                                dup_skipped += 1
                                continue
                            seen_hashes.add(h)
                            k = f"q_{hashlib.md5(q.encode('utf-8')).hexdigest()}"
                            self.samples.append((inp_t, lbl_t))
                            self.keys.append(k)
                            jsonl_added += 1
                        except Exception:
                            pass
            if jsonl_added > 0:
                print(f"  [DATA] JSONL Sources: {jsonl_added} unique samples loaded")

        print(f"  [DATA] Total Active Dataset: {len(self.samples)} unique samples ({dup_skipped} duplicates skipped)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

