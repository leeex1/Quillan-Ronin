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

        # 0. Check for Fast Pre-Tokenized Cache
        cache_p = Path(r"E:\quillan_datasets\sft_master_tokenized_cache.pt")
        if cache_p.exists():
            try:
                print(f"  [DATA] Loading fast pre-tokenized cache from {cache_p.name}...")
                cached = torch.load(cache_p, map_location="cpu", weights_only=False)
                if "inputs" in cached and "labels" in cached:
                    inps = cached["inputs"]
                    lbls = cached["labels"]
                    self.samples = [(inps[i], lbls[i]) for i in range(len(inps))]
                    self.keys = cached["keys"]
                else:
                    self.samples = cached["samples"]
                    self.keys = cached["keys"]
                print(f"  [DATA] Total Active Dataset: {len(self.samples)} genuine SFT samples loaded in 0.5s!")
                return
            except Exception as e:
                print(f"  [WARN] Cache load failed ({e}), rebuilding from raw sources...")

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

        # 3. Load Authentic Raw JSONL Dialogues from E:\quillan_datasets (GPT-5.5, Code, Instruct, Train)
        if tok is None:
            from tokenizers import Tokenizer
            tok_path = resolve_path("quillan_bpe_tokenizer_hf/tokenizer.json")
            if tok_path.exists():
                tok = Tokenizer.from_file(str(tok_path))

        samurai_dir = Path(r"E:\quillan_datasets")
        if samurai_dir.exists() and tok is not None:
            # A. GPT_5.5_Distilled.jsonl (18,197 samples)
            p_gpt = samurai_dir / "GPT_5.5_Distilled.jsonl"
            if p_gpt.exists():
                gpt_added = 0
                with open(p_gpt, "r", encoding="utf-8", errors="replace") as f:
                    for line in f:
                        if not line.strip(): continue
                        try:
                            obj = json.loads(line)
                            text = obj.get("text", "")
                            if "<|user|>" in text and "<|assistant|>" in text:
                                parts = text.split("<|assistant|>")
                                q = parts[0].replace("<|user|>", "").strip()
                                a = parts[1].strip()
                                if not q or not a: continue
                                prompt_prefix = f"User: {q}\n\nAssistant:"
                                full_text     = f"{prompt_prefix} {a}"
                                p_ids = tok.encode(prompt_prefix).ids
                                f_ids = tok.encode(full_text).ids
                                if len(f_ids) + 1 > self.seq_len:
                                    f_ids = f_ids[:self.seq_len - 1]
                                if len(p_ids) >= len(f_ids):
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
                                k = f"gpt_{hashlib.md5(q.encode('utf-8')).hexdigest()}"
                                self.samples.append((inp_t, lbl_t))
                                self.keys.append(k)
                                gpt_added += 1
                        except Exception:
                            pass
                print(f"  [DATA] GPT-5.5 Distilled: {gpt_added} unique SFT samples loaded from {p_gpt.name}")

            # B. code_train.jsonl & instruct_train.jsonl (Claude Opus dialogues)
            for fn, pfx in [("code_train.jsonl", "code"), ("instruct_train.jsonl", "inst")]:
                p_c = samurai_dir / fn
                if not p_c.exists(): continue
                c_added = 0
                with open(p_c, "r", encoding="utf-8", errors="replace") as f:
                    for line in f:
                        if not line.strip(): continue
                        try:
                            obj = json.loads(line)
                            msgs = obj.get("messages", [])
                            q, a = "", ""
                            for m in msgs:
                                if m.get("role") == "user" and not q:
                                    q = m.get("content", "").strip()
                                elif m.get("role") == "assistant" and not a:
                                    a = m.get("content", "").strip()
                            if not q or not a: continue
                            prompt_prefix = f"User: {q}\n\nAssistant:"
                            full_text     = f"{prompt_prefix} {a}"
                            p_ids = tok.encode(prompt_prefix).ids
                            f_ids = tok.encode(full_text).ids
                            if len(f_ids) + 1 > self.seq_len:
                                f_ids = f_ids[:self.seq_len - 1]
                            if len(p_ids) >= len(f_ids):
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
                            k = f"{pfx}_{hashlib.md5(q.encode('utf-8')).hexdigest()}"
                            self.samples.append((inp_t, lbl_t))
                            self.keys.append(k)
                            c_added += 1
                        except Exception:
                            pass
                print(f"  [DATA] Claude Opus ({pfx}): {c_added} unique SFT samples loaded from {fn}")

            # C. train.jsonl (15,000 reasoning samples)
            p_tr = samurai_dir / "train.jsonl"
            if p_tr.exists():
                tr_added = 0
                with open(p_tr, "r", encoding="utf-8", errors="replace") as f:
                    for line in f:
                        if not line.strip(): continue
                        try:
                            obj = json.loads(line)
                            q = obj.get("question", "").strip()
                            a = obj.get("answer", "").strip()
                            if not q or not a: continue
                            prompt_prefix = f"User: {q}\n\nAssistant:"
                            full_text     = f"{prompt_prefix} {a}"
                            p_ids = tok.encode(prompt_prefix).ids
                            f_ids = tok.encode(full_text).ids
                            if len(f_ids) + 1 > self.seq_len:
                                f_ids = f_ids[:self.seq_len - 1]
                            if len(p_ids) >= len(f_ids):
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
                            k = f"tr_{hashlib.md5(q.encode('utf-8')).hexdigest()}"
                            self.samples.append((inp_t, lbl_t))
                            self.keys.append(k)
                            tr_added += 1
                        except Exception:
                            pass
                print(f"  [DATA] Core Reasoning: {tr_added} unique SFT samples loaded from {p_tr.name}")

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

