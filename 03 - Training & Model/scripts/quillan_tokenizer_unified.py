#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified Modular Dynamic Tokenizer for Quillan-Ronin v5.4-ONI
=============================================================
Singular source of truth that unifies the legacy tokenizers and provides
native support for the 50,262 vocabulary with 5 dialogue/turn special tokens:

  - Custom Quillan BPE (50,262 vocabulary, specials at 0/1/2 and 50257-50261)
  - Unified special token mapping for chat templates and multi-turn dialogue
  - Modular domain-specific preprocessing (code, dialogue, scientific, general)
  - Dynamic legacy ID range detection and translation via text roundtrip

Special Token Architecture:
  0: <|endoftext|>
  1: <|pad|>
  2: <|bos|>
  50257: <|start|>
  50258: <|user|>
  50259: <|assistant|>
  50260: <|im_start|>
  50261: <|im_end|>
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

from tokenizers import Tokenizer

logger = logging.getLogger(__name__)

# Search candidate locations for tokenizer.json
POSSIBLE_PATHS = [
    Path(__file__).resolve().parent / "tokenizer.json",
    Path(__file__).resolve().parent.parent / "quillan_bpe_tokenizer_hf" / "tokenizer.json",
    Path(__file__).resolve().parent.parent / "tokenizer.json",
    Path(r"C:\02_QUILLAN\03 - Training & Model\scripts\tokenizer.json"),
    Path(r"C:\02_QUILLAN\03 - Training & Model\quillan_bpe_tokenizer_hf\tokenizer.json"),
]

CUSTOM_PATH: Path = next((p for p in POSSIBLE_PATHS if p.is_file()), POSSIBLE_PATHS[0])
TIKTOKEN_NAME = "gpt2"
LLAMA_PATH = Path(r"C:\02_QUILLAN\training_data\tokenizer.json")

# Unified special tokens — single mapping, all aliases resolve here
UNIFIED_SPECIALS: Dict[str, int] = {
    "<|endoftext|>": 0,
    "<|pad|>": 1,
    "<|bos|>": 2,
    "<|begin_of_text|>": 2,  # Llama alias -> bos
    "<|eot_id|>": 0,          # Llama alias -> endoftext
    "<|end_of_text|>": 0,
    "<|start|>": 50257,
    "<|user|>": 50258,
    "<|assistant|>": 50259,
    "<|im_start|>": 50260,
    "<|im_end|>": 50261,
}

REVERSE_SPECIALS: Dict[int, str] = {
    0: "<|endoftext|>",
    1: "<|pad|>",
    2: "<|bos|>",
    50257: "<|start|>",
    50258: "<|user|>",
    50259: "<|assistant|>",
    50260: "<|im_start|>",
    50261: "<|im_end|>",
}

DOMAIN_PREFIXES = {
    "code": "<|code|>",
    "dialogue": "<|dialogue|>",
    "scientific": "<|science|>",
    "general": "",
}


class UnifiedQuillanTokenizer:
    """Modular dynamic wrapper around the custom Quillan BPE."""

    DEFAULT_VOCAB_SIZE: int = 50262

    def __init__(self, base_path: Optional[Union[str, Path]] = None):
        if base_path:
            p = Path(base_path)
            self.base_path = p if p.is_file() else (p / "tokenizer.json")
        else:
            self.base_path = next((cp for cp in POSSIBLE_PATHS if cp.is_file()), CUSTOM_PATH)

        if not self.base_path.is_file():
            raise FileNotFoundError(f"UnifiedQuillanTokenizer asset not found: {self.base_path}")

        self._tok: Tokenizer = Tokenizer.from_file(str(self.base_path))
        self.vocab_size = self._tok.get_vocab_size()
        # Cache vocab for reverse lookup
        self._vocab = self._tok.get_vocab()
        self._id_to_token = {v: k for k, v in self._vocab.items()}
        # Domain modules (preprocessing configs)
        self.domains = set(DOMAIN_PREFIXES.keys())
        # Lazy-loaded legacy decoders for translation
        self._tiktoken = None
        self._llama_tok = None

    # -- Core encode/decode (modular) --

    def encode(self, text: str, domain: str = "general", add_special: bool = False) -> List[int]:
        if domain not in self.domains:
            domain = "general"
        # Domain-specific preprocessing (modular)
        if domain == "code":
            # Preserve indentation, normalize tabs to 4 spaces
            text = text.replace("\t", "    ")
        elif domain == "dialogue":
            # Ensure role prefixes are clean
            text = text.strip()
        # Base BPE encode
        ids = self._tok.encode(text).ids
        if add_special:
            ids = ids + [self.eos_token_id]
        return ids

    def decode(self, ids: List[int], skip_special: bool = False) -> str:
        return self._tok.decode(ids, skip_special_tokens=skip_special)

    def batch_encode(self, texts: List[str], domain: str = "general") -> List[List[int]]:
        return [self.encode(t, domain=domain) for t in texts]

    def batch_decode(self, batch_ids: List[List[int]], skip_special: bool = False) -> List[str]:
        return [self.decode(ids, skip_special=skip_special) for ids in batch_ids]

    # -- Special tokens (unified) --

    @property
    def eos_token_id(self) -> int:
        return self.token_to_id("<|endoftext|>") or 0

    @property
    def pad_token_id(self) -> int:
        return self.token_to_id("<|pad|>") or 1

    @property
    def bos_token_id(self) -> int:
        return self.token_to_id("<|bos|>") or 2

    @property
    def start_token_id(self) -> int:
        return UNIFIED_SPECIALS["<|start|>"]

    @property
    def user_token_id(self) -> int:
        return UNIFIED_SPECIALS["<|user|>"]

    @property
    def assistant_token_id(self) -> int:
        return UNIFIED_SPECIALS["<|assistant|>"]

    @property
    def im_start_token_id(self) -> int:
        return UNIFIED_SPECIALS["<|im_start|>"]

    @property
    def im_end_token_id(self) -> int:
        return UNIFIED_SPECIALS["<|im_end|>"]

    def token_to_id(self, token: str) -> Optional[int]:
        # Check unified aliases first
        if token in UNIFIED_SPECIALS:
            return UNIFIED_SPECIALS[token]
        return self._vocab.get(token)

    def id_to_token(self, token_id: int) -> Optional[str]:
        if token_id in REVERSE_SPECIALS:
            return REVERSE_SPECIALS[token_id]
        return self._id_to_token.get(token_id)

    # -- Dynamic legacy translation --

    def _get_tiktoken(self):
        if self._tiktoken is None:
            import tiktoken
            self._tiktoken = tiktoken.get_encoding(TIKTOKEN_NAME)
        return self._tiktoken

    def translate_legacy_ids(self, ids: List[int], source: str = "tiktoken_gpt2") -> List[int]:
        """Translate legacy token IDs to unified custom IDs via text roundtrip."""
        if source == "tiktoken_gpt2":
            tt = self._get_tiktoken()
            try:
                text = tt.decode(ids)
            except Exception:
                text = ""
            return self.encode(text)
        elif source == "llama_128k":
            filtered = [i for i in ids if 0 <= i < self.vocab_size]
            return self.encode(self.decode(filtered)) if filtered else []
        else:
            return list(ids)

    def is_legacy_range(self, ids: List[int]) -> str:
        """Heuristic: detect which legacy tokenizer produced these IDs."""
        if not ids:
            return "empty"
        try:
            text_custom = self.decode(ids)
            if "\ufffd" in text_custom or len(text_custom) < len(ids) * 0.5:
                return "tiktoken_gpt2"
        except Exception:
            return "tiktoken_gpt2"
        return "custom_50262"

    def __len__(self) -> int:
        return self.vocab_size

    # -- Info --

    def info(self) -> dict:
        return {
            "base": str(self.base_path),
            "vocab_size": self.vocab_size,
            "eos": self.eos_token_id,
            "pad": self.pad_token_id,
            "bos": self.bos_token_id,
            "start": self.start_token_id,
            "user": self.user_token_id,
            "assistant": self.assistant_token_id,
            "im_start": self.im_start_token_id,
            "im_end": self.im_end_token_id,
            "domains": sorted(self.domains),
            "unified_specials": UNIFIED_SPECIALS,
        }
