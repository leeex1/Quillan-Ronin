#!/usr/bin/env python3
"""Thin wrapper around HF tokenizers for Quillan training pipeline compatibility."""
from pathlib import Path
from tokenizers import Tokenizer

class QuillanBPETokenizer:
    """Drops-in replacement for legacy QuillanBPETokenizer using HF tokenizers (Rust)."""
    
    def __init__(self, path=None, vocab_size=50257):
        self._tok = None
        self.vocab_size = vocab_size
        if path:
            self.load(str(path))

    def load(self, path):
        p = Path(path)
        if p.suffix == '.pkl':
            # Check for quillan_bpe_tokenizer_hf/tokenizer.json
            hf_path = p.parent / "quillan_bpe_tokenizer_hf" / "tokenizer.json"
            if hf_path.exists():
                path = str(hf_path)
            else:
                hf_path2 = p.parent / "tokenizer.json"
                if hf_path2.exists():
                    path = str(hf_path2)
        self._tok = Tokenizer.from_file(path)
        self.vocab_size = self._tok.get_vocab_size()

    def save(self, path):
        self._tok.save(str(path))

    def encode(self, text):
        return self._tok.encode(text).ids

    def decode(self, ids):
        return self._tok.decode(ids)

    @property
    def eos_token_id(self):
        return self._tok.token_to_id("<|endoftext|>") or 0

    @property
    def pad_token_id(self):
        return self._tok.token_to_id("<|pad|>") or 1

    @property
    def bos_token_id(self):
        return self._tok.token_to_id("<|bos|>") or 2
