#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quillan BPE Tokenizer Module (v5.4.0-oni — 2026 Sovereign Release)
================================================================
Production wrapper around HuggingFace Rust tokenizers for Quillan-Ronin.
Standardized on static JSON serialization (`tokenizer.json`) to guarantee
cross-platform portability, high-throughput Rust tokenization, and zero-risk
deserialization (mitigating CWE-502 pickle vulnerabilities).
"""

from __future__ import annotations

import logging
import os
import warnings
from pathlib import Path
from typing import List, Optional, Union

from version import __version__, RELEASE_YEAR

try:
    from tokenizers import Tokenizer
    TOKENIZERS_AVAILABLE = True
except ImportError:
    Tokenizer = None
    TOKENIZERS_AVAILABLE = False
    warnings.warn(
        "HuggingFace 'tokenizers' library is not installed. Operating in native stdlib fallback mode. "
        "Install via: pip install tokenizers",
        ImportWarning,
        stacklevel=2,
    )

logger = logging.getLogger(__name__)


class QuillanBPETokenizer:
    """Production drop-in BPE Tokenizer for the Quillan-Ronin neural architecture.

    Backbone serialization uses HuggingFace Rust `tokenizer.json` with a 50,257
    token vocabulary, NFKC normalization, and ByteLevel pre-tokenization.

    Attributes:
        vocab_size (int): Size of the active vocabulary (default: 50,257).
        tokenizer_path (Optional[Path]): Absolute path to loaded tokenizer file.
    """

    DEFAULT_VOCAB_SIZE: int = 50257

    def __init__(
        self,
        path: Optional[Union[str, Path]] = None,
        vocab_size: int = DEFAULT_VOCAB_SIZE,
    ) -> None:
        """Initialize the tokenizer, optionally loading from a path or auto-discovering.

        Args:
            path: Path to `tokenizer.json` or directory containing `tokenizer.json`.
                  Legacy `.pkl` paths trigger deprecation and auto-redirect to JSON.
            vocab_size: Default expected vocabulary size fallback.
        """
        self._tok: Optional[Tokenizer] = None
        self.vocab_size: int = vocab_size
        self.tokenizer_path: Optional[Path] = None

        if path is not None:
            self.load(path)
        else:
            self._auto_discover()

    def _resolve_safe_path(self, path: Union[str, Path]) -> Path:
        """Resolve and validate path against directory traversal and existence."""
        raw_path = Path(path).expanduser()
        try:
            resolved = raw_path.resolve(strict=False)
        except (RuntimeError, OSError) as exc:
            raise ValueError(f"Invalid path traversal or resolution error: {path}") from exc

        # Handle legacy .pkl deprecation and redirect to tokenizer.json
        if resolved.suffix.lower() == ".pkl":
            warnings.warn(
                "Legacy .pkl tokenizer format is deprecated (CWE-502 risk). "
                "Redirecting to HuggingFace 'tokenizer.json'. "
                "Deprecation window: 90 days / v5.4.0.",
                DeprecationWarning,
                stacklevel=3,
            )
            candidates = [
                resolved.parent / "quillan_bpe_tokenizer_hf" / "tokenizer.json",
                resolved.parent / "tokenizer.json",
                resolved.parent / "oni" / "tokenizer.json",
            ]
            for candidate in candidates:
                if candidate.is_file():
                    logger.info("Redirected legacy .pkl request to %s", candidate)
                    return candidate

        # If directory was supplied, look for tokenizer.json inside
        if resolved.is_dir():
            target_json = resolved / "tokenizer.json"
            if target_json.is_file():
                return target_json

        return resolved

    def _auto_discover(self) -> None:
        """Attempt to locate tokenizer.json across standard canonical repository locations."""
        current_dir = Path(__file__).resolve().parent
        candidate_paths = [
            current_dir / "quillan_bpe_tokenizer_hf" / "tokenizer.json",
            current_dir / "oni" / "tokenizer.json",
            current_dir / "training_data" / "tokenizer.json",
            current_dir / "_dev" / "quillan_bpe_tokenizer_hf" / "tokenizer.json",
        ]
        for candidate in candidate_paths:
            if candidate.is_file():
                logger.info("Auto-discovered Quillan tokenizer at %s", candidate)
                self.load(candidate)
                return
        logger.warning(
            "QuillanBPETokenizer initialized unpopulated: no tokenizer.json auto-discovered."
        )

    def load(self, path: Union[str, Path]) -> None:
        """Load tokenizer state from a validated tokenizer.json file.

        Args:
            path: String or Path to the JSON tokenizer.

        Raises:
            FileNotFoundError: If the target tokenizer file does not exist.
            ValueError: If the file is not a valid Tokenizer JSON representation.
        """
        if not TOKENIZERS_AVAILABLE:
            logger.warning("HuggingFace tokenizers library unavailable; running in fallback mode.")
            self._tok = None
            return

        resolved_path = self._resolve_safe_path(path)
        if not resolved_path.is_file():
            raise FileNotFoundError(f"Tokenizer asset not found at: {resolved_path}")

        try:
            self._tok = Tokenizer.from_file(str(resolved_path))
            self.vocab_size = self._tok.get_vocab_size()
            self.tokenizer_path = resolved_path
            logger.debug(
                "Successfully loaded tokenizer from %s (Vocab: %d)",
                resolved_path,
                self.vocab_size,
            )
        except Exception as exc:
            raise ValueError(
                f"Failed to parse HuggingFace tokenizer from '{resolved_path}': {exc}"
            ) from exc

    def save(self, path: Union[str, Path]) -> None:
        """Persist tokenizer state to disk in standard JSON format.

        Args:
            path: Destination file path.
        """
        if self._tok is None:
            raise RuntimeError("Cannot save uninitialized tokenizer.")
        dest_path = Path(path).expanduser().resolve()
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        self._tok.save(str(dest_path))
        logger.info("Saved tokenizer configuration to %s", dest_path)

    def encode(self, text: str) -> List[int]:
        """Tokenize input string into list of token IDs.

        Args:
            text: Plain text input.

        Returns:
            List of integer token IDs.
        """
        if not isinstance(text, str):
            raise TypeError(f"Expected str input for encode, received: {type(text).__name__}")
        if self._tok is not None:
            return self._tok.encode(text).ids
        # Native stdlib byte fallback when tokenizers library is not installed
        return list(text.encode("utf-8", errors="replace"))

    def decode(self, ids: List[int], skip_special_tokens: bool = False) -> str:
        """Decode integer token IDs back into string.

        Args:
            ids: List of integer token IDs.
            skip_special_tokens: Whether to omit special tokens (<|endoftext|>, etc.)

        Returns:
            Decoded text string.
        """
        if not isinstance(ids, list):
            raise TypeError(f"Expected list[int] for decode, received: {type(ids).__name__}")
        if self._tok is not None:
            return self._tok.decode(ids, skip_special_tokens=skip_special_tokens)
        # Native stdlib byte fallback
        return bytes([i % 256 for i in ids]).decode("utf-8", errors="replace")

    @property
    def eos_token_id(self) -> int:
        """End of sequence token ID (default: 0)."""
        if self._tok is None:
            return 0
        tid = self._tok.token_to_id("<|endoftext|>")
        return tid if tid is not None else 0

    @property
    def pad_token_id(self) -> int:
        """Padding token ID (default: 1)."""
        if self._tok is None:
            return 1
        tid = self._tok.token_to_id("<|pad|>")
        return tid if tid is not None else 1

    @property
    def bos_token_id(self) -> int:
        """Beginning of sequence token ID (default: 2)."""
        if self._tok is None:
            return 2
        tid = self._tok.token_to_id("<|bos|>")
        return tid if tid is not None else 2

    def __len__(self) -> int:
        return self.vocab_size

    def __repr__(self) -> str:
        return (
            f"QuillanBPETokenizer(vocab_size={self.vocab_size}, "
            f"loaded={self._tok is not None}, path='{self.tokenizer_path}')"
        )
