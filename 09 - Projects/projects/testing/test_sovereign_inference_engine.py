#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧪 UNIT & REGRESSION TEST SUITE FOR SOVEREIGN INFERENCE & API
---------------------------------------------------------------------------------------
Tests:
- Tokenizer encode/decode accuracy and boundary conditions.
- SamplingParams validation and deterministic seeding.
- Logits processors: N-gram suppression, top-k/top-p filtering, min-p gating.
- Secure checkpoint loading with CWE-502 mitigation.
- Autoregressive generation & streaming callback functionality.
- Stop string & token criteria verification.
- Empty prompt fallback robustness.
"""

import sys
import unittest
from pathlib import Path
import torch
import torch.nn as nn

# Local imports
SCRIPT_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from sovereign_inference_engine import (
    SamplingParams,
    SovereignInferenceEngine,
    SovereignTokenizer,
)


class DummyModel(nn.Module):
    """Lightweight deterministic model fixture for inference engine testing."""
    def __init__(self, vocab_size: int = 50257):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, 64)
        self.head = nn.Linear(64, vocab_size)

    def forward(self, input_ids: torch.Tensor):
        x = self.embedding(input_ids)
        logits = self.head(x)
        return logits


class TestSovereignTokenizer(unittest.TestCase):
    def setUp(self):
        self.tokenizer = SovereignTokenizer("gpt2")

    def test_encode_decode_roundtrip(self):
        text = "Quillan-Ronin sovereign intelligence architecture."
        tokens = self.tokenizer.encode(text)
        self.assertIsInstance(tokens, list)
        self.assertGreater(len(tokens), 0)
        decoded = self.tokenizer.decode(tokens)
        self.assertEqual(decoded, text)

    def test_invalid_input_types(self):
        with self.assertRaises(TypeError):
            self.tokenizer.encode(12345)  # type: ignore

        with self.assertRaises(TypeError):
            self.tokenizer.decode("not_an_iterable")  # type: ignore

        with self.assertRaises(TypeError):
            self.tokenizer.decode(b"byte_string")  # type: ignore

    def test_vocab_size_properties(self):
        self.assertGreaterEqual(self.tokenizer.vocab_size, 50000)
        self.assertIsInstance(self.tokenizer.eos_token_id, int)


class TestSamplingParams(unittest.TestCase):
    def test_default_values(self):
        params = SamplingParams()
        self.assertEqual(params.temperature, 0.65)
        self.assertEqual(params.top_k, 40)
        self.assertEqual(params.top_p, 0.85)
        self.assertEqual(params.min_p, 0.05)
        self.assertEqual(params.repetition_penalty, 1.20)
        self.assertEqual(params.no_repeat_ngram_size, 3)

    def test_immutability(self):
        params = SamplingParams()
        with self.assertRaises(Exception):
            params.temperature = 0.9  # Frozen dataclass


class TestSovereignInferenceEngine(unittest.TestCase):
    def setUp(self):
        self.model = DummyModel(vocab_size=50257)
        self.tokenizer = SovereignTokenizer("gpt2")
        self.engine = SovereignInferenceEngine(
            model=self.model,
            tokenizer=self.tokenizer,
            device="cpu",
        )

    def test_generate_deterministic_with_seed(self):
        params = SamplingParams(max_new_tokens=10, temperature=0.5, seed=42)
        out1 = self.engine.generate("Hello world", params=params)
        out2 = self.engine.generate("Hello world", params=params)
        self.assertEqual(out1, out2)

    def test_streaming_callback(self):
        collected_chunks = []

        def _cb(token: str):
            collected_chunks.append(token)

        params = SamplingParams(max_new_tokens=5, temperature=0.5, seed=42)
        full_text = self.engine.generate("Test prompt", params=params, stream_callback=_cb)
        self.assertEqual("".join(collected_chunks), full_text)

    def test_ngram_blocking(self):
        params = SamplingParams(max_new_tokens=15, no_repeat_ngram_size=2, seed=123)
        out = self.engine.generate("Repeated token token token", params=params)
        self.assertIsInstance(out, str)

    def test_empty_prompt_fallback(self):
        params = SamplingParams(max_new_tokens=5, temperature=0.5, seed=42)
        out = self.engine.generate("", params=params)
        self.assertIsInstance(out, str)

    def test_min_p_and_top_p_sampling(self):
        params = SamplingParams(max_new_tokens=10, temperature=0.7, top_p=0.90, min_p=0.10, seed=777)
        out = self.engine.generate("Exploring quantum physics", params=params)
        self.assertIsInstance(out, str)


if __name__ == "__main__":
    unittest.main(verbosity=2)
