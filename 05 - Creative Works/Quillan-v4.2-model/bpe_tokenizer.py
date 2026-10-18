#!/usr/bin/env python3
"""
BPE Tokenization implementation for better language understanding
"""

import re
from collections import defaultdict, Counter
import pickle
import os

class BPETokenizer:
    """Byte Pair Encoding tokenizer for better language modeling"""

    def __init__(self, vocab_size=5000):
        self.vocab_size = vocab_size
        self.merges = {}
        self.vocab = None
        self.inverse_vocab = None

    def get_stats(self, ids):
        """Count pairs of consecutive symbols"""
        counts = defaultdict(int)
        for pair in zip(ids, ids[1:]):
            counts[pair] += 1
        return counts

    def merge(self, ids, pair, idx):
        """Merge pair into single token"""
        new_ids = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
                new_ids.append(idx)
                i += 2
            else:
                new_ids.append(ids[i])
                i += 1
        return new_ids

    def train(self, text, verbose=False):
        """Train BPE tokenizer on text"""
        # Basic character-level tokenization first
        tokens = list(text.encode('utf-8'))

        # Build initial vocab (0-255 for bytes)
        vocab = {idx: bytes([idx]) for idx in range(256)}
        num_merges = self.vocab_size - 256

        ids = list(tokens)

        for i in range(num_merges):
            stats = self.get_stats(ids)
            if not stats:
                break

            # Find most frequent pair
            pair = max(stats, key=stats.get)
            idx = 256 + i

            # Merge the pair
            ids = self.merge(ids, pair, idx)

            # Update vocab
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]

            # Save merge
            self.merges[pair] = idx

            if verbose and i % 100 == 0:
                print(f"Merge {i+1}/{num_merges}: {pair} -> {idx}")

        # Build final vocab and inverse vocab
        self.vocab = vocab
        self.inverse_vocab = {v: k for k, v in vocab.items()}

        print(f"✅ BPE Tokenizer trained with {len(self.vocab)} tokens")

    def encode(self, text):
        """Encode text to token IDs"""
        tokens = list(text.encode('utf-8'))

        # Apply merges in order
        ids = tokens[:]
        for pair, idx in self.merges.items():
            ids = self.merge(ids, pair, idx)

        return ids

    def decode(self, ids):
        """Decode token IDs to text"""
        tokens = b""
        for idx in ids:
            if idx in self.vocab:
                tokens += self.vocab[idx]
            else:
                # Fallback for unknown tokens
                tokens += b"?"

        try:
            return tokens.decode('utf-8', errors='replace')
        except:
            return str(tokens)

    def save(self, path):
        """Save tokenizer"""
        with open(path, 'wb') as f:
            pickle.dump({
                'merges': self.merges,
                'vocab': self.vocab,
                'vocab_size': self.vocab_size
            }, f)

    def load(self, path):
        """Load tokenizer"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.merges = data['merges']
            self.vocab = data['vocab']
            self.vocab_size = data['vocab_size']
            self.inverse_vocab = {v: k for k, v in self.vocab.items()}

def create_optimized_tokenizer(text, vocab_size=8000):
    """Create and train optimized BPE tokenizer"""
    print(f"🏗️ Training BPE tokenizer with vocab_size={vocab_size}...")

    tokenizer = BPETokenizer(vocab_size=vocab_size)
    tokenizer.train(text, verbose=True)

    # Test the tokenizer
    test_text = "Hello, how are you doing today?"
    encoded = tokenizer.encode(test_text)
    decoded = tokenizer.decode(encoded)

    print(f"📝 Test encoding: '{test_text}'")
    print(f"🔢 Encoded: {encoded[:20]}... ({len(encoded)} tokens)")
    print(f"📝 Decoded: '{decoded}'")

    # Calculate compression ratio
    char_tokens = len(test_text.encode('utf-8'))
    bpe_tokens = len(encoded)
    compression = char_tokens / bpe_tokens
    print(f"🗜️  Compression ratio: {compression:.2f}x")

    return tokenizer

if __name__ == "__main__":
    # Test tokenizer
    sample_text = "Hello world! This is a test of the BPE tokenizer. " * 100
    tokenizer = create_optimized_tokenizer(sample_text)
    tokenizer.save("bpe_tokenizer.pkl")
