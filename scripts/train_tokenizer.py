#!/usr/bin/env python3
"""Fast BPE training using HuggingFace tokenizers (Rust). Saves both HF and legacy .pkl format.

Usage: python train_tokenizer_fast.py
"""
import json, os, sys, time, pickle
from pathlib import Path
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, normalizers

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "training_data"
OUTPUT_HF_DIR = ROOT / "quillan_bpe_tokenizer_hf"
OUTPUT_HF_PATH = OUTPUT_HF_DIR / "tokenizer.json"
OUTPUT_LEGACY = ROOT / "quillan_bpe_tokenizer.pkl"
VOCAB_SIZE = 50257

# Use GPT_5.5_Distilled.jsonl (has "text" field, 44MB)
CORPUS = DATA_DIR / "GPT_5.5_Distilled.jsonl"

def train_tokenizer():
    print(f"[*] Training HF BPE tokenizer on {CORPUS.name}...")
    tok = Tokenizer(models.BPE(unk_token="<|endoftext|>"))
    tok.normalizer = normalizers.NFKC()
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        min_frequency=2,
        special_tokens=["<|endoftext|>", "<|pad|>", "<|bos|>"],
    )

    def lines():
        with open(CORPUS, 'r', encoding='utf-8', errors='replace') as f:
            for line in f:
                try:
                    d = json.loads(line)
                    t = d.get('text', '')
                    if len(t) > 10:
                        yield t
                except:
                    pass

    start = time.time()
    tok.train_from_iterator(lines(), trainer=trainer)
    elapsed = time.time() - start
    print(f"[*] Trained in {elapsed:.1f}s")
    print(f"[*] Vocab size: {tok.get_vocab_size()}")

    os.makedirs(OUTPUT_HF_DIR, exist_ok=True)
    tok.save(str(OUTPUT_HF_PATH))
    print(f"[*] Saved: {OUTPUT_HF_PATH}")

    # Build legacy .pkl format: {merges: {pair: idx}, vocab: {id: bytes}, vocab_size: int}
    merges_dict = {}
    for pair, idx in tok.model.merges:
        merges_dict[tuple(pair.tolist() if hasattr(pair, 'tolist') else pair)] = idx

    vocab = tok.get_vocab()
    legacy = {
        "merges": merges_dict,
        "vocab": {i: token.encode('utf-8') for token, i in vocab.items()},
        "vocab_size": VOCAB_SIZE,
    }
    with open(OUTPUT_LEGACY, 'wb') as f:
        pickle.dump(legacy, f)
    print(f"[*] Saved: {OUTPUT_LEGACY} ({os.path.getsize(OUTPUT_LEGACY)/1024:.0f} KB)")

    test = tok.encode("Hello, how are you doing today?")
    compression = len("Hello, how are you doing today?".encode('utf-8')) / max(len(test.ids), 1)
    print(f"[*] Test: '{'Hello, how are you doing today?'}' -> {len(test.ids)} tokens ({compression:.2f}x)")
    return tok

if __name__ == "__main__":
    train_tokenizer()
