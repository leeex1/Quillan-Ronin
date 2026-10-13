#!/usr/bin/env python3
"""Train the legacy Quillan BPE tokenizer properly on a corpus with text field."""
import os, sys, pickle
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from quillan_bpe_tokenizer import QuillanBPETokenizer, build_tokenizer

ROOT = Path(__file__).resolve().parent

# Use GPT_5.5_Distilled.jsonl which has "text" field
CORPUS = ROOT / "training_data" / "GPT_5.5_Distilled.jsonl"
if not CORPUS.exists():
    CORPUS = ROOT / "training_data" / "quillan_corpus_CLEAN_V7.jsonl"

OUTPUT = ROOT / "quillan_bpe_tokenizer.pkl"

print(f"[*] Training tokenizer on {CORPUS.name} ({CORPUS.stat().st_size / 1e6:.0f} MB)")
tok = build_tokenizer(str(CORPUS), vocab_size=50257, save_path=str(OUTPUT))

print(f"\n[*] Verify saved file...")
with open(OUTPUT, 'rb') as f:
    data = pickle.load(f)
print(f"  merges: {len(data['merges'])}")
print(f"  vocab: {len(data['vocab'])}")
print(f"  vocab_size: {data['vocab_size']}")

tok2 = QuillanBPETokenizer()
tok2.load(str(OUTPUT))
test = tok2.encode("Hello, how are you doing today?")
print(f"  encode: {test[:15]}... ({len(test)} tokens)")
ratio = len("Hello, how are you doing today?".encode('utf-8')) / max(len(test), 1)
print(f"  compression: {ratio:.2f}x")
