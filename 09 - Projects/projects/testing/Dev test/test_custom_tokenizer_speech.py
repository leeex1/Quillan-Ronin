import os
import sys
import json
import torch
import torch.nn.functional as F
from pathlib import Path

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

REPO_ROOT = Path(r"C:\02_QUILLAN")
sys.path.insert(0, str(REPO_ROOT))

try:
    from tokenizers import Tokenizer
    TOK_AVAILABLE = True
except ImportError:
    TOK_AVAILABLE = False

print("==================================================================")
print("   👑 TESTING QUILLAN CUSTOM TOKENIZER & MODEL ALIGNMENT")
print("==================================================================")

tok_path = REPO_ROOT / "training_data" / "tokenizer.json"

if TOK_AVAILABLE and tok_path.exists():
    tokenizer = Tokenizer.from_file(str(tok_path))
    vocab_size = tokenizer.get_vocab_size()
    print(f"[+] Custom Tokenizer Loaded Successfully! Vocab Size: {vocab_size}")
    
    sample_text = "Hello! I am Quillan-Ronin v5.3.1."
    encoded = tokenizer.encode(sample_text)
    decoded = tokenizer.decode(encoded.ids)
    print(f"    Sample Input:  '{sample_text}'")
    print(f"    Encoded IDs:   {encoded.ids[:10]}")
    print(f"    Decoded Text:  '{decoded}'")
else:
    print(f"[!] Tokenizer library available: {TOK_AVAILABLE}, File exists: {tok_path.exists()}")

print("\n[+] Custom Tokenizer Test Completed!")
