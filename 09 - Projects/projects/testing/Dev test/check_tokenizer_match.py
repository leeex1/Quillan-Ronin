#!/usr/bin/env python3
import torch, tiktoken, json
from pathlib import Path

enc_gpt2 = tiktoken.get_encoding("gpt2")

pt_file = Path(r"C:\02_QUILLAN\training_data\quillan_corpus_CLEAN_V7.pt")
data = torch.load(pt_file, map_location="cpu", weights_only=False)

sample_tokens = data[:50].tolist()
print("Sample tokens from pt file:", sample_tokens[:20])
print("\nDecoded with GPT-2 tiktoken:")
try:
    print(enc_gpt2.decode(sample_tokens))
except Exception as e:
    print("GPT-2 decode error:", e)

# Check custom tokenizers in repo
tok_json = Path(r"C:\02_QUILLAN\Quillan-v4.2-model\tokenizer.json")
if tok_json.exists():
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(tok_json))
    print("\nDecoded with Quillan-v4.2 tokenizer.json:")
    try:
        print(tok.decode(sample_tokens))
    except Exception as e:
        print("Custom decode error:", e)
