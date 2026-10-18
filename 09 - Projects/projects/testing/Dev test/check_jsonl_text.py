#!/usr/bin/env python3
import json, tiktoken
from pathlib import Path

enc_gpt2 = tiktoken.get_encoding("gpt2")

jsonl_files = [
    Path(r"C:\02_QUILLAN\training_data\GPT_5.5_Distilled.jsonl"),
    Path(r"C:\02_QUILLAN\training_data\instruct_train.jsonl"),
    Path(r"C:\02_QUILLAN\training_data\code_train.jsonl"),
    Path(r"C:\02_QUILLAN\training_data\quillan_corpus_CLEAN_V7.jsonl")
]

print("=== CHECKING RAW JSONL CORPUS SAMPLES ===")
for jf in jsonl_files:
    if jf.exists():
        with open(jf, "r", encoding="utf-8", errors="ignore") as f:
            for i in range(2):
                line = f.readline().strip()
                if line:
                    try:
                        d = json.loads(line)
                        txt = d.get("instruction") or d.get("prompt") or d.get("text") or str(d)
                        toks = enc_gpt2.encode(txt[:100])
                        dec = enc_gpt2.decode(toks)
                        print(f"\n[{jf.name}] Sample {i+1}:")
                        print("  Text:", dec[:80])
                        print("  Token Count:", len(toks))
                    except Exception as e:
                        print(f"[{jf.name}] Error:", e)
