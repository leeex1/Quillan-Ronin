import os
import json
import tiktoken
from pathlib import Path

enc = tiktoken.get_encoding("gpt2")
data_dir = Path(r"C:\02_QUILLAN\training_data")
out_file = data_dir / "unified_tokenized_corpus.jsonl"

jsonl_files = [
    "quillan_corpus_CLEAN_V7.jsonl",
    "GPT_5.5_Distilled.jsonl",
    "code_train.jsonl",
    "full_train.jsonl",
    "instruct_train.jsonl",
    "train.jsonl",
    "quillan_science_absolute.jsonl",
    "quillan_science_additional.jsonl",
    "pdf_papers_corpus.jsonl",
    "quillan_12mb_training_dataset.jsonl"
]

print(f"[*] Starting tokenization of all real datasets in {data_dir}...")
total_written = 0

with open(out_file, "w", encoding="utf-8") as out_f:
    for fname in jsonl_files:
        fpath = data_dir / fname
        if not fpath.exists():
            print(f"[!] File not found, skipping: {fname}")
            continue
            
        print(f"[*] Processing {fname} ({fpath.stat().st_size / 1e6:.1f} MB)...")
        file_count = 0
        
        with open(fpath, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line: continue
                try:
                    obj = json.loads(line)
                    
                    # 1. Pre-tokenized input_ids
                    if "input_ids" in obj and len(obj["input_ids"]) >= 32:
                        tokens = obj["input_ids"]
                        labels = obj.get("labels", tokens)
                    elif "tokens" in obj and len(obj["tokens"]) >= 32:
                        tokens = obj["tokens"]
                        labels = obj.get("target_ids", tokens)
                    # 2. Raw text field
                    elif "text" in obj:
                        text = obj["text"]
                        if len(text) < 40: continue
                        tokens = enc.encode(text)
                        labels = list(tokens)
                    # 3. Messages array
                    elif "messages" in obj:
                        formatted = ""
                        for msg in obj["messages"]:
                            r = msg.get("role", "")
                            c = msg.get("content", "")
                            formatted += f"<|{r}|>\n{c}\n"
                        tokens = enc.encode(formatted)
                        labels = list(tokens)
                    # 4. Question / Answer
                    elif "question" in obj and "answer" in obj:
                        q = obj["question"]
                        a = obj["answer"]
                        formatted = f"<|user|>\n{q}\n<|assistant|>\n{a}"
                        tokens = enc.encode(formatted)
                        labels = list(tokens)
                    else:
                        continue
                        
                    if len(tokens) >= 32:
                        out_f.write(json.dumps({"input_ids": tokens[:512], "labels": labels[:512]}) + "\n")
                        file_count += 1
                        total_written += 1
                except Exception:
                    pass
                    
        print(f"  [+] {fname}: extracted & tokenized {file_count} samples")

print(f"\n[COMPLETE] Unified Tokenized Real Corpus compiled: {total_written} total samples written to {out_file}")
