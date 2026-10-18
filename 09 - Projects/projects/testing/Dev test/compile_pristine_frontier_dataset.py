#!/usr/bin/env python3
"""
👑 Compile Pristine 37K+ Frontier Instruction Dataset
Parses raw JSONL files (Claude-Opus code, instruct, GPT-5.5 distilled, science)
and tokenizes with tiktoken gpt2 with exact target masking.
"""

import json, sys, os, torch, tiktoken
from pathlib import Path

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

REPO_ROOT = Path(r"C:\02_QUILLAN")
DATA_DIR = REPO_ROOT / "training_data"
out_pt = DATA_DIR / "pristine_frontier_gold_37k.pt"

enc = tiktoken.get_encoding("gpt2")
MAX_SEQ_LEN = 256
EOT_TOKEN = enc.eot_token

all_samples = []

print("==================================================================", flush=True)
print("   👑 COMPILING PRISTINE FRONTIER INSTRUCTION DATASET (37K+)", flush=True)
print("==================================================================\n", flush=True)

# 1. Parse code_train.jsonl, instruct_train.jsonl, full_train.jsonl
for fname in ['code_train.jsonl', 'instruct_train.jsonl', 'full_train.jsonl']:
    fpath = DATA_DIR / fname
    if not fpath.exists(): continue
    print(f"[*] Processing {fname}...", flush=True)
    count = 0
    with open(fpath, 'r', encoding='utf-8', errors='ignore') as fp:
        for line in fp:
            line = line.strip()
            if not line: continue
            try:
                d = json.loads(line)
                user_msg = ""
                asst_msg = ""
                
                if 'messages' in d:
                    msgs = d['messages']
                    if isinstance(msgs, str):
                        try:
                            import ast
                            msgs = ast.literal_eval(msgs)
                        except Exception:
                            continue
                    for m in msgs:
                        role = m.get('role', '')
                        content = m.get('content', '')
                        if role == 'user': user_msg = content
                        elif role == 'assistant': asst_msg = content
                elif 'text' in d:
                    text = d['text']
                    if '<|user|>' in text and '<|assistant|>' in text:
                        parts = text.split('<|assistant|>')
                        user_msg = parts[0].replace('<|user|>', '').strip()
                        asst_msg = parts[1].replace('<|endoftext|>', '').strip()
                    elif 'Question:' in text and 'Answer:' in text:
                        parts = text.split('Answer:')
                        user_msg = parts[0].replace('Question:', '').strip()
                        asst_msg = parts[1].replace('<|endoftext|>', '').strip()

                if user_msg and asst_msg:
                    all_samples.append((user_msg, asst_msg))
                    count += 1
            except Exception:
                continue
    print(f"    [+] Extracted {count:,} samples from {fname}", flush=True)

# 2. Parse GPT_5.5_Distilled.jsonl
fpath = DATA_DIR / "GPT_5.5_Distilled.jsonl"
if fpath.exists():
    print(f"[*] Processing GPT_5.5_Distilled.jsonl...", flush=True)
    count = 0
    with open(fpath, 'r', encoding='utf-8', errors='ignore') as fp:
        for line in fp:
            line = line.strip()
            if not line: continue
            try:
                d = json.loads(line)
                text = d.get('text', '')
                if '<|user|>' in text and '<|assistant|>' in text:
                    parts = text.split('<|assistant|>')
                    u = parts[0].replace('<|user|>', '').strip()
                    a = parts[1].replace('<|endoftext|>', '').strip()
                    if u and a:
                        all_samples.append((u, a))
                        count += 1
            except Exception:
                continue
    print(f"    [+] Extracted {count:,} samples from GPT_5.5_Distilled.jsonl", flush=True)

# 3. Parse quillan_science_absolute.jsonl
fpath = DATA_DIR / "quillan_science_absolute.jsonl"
if fpath.exists():
    print(f"[*] Processing quillan_science_absolute.jsonl...", flush=True)
    count = 0
    with open(fpath, 'r', encoding='utf-8', errors='ignore') as fp:
        for line in fp:
            line = line.strip()
            if not line: continue
            try:
                d = json.loads(line)
                text = d.get('text', '')
                if 'SOLUTION:' in text:
                    parts = text.split('SOLUTION:')
                    u = parts[0].strip()
                    a = parts[1].strip()
                    if u and a:
                        all_samples.append((u, a))
                        count += 1
                elif 'Answer:' in text:
                    parts = text.split('Answer:')
                    u = parts[0].strip()
                    a = parts[1].strip()
                    if u and a:
                        all_samples.append((u, a))
                        count += 1
            except Exception:
                continue
    print(f"    [+] Extracted {count:,} samples from quillan_science_absolute.jsonl", flush=True)

# 4. Parse reasoning gold
fpath = DATA_DIR / "Quillan_Clean_Reasoning_Gold_Dataset.jsonl"
if fpath.exists():
    print(f"[*] Processing Quillan_Clean_Reasoning_Gold_Dataset.jsonl...", flush=True)
    count = 0
    with open(fpath, 'r', encoding='utf-8', errors='ignore') as fp:
        for line in fp:
            line = line.strip()
            if not line: continue
            try:
                d = json.loads(line)
                u = d.get('prompt', '') or d.get('instruction', '') or d.get('question', '')
                a = d.get('response', '') or d.get('output', '') or d.get('answer', '')
                if u and a:
                    all_samples.append((str(u).strip(), str(a).strip()))
                    count += 1
            except Exception:
                continue
    print(f"    [+] Extracted {count:,} samples from Reasoning Gold", flush=True)

print(f"\n[+] Total Valid Frontier Pairs Collected: {len(all_samples):,}", flush=True)

# 5. Tokenize with Target Masking
print("[*] Tokenizing and building target-masked tensors...", flush=True)
input_ids_list = []
labels_list = []

for u, a in all_samples:
    prompt_str = f"<|user|>\n{u}\n<|assistant|>\n"
    asst_str = f"{a}"
    
    p_toks = enc.encode(prompt_str, disallowed_special=())
    a_toks = enc.encode(asst_str, disallowed_special=()) + [EOT_TOKEN]
    
    # Truncate if too long
    if len(p_toks) + len(a_toks) > MAX_SEQ_LEN:
        max_a = MAX_SEQ_LEN - min(len(p_toks), 100)
        p_toks = p_toks[:100]
        a_toks = a_toks[:max_a]
    
    seq_toks = p_toks + a_toks
    labels = [-100] * len(p_toks) + a_toks
    
    # Pad to MAX_SEQ_LEN
    pad_len = MAX_SEQ_LEN - len(seq_toks)
    if pad_len > 0:
        seq_toks = seq_toks + [0] * pad_len
        labels = labels + [-100] * pad_len
        
    input_ids_list.append(torch.tensor(seq_toks[:MAX_SEQ_LEN], dtype=torch.long))
    labels_list.append(torch.tensor(labels[:MAX_SEQ_LEN], dtype=torch.long))

input_ids_tensor = torch.stack(input_ids_list)
labels_tensor = torch.stack(labels_list)

print(f"[+] Final Tensor Shape: {input_ids_tensor.shape}", flush=True)

torch.save({
    'input_ids': input_ids_tensor,
    'labels': labels_tensor,
    'num_samples': len(input_ids_list),
    'max_seq_len': MAX_SEQ_LEN
}, out_pt)

print(f"[+] Saved pristine gold dataset to {out_pt.name} ({out_pt.stat().st_size / 1e6:.1f} MB)!\n", flush=True)
