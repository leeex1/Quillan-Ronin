import json
import glob
import os
import tiktoken

enc = tiktoken.get_encoding("gpt2")
data_dir = r"C:\02_QUILLAN\training_data"

print("--- SEARCHING FOR DIALOGUE HEADERS IN DATASETS ---")
for fpath in glob.glob(os.path.join(data_dir, "*.jsonl")):
    fname = os.path.basename(fpath)
    count = 0
    has_user = 0
    with open(fpath, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            count += 1
            if "<|user|>" in line or "User:" in line or "Instruction:" in line:
                has_user += 1
            if count >= 500: break
    print(f"{fname:35s}: {count} lines checked, {has_user} have dialogue headers")
