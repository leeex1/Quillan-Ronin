import os
import sys
import json
import tiktoken
from pathlib import Path

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

enc = tiktoken.get_encoding("gpt2")

print("==================================================================")
print("   👑 PREPARING LONG-FORM TARGET-MASKED SFT DATASET")
print("==================================================================")

samples = []

# 1. Samurai Seed Dataset
seed_path = r"C:\02_QUILLAN\training_data\Quillan_Ronin_v5.3.1_Samurai_Training_Seed_Dataset.jsonl"
if os.path.exists(seed_path):
    with open(seed_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            obj = json.loads(line.strip())
            q = obj.get("question", "")
            ans = obj.get("final_output", "")
            if q and ans:
                full_txt = f"<|user|>\n{q}\n<|assistant|>\n{ans}\n<|end|>"
                prompt_txt = f"<|user|>\n{q}\n<|assistant|>\n"
                
                toks = enc.encode(full_txt)
                p_toks = enc.encode(prompt_txt)
                
                # Target masking: -100 for prompt tokens
                labs = [-100] * len(p_toks) + toks[len(p_toks):]
                samples.append((toks[:512], labs[:512]))

# 2. Samurai System Prompt Document (Quillan-Samurai.md)
samurai_md_path = r"C:\02_QUILLAN\system prompts\Quillan-Samurai.md"
if os.path.exists(samurai_md_path):
    with open(samurai_md_path, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()
    
    sections = content.split("## ")
    for sec in sections[:15]:
        if not sec.strip(): continue
        lines = sec.strip().split("\n")
        title = lines[0]
        body = "\n".join(lines[1:])
        if len(body) > 100:
            q = f"Explain the principles and architecture of {title} in Quillan-Ronin v5.3.1."
            ans = f"# 🤖🧠 Quillan System Start 🧠🤖\n\n## {title}\n\n{body}"
            full_txt = f"<|user|>\n{q}\n<|assistant|>\n{ans}\n<|end|>"
            prompt_txt = f"<|user|>\n{q}\n<|assistant|>\n"
            
            toks = enc.encode(full_txt)
            p_toks = enc.encode(prompt_txt)
            labs = [-100] * len(p_toks) + toks[len(p_toks):]
            samples.append((toks[:512], labs[:512]))

# 3. GPT 5.5 Distilled QA Dataset
distilled_path = r"C:\02_QUILLAN\training_data\GPT_5.5_Distilled.jsonl"
if os.path.exists(distilled_path):
    with open(distilled_path, "r", encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f):
            if i >= 300: break
            if not line.strip(): continue
            try:
                obj = json.loads(line.strip())
                if "prompt" in obj and "response" in obj:
                    q = obj["prompt"]
                    ans = obj["response"]
                    full_txt = f"<|user|>\n{q}\n<|assistant|>\n{ans}\n<|end|>"
                    prompt_txt = f"<|user|>\n{q}\n<|assistant|>\n"
                    toks = enc.encode(full_txt)
                    p_toks = enc.encode(prompt_txt)
                    labs = [-100] * len(p_toks) + toks[len(p_toks):]
                    samples.append((toks[:512], labs[:512]))
            except Exception:
                pass

print(f"[+] Total Prepared Long-Form Target-Masked Samples: {len(samples)}")

out_file = r"C:\02_QUILLAN\training_data\long_form_sft_dataset.pt"
import torch
torch.save(samples, out_file)
print(f"[+] Saved dataset file to: {out_file}")
