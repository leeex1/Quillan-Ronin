import os, sys, json, torch
from tokenizers import Tokenizer

tok_path = r'C:\02_QUILLAN\02 - Knowledge Foundation\knowledge\legacy\00 - Meta\oni\tokenizer.json'
tok = Tokenizer.from_file(tok_path)

data_files = [
    r'C:\02_QUILLAN\03 - Training & Model\training_data\sovereign_thinking_gold.jsonl',
    r'C:\02_QUILLAN\03 - Training & Model\training_data\Quillan_Direct_Answers_Gold.jsonl',
    r'C:\02_QUILLAN\03 - Training & Model\training_data\Quillan_Universal_Sovereign_Gold_1000.jsonl'
]

samples = []
for p in data_files:
    if os.path.exists(p):
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                d = json.loads(line)
                q = d.get("question", d.get("prompt", "")).strip()
                a = d.get("response", d.get("answer", "")).strip()
                if not q or not a: continue
                
                if not q.startswith("Question:"):
                    prompt_str = f"Question: {q}\nAnswer:\n"
                elif not q.endswith("Answer:\n"):
                    prompt_str = f"{q}\nAnswer:\n"
                else:
                    prompt_str = q
                    
                target_str = a
                if not target_str.endswith("<|im_end|>"):
                    target_str += "<|im_end|>"
                    
                p_ids = tok.encode(prompt_str).ids
                a_ids = tok.encode(target_str).ids
                
                full_ids = p_ids + a_ids
                if len(full_ids) > 160:
                    full_ids = full_ids[:160]
                    
                input_ids = torch.tensor(full_ids[:-1], dtype=torch.long)
                target_ids = torch.tensor(full_ids[1:], dtype=torch.long)
                
                prompt_len = min(len(p_ids) - 1, len(target_ids))
                if prompt_len > 0:
                    target_ids[:prompt_len] = -100
                    
                if (target_ids != -100).sum() > 0:
                    samples.append((input_ids, target_ids))

print(f"Loaded {len(samples)} gold samples successfully.")
print(f"Sample 0 input length: {len(samples[0][0])}, non-masked targets: {(samples[0][1] != -100).sum().item()}")

