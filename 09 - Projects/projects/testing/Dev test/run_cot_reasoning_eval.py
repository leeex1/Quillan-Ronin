import os
import sys
import time
import math
import torch
import torch.nn.functional as F
import tiktoken
from pathlib import Path

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

REPO_ROOT = Path(r"C:\02_QUILLAN")
sys.path.insert(0, str(REPO_ROOT))

from _dev.quillan_v8_saturated import QuillanRoninSovereign, QuillanArchConfig

print("==================================================================")
print("   👑 QUILLAN-RONIN v5.3.1 — 10-QUESTION REASONING EVALUATION TEST")
print("==================================================================")

enc = tiktoken.get_encoding("gpt2")

cfg = QuillanArchConfig(
    hidden_dim=1024, ffn_dim=2048, num_experts=34,
    text_only=True, eggroll_rank=256
)
model = QuillanRoninSovereign(cfg).to("cpu")

ckpt_path = r"C:\02_QUILLAN\checkpoints\checkpoints_sft\quillan_thinking_reasoning_master.pt"
print(f"[*] Loading Thinking Reasoning Master Checkpoint: {ckpt_path}")
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
sd = ckpt.get("model_state_dict", ckpt)
model.load_state_dict(sd)
print("[+] Successfully loaded 510/510 parameter layers.")
model.eval()

questions = [
    ("Mathematics", "A triangle has sides of length 5, 12, and 13. Is it a right triangle? What is its area?"),
    ("Physics", "Why is the sky blue during the day and why does it turn red during sunset?"),
    ("Computer Science", "What is the time complexity of QuickSort in the average and worst cases?"),
    ("Logic & Reasoning", "If all A are B, and all B are C, are all A guaranteed to be C? Explain."),
    ("Chemistry", "What happens to the pressure of a gas if its volume is halved at constant temperature?"),
    ("Software Engineering", "Explain the difference between a process and a thread in operating systems."),
    ("Geometry", "What is the volume of a sphere with radius r = 3?"),
    ("Biology", "What is the primary function of mitochondria in eukaryotic cells?"),
    ("Economics", "What is the law of supply and demand in free market economics?"),
    ("General Science", "How does gravity affect light near a massive object like a black hole?")
]

def generate_response(prompt, max_tokens=250, temp=0.25, top_p=0.9):
    tokens = enc.encode(prompt)
    generated = list(tokens)
    
    with torch.no_grad():
        for _ in range(max_tokens):
            inp = torch.tensor([generated[-128:]], dtype=torch.long)
            logits = model(inp)
            if isinstance(logits, tuple):
                logits = logits[0]
            curr_logits = logits[:, -1, :].clone()
            
            # Zero-Stutter penalty on immediate previous token
            if len(generated) > 0:
                prev_tok = generated[-1]
                curr_logits[0, prev_tok] -= 50.0
                
            # Sliding window repetition penalty
            recent_tokens = generated[-48:]
            for tid in set(recent_tokens):
                count = recent_tokens.count(tid)
                curr_logits[0, tid] -= (4.0 * count)

            if temp == 0.0:
                next_tok = torch.argmax(curr_logits, dim=-1).item()
            else:
                scaled_logits = curr_logits / temp
                probs = F.softmax(scaled_logits, dim=-1)
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                cum_probs = torch.cumsum(sorted_probs, dim=-1)
                sorted_indices_to_remove = cum_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                scaled_logits[indices_to_remove] = float('-inf')
                probs = F.softmax(scaled_logits, dim=-1)
                next_tok = torch.multinomial(probs, 1).item()

            generated.append(next_tok)
            word_bytes = enc.decode_bytes([next_tok])
            word_str = word_bytes.decode('utf-8', errors='ignore')
            print(word_str, end="", flush=True)

            if next_tok == 50256:
                break

for idx, (domain, q_text) in enumerate(questions, 1):
    prompt = f"<|system|>\n# 🤖🧠 Quillan System Start 🧠🤖\n<|user|>\n[{domain}] {q_text}\n<|assistant|>\n"
    print(f"\n\n==================================================")
    print(f"QUESTION {idx}/10 [{domain}]: {q_text}")
    print("==================================================")
    generate_response(prompt, max_tokens=180, temp=0.25)

print("\n\n==================================================")
print("   EVALUATION COMPLETE")
print("==================================================")
