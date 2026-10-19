import os
import sys
import time
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
print("   👑 QUILLAN-RONIN v5.3.1 — PRODUCTION INFERENCE EVALUATION")
print("==================================================================")

enc = tiktoken.get_encoding("gpt2")

cfg = QuillanArchConfig(
    hidden_dim=1024, ffn_dim=2048, num_experts=34,
    text_only=True, eggroll_rank=256
)
model = QuillanRoninSovereign(cfg).to("cpu")

ckpt_path = r"C:\02_QUILLAN\checkpoints\checkpoints_sft\quillan_final_master_v531.pt"
print(f"[*] Loading Production Checkpoint: {ckpt_path}")
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
sd = ckpt.get("model_state_dict", ckpt)
model.load_state_dict(sd)
model.eval()
print("[+] Production model loaded successfully.")

def generate_production(prompt, max_tokens=150, temp=0.0, top_p=0.9, repetition_penalty=1.2):
    tokens = enc.encode(prompt)
    generated = list(tokens)
    
    print(f"\n==================================================")
    print(f"PROMPT:\n{prompt.strip()}")
    print(f"SAMPLING MODE: temp={temp}, top_p={top_p}, rep_pen={repetition_penalty}")
    print("==================================================")
    print("RESPONSE:\n", end="", flush=True)
    
    with torch.no_grad():
        for _ in range(max_tokens):
            inp = torch.tensor([generated[-128:]], dtype=torch.long)
            logits = model(inp)
            if isinstance(logits, tuple):
                logits = logits[0]
            curr_logits = logits[:, -1, :].clone()
            
            # Multiplicative repetition penalty
            for tid in set(generated[-64:]):
                if curr_logits[0, tid] < 0:
                    curr_logits[0, tid] *= repetition_penalty
                else:
                    curr_logits[0, tid] /= repetition_penalty

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
    print("\n" + "-" * 50)

print("\n=== EVALUATING GREEDY DECODING vs NUCLEUS SAMPLING ===")

# Test 1: Greedy Decoding
generate_production("<|user|>\nHello! Who are you?\n<|assistant|>\n", max_tokens=100, temp=0.0)

# Test 2: Low-Temperature Nucleus Sampling
generate_production("<|user|>\nHello! Who are you?\n<|assistant|>\n", max_tokens=100, temp=0.4, top_p=0.85, repetition_penalty=1.3)

# Test 3: Capabilities Question
generate_production("<|user|>\nExplain your 34 Council Experts.\n<|assistant|>\n", max_tokens=120, temp=0.3, top_p=0.85, repetition_penalty=1.3)
