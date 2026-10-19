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
print("   👑 QUILLAN-RONIN v5.3.1 — LONG-FORM PRODUCTION SPEECH TEST")
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

def generate_long_form(prompt, max_tokens=1000, temp=0.5, top_p=0.88):
    tokens = enc.encode(prompt)
    generated = list(tokens)
    
    print(f"\n==================================================")
    print(f"LONG-FORM PROMPT:\n{prompt.strip()}")
    print("==================================================")
    print("RESPONSE STREAM:\n", end="", flush=True)
    
    t0 = time.time()
    tokens_generated = 0
    
    with torch.no_grad():
        for _ in range(max_tokens):
            inp = torch.tensor([generated[-128:]], dtype=torch.long)
            logits = model(inp)
            if isinstance(logits, tuple):
                logits = logits[0]
            curr_logits = logits[:, -1, :].clone()
            
            # Constant subtraction penalty for repetition control
            recent_tokens = generated[-32:]
            for tid in set(recent_tokens):
                count = recent_tokens.count(tid)
                curr_logits[0, tid] -= (2.5 * count)

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
            tokens_generated += 1
            
            word_bytes = enc.decode_bytes([next_tok])
            word_str = word_bytes.decode('utf-8', errors='ignore')
            print(word_str, end="", flush=True)

            if next_tok == 50256:
                break
                
    elapsed = time.time() - t0
    tok_per_sec = tokens_generated / elapsed if elapsed > 0 else 0
    print(f"\n" + "-" * 50)
    print(f"[STATS] Generated {tokens_generated} tokens in {elapsed:.2f}s ({tok_per_sec:.1f} tok/s)\n")

print("\n=== STARTING 1,000-TOKEN LONG-FORM GENERATION ===")

prompts = [
    "<|system|>\n# 🤖🧠 Quillan System Start 🧠🤖\n<|user|>\nWrite a comprehensive, multi-paragraph essay detailing how the 34 Council Experts, 9-Vector Prism Decomposition, and Flash Diffusion Core work together in Quillan-Ronin v5.3.1.\n<|assistant|>\n",
    "<|system|>\n# 🤖🧠 Quillan System Start 🧠🤖\n<|user|>\nWrite a complete, fully functioning Python script that implements a BitNet 1.58b ternary linear layer with unit tests and clear docstrings.\n<|assistant|>\n"
]

for p in prompts:
    generate_long_form(p, max_tokens=600, temp=0.5, top_p=0.88)
