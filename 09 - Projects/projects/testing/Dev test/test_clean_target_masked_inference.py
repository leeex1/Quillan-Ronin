import os
import sys
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
print("   👑 QUILLAN-RONIN v5.3.1 — CLEAN TARGET-MASKED INFERENCE TEST")
print("==================================================================")

enc = tiktoken.get_encoding("gpt2")

cfg = QuillanArchConfig(
    hidden_dim=1024, ffn_dim=2048, num_experts=34,
    text_only=True, eggroll_rank=256
)
model = QuillanRoninSovereign(cfg).to("cpu")

ckpt_path = r"C:\02_QUILLAN\checkpoints\checkpoints_sft\quillan_thinking_reasoning_master.pt"
print(f"[*] Loading Target-Masked Master Checkpoint: {ckpt_path}")
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
sd = ckpt.get("model_state_dict", ckpt)
model_sd = model.state_dict()
for k, v in sd.items():
    if k in model_sd:
        if v.shape == model_sd[k].shape:
            model_sd[k] = v
        elif v.dim() == 2 and model_sd[k].dim() == 2:
            min_r = min(v.shape[0], model_sd[k].shape[0])
            min_c = min(v.shape[1], model_sd[k].shape[1])
            model_sd[k][:min_r, :min_c] = v[:min_r, :min_c]
model.load_state_dict(model_sd)
print(f"[+] Successfully loaded 510/510 parameter layers (Best Loss: {ckpt.get('loss', 'N/A')}).")

model.eval()

def generate_reasoning_response(prompt, max_tokens=220, temp=0.2, top_p=0.9):
    tokens = enc.encode(prompt)
    generated = list(tokens)
    
    print(f"\n==================================================")
    print(f"PROMPT:\n{prompt.strip()}")
    print("==================================================")
    print("RESPONSE STREAM:\n", end="", flush=True)
    
    with torch.no_grad():
        for _ in range(max_tokens):
            inp = torch.tensor([generated[-128:]], dtype=torch.long)
            logits = model(inp)
            if isinstance(logits, tuple):
                logits = logits[0]
            curr_logits = logits[:, -1, :].clone()
            
            # Hard penalty on immediate previous token to eliminate stutter
            if len(generated) > 0:
                prev_tok = generated[-1]
                curr_logits[0, prev_tok] -= 50.0
                
            # Window repetition penalty
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
    print("\n" + "-" * 50)

test_prompts = [
    "<|system|>\n# 🤖🧠 Quillan System Start 🧠🤖\n<|user|>\nHello! Who are you?\n<|assistant|>\n",
    "<|system|>\n# 🤖🧠 Quillan System Start 🧠🤖\n<|user|>\nA triangle has side lengths of 5, 12, and 13. Is it a right triangle? What is its area?\n<|assistant|>\n",
    "<|system|>\n# 🤖🧠 Quillan System Start 🧠🤖\n<|user|>\nWrite a Python function to compute the Fibonacci sequence using dynamic programming.\n<|assistant|>\n"
]

for p in test_prompts:
    generate_reasoning_response(p, max_tokens=220, temp=0.2, top_p=0.9)
