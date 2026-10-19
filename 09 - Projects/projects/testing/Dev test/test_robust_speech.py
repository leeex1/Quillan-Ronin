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
print("   👑 QUILLAN-RONIN v5.3.1 — ROBUST LONG-FORM SPEECH TEST")
print("==================================================================")

enc = tiktoken.get_encoding("gpt2")

cfg = QuillanArchConfig(
    hidden_dim=1024, ffn_dim=2048, num_experts=34,
    text_only=True, eggroll_rank=256
)
model = QuillanRoninSovereign(cfg).to("cpu")

ckpt_path = r"C:\02_QUILLAN\checkpoints\checkpoints_sft\quillan_fluent_aligned.pt"
if not os.path.exists(ckpt_path):
    ckpt_path = r"C:\02_QUILLAN\checkpoints\checkpoints_sft\quillan_final_best.pt"

print(f"[*] Loading model checkpoint: {ckpt_path}")
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
sd = ckpt.get("model_state_dict", ckpt)
msd = model.state_dict()
copied = 0
for k, v in sd.items():
    if k in msd and v.shape == msd[k].shape:
        msd[k].copy_(v)
        copied += 1
model.load_state_dict(msd)
print(f"[+] Loaded {copied}/{len(msd)} parameter layers.")
model.eval()

def generate_robust(prompt, max_new_tokens=300, temp=0.7, top_p=0.9, rep_penalty=1.35):
    tokens = enc.encode(prompt)
    generated = list(tokens)
    
    print(f"\n==================================================")
    print(f"PROMPT:\n{prompt.strip()}")
    print("==================================================")
    print("RESPONSE:\n", end="", flush=True)
    
    t0 = time.time()
    with torch.no_grad():
        for i in range(max_new_tokens):
            inp = torch.tensor([generated[-128:]], dtype=torch.long)
            logits = model(inp)[:, -1, :].clone()
            
            # Repetition penalty on space and recently generated tokens
            for tid in set(generated[-32:]):
                if logits[0, tid] < 0:
                    logits[0, tid] *= rep_penalty
                else:
                    logits[0, tid] /= rep_penalty

            # Suppress consecutive space tokens (ID 220)
            if len(generated) > 0 and generated[-1] == 220:
                logits[0, 220] = float('-inf')

            if temp > 0:
                scaled_logits = logits / temp
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
            else:
                next_tok = torch.argmax(logits, dim=-1).item()

            generated.append(next_tok)
            word_bytes = enc.decode_bytes([next_tok])
            word_str = word_bytes.decode('utf-8', errors='ignore')
            print(word_str, end="", flush=True)

            if next_tok == 50256:
                break

    t1 = time.time()
    gen_len = len(generated) - len(tokens)
    print(f"\n\n--------------------------------------------------")
    print(f"[{gen_len} tokens generated | Total Time: {t1-t0:.1f}s]")

prompts = [
    "<|user|>\nPlease introduce yourself in detail. Explain your architecture, your 34 Council Experts, your 9-Vector cognitive decomposition, and how you assist developers.\n<|assistant|>\n",
    "<|user|>\nWrite a Python script to build a simple web server.\n<|assistant|>\n"
]

for p in prompts:
    generate_robust(p, max_new_tokens=250, temp=0.7, top_p=0.9, rep_penalty=1.35)
