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

enc = tiktoken.get_encoding("gpt2")

cfg = QuillanArchConfig(
    hidden_dim=1024, ffn_dim=2048, num_experts=34,
    text_only=True, eggroll_rank=256
)
model = QuillanRoninSovereign(cfg).to("cpu")

ckpt_path = r"C:\02_QUILLAN\checkpoints\checkpoints_sft\quillan_longform_best.pt"
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

def generate_speech(prompt, max_tokens=150, temp=0.6, top_p=0.88, rep_penalty=1.35):
    tokens = enc.encode(prompt)
    generated = list(tokens)
    
    with torch.no_grad():
        for _ in range(max_tokens):
            inp = torch.tensor([generated[-128:]], dtype=torch.long)
            logits = model(inp)[:, -1, :].clone()
            
            for tid in set(generated[-32:]):
                if logits[0, tid] < 0:
                    logits[0, tid] *= rep_penalty
                else:
                    logits[0, tid] /= rep_penalty

            if generated[-1] == 220:
                logits[0, 220] = float('-inf')

            if temp > 0:
                logits = logits / temp
                probs = F.softmax(logits, dim=-1)
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                cum_probs = torch.cumsum(sorted_probs, dim=-1)
                sorted_indices_to_remove = cum_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float('-inf')
                probs = F.softmax(logits, dim=-1)
                next_tok = torch.multinomial(probs, 1).item()
            else:
                next_tok = torch.argmax(logits, dim=-1).item()

            generated.append(next_tok)
            if next_tok == 50256:
                break

    raw_bytes = enc.decode_bytes(generated[len(tokens):])
    full_output = raw_bytes.decode('utf-8', errors='ignore')
    return full_output

prompts = [
    "<|user|>\nPlease introduce yourself in detail. Explain your architecture, your 34 Council Experts, your 9-Vector cognitive decomposition, and how you assist developers.\n<|assistant|>\n",
    "<|user|>\nWrite a Python function to implement quicksort with step-by-step explanations.\n<|assistant|>\n"
]

print("\n==================================================================")
print("   👑 QUILLAN-RONIN v5.3.1 — LONG-FORM ALIGNED SPEECH DEMO")
print("==================================================================")

for p in prompts:
    print(f"\nPROMPT:\n{p.strip()}")
    resp = generate_speech(p, max_tokens=140, temp=0.6, top_p=0.88, rep_penalty=1.35)
    print("------------------------------------------")
    print(f"RESPONSE:\n{resp.strip()}")
    print("==========================================")
