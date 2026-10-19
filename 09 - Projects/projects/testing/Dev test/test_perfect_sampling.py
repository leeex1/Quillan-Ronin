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

def generate_perfect(prompt, max_tokens=150, temp=0.7, top_p=0.9, top_k=40, penalty_sub=3.5):
    tokens = enc.encode(prompt)
    generated = list(tokens)
    
    with torch.no_grad():
        for _ in range(max_tokens):
            inp = torch.tensor([generated[-128:]], dtype=torch.long)
            logits = model(inp)[:, -1, :].clone()
            
            # Constant logit subtraction penalty on recent tokens
            recent_tokens = generated[-40:]
            for tid in set(recent_tokens):
                count = recent_tokens.count(tid)
                logits[0, tid] -= (penalty_sub * count)

            # Suppress consecutive space tokens (ID 220)
            if generated[-1] == 220:
                logits[0, 220] = float('-inf')

            if temp > 0:
                scaled_logits = logits / temp
                # Top-K
                if top_k > 0:
                    v, _ = torch.topk(scaled_logits, min(top_k, scaled_logits.size(-1)))
                    scaled_logits[scaled_logits < v[:, [-1]]] = float('-inf')
                # Top-P
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
            if next_tok == 50256:
                break

    raw_bytes = enc.decode_bytes(generated[len(tokens):])
    full_output = raw_bytes.decode('utf-8', errors='ignore')
    return full_output

prompts = [
    "<|user|>\nPlease introduce yourself in detail. Explain your architecture, your 34 Council Experts, your 9-Vector cognitive decomposition, and how you assist developers.\n<|assistant|>\n",
    "<|user|>\nWrite a Python function to add two numbers with explanations.\n<|assistant|>\n"
]

print("\n==================================================================")
print("   👑 QUILLAN-RONIN v5.3.1 — PERFECT PENALTY SPEECH DEMO")
print("==================================================================")

for p in prompts:
    print(f"\nPROMPT:\n{p.strip()}")
    resp = generate_perfect(p, max_tokens=140, temp=0.7, top_p=0.9, top_k=40, penalty_sub=3.5)
    print("------------------------------------------")
    print(f"RESPONSE:\n{resp.strip()}")
    print("==========================================")
