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

ckpt_path = r"C:\02_QUILLAN\checkpoints\checkpoints_sft\quillan_fluent_aligned.pt"
print(f"[*] Loading aligned model checkpoint: {ckpt_path}")
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
sd = ckpt.get("model_state_dict", ckpt)
msd = model.state_dict()
copied = 0
for k, v in sd.items():
    if k in msd and v.shape == msd[k].shape:
        msd[k].copy_(v)
        copied += 1
model.load_state_dict(msd)
print(f"[+] Successfully loaded {copied}/{len(msd)} parameter layers.")
model.eval()

def generate_clean_response(prompt, max_tokens=80, temp=0.6, top_p=0.88, rep_penalty=1.3):
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

    full_output = enc.decode(generated[len(tokens):])
    return full_output

prompts = [
    "<|user|>\nHello! Who are you?\n<|assistant|>\n",
    "<|user|>\nWhat can you help me with?\n<|assistant|>\n",
    "<|user|>\nExplain quantum entanglement in simple terms.\n<|assistant|>\n",
    "<|user|>\nWrite a Python function to add two numbers.\n<|assistant|>\n"
]

print("\n==================================================================")
print("   👑 QUILLAN-RONIN v5.3.1 - ALIGNED FLUENT SPEECH DEMO")
print("==================================================================")

for p in prompts:
    print(f"\nPROMPT:\n{p.strip()}")
    resp = generate_clean_response(p, max_tokens=70, temp=0.6, top_p=0.88, rep_penalty=1.3)
    print("------------------------------------------")
    print(f"RESPONSE:\n{resp.strip()}")
    print("==========================================")
