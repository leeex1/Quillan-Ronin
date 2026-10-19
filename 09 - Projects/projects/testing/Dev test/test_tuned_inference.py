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

tik_enc = tiktoken.get_encoding("gpt2")
cfg = QuillanArchConfig(hidden_dim=1024, ffn_dim=2048, num_experts=34, text_only=True, eggroll_rank=256)
model = QuillanRoninSovereign(cfg).to("cpu")

ckpt_path = r"C:\02_QUILLAN\checkpoints\checkpoints_sft\quillan_final_best.pt"
print(f"[*] Loading state from: {ckpt_path}")
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
sd = ckpt.get("model_state_dict", ckpt)
msd = model.state_dict()
for k, v in sd.items():
    if k in msd and v.shape == msd[k].shape:
        msd[k].copy_(v)
model.load_state_dict(msd)
model.eval()

def generate_tuned(prompt, max_tokens=80, temp=0.75, top_p=0.9, rep_penalty=1.4, top_k=50):
    tokens = tik_enc.encode(prompt)
    generated = list(tokens)
    
    print(f"\nPROMPT: {prompt.strip()}")
    print("RESPONSE: ", end="", flush=True)
    
    with torch.no_grad():
        for _ in range(max_tokens):
            inp = torch.tensor([generated[-256:]], dtype=torch.long)
            logits = model(inp)[:, -1, :].clone()
            
            # Apply Repetition Penalty to recently generated tokens
            for tid in set(generated[-32:]):
                if logits[0, tid] < 0:
                    logits[0, tid] *= rep_penalty
                else:
                    logits[0, tid] /= rep_penalty

            # Temperature
            if temp > 0:
                logits = logits / temp
                # Top-K
                if top_k > 0:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = float('-inf')
                # Top-P (nucleus)
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
            word = tik_enc.decode([next_tok])
            print(word, end="", flush=True)
            if next_tok == 50256:
                break

    print("\n" + "-" * 50)

test_prompts = [
    "<|user|>\nHello! Who are you?\n<|assistant|>\n",
    "<|user|>\nWhat can you do?\n<|assistant|>\n",
    "Once upon a time in a digital world,",
    "The core purpose of artificial intelligence is to"
]

for p in test_prompts:
    generate_tuned(p, max_tokens=70, temp=0.75, top_p=0.9, rep_penalty=1.4, top_k=40)
