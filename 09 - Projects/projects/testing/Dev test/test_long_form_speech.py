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
print("   👑 QUILLAN-RONIN v5.3.1 — LONG-FORM HIGH-SPEED SPEECH TEST")
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

def generate_long_form(prompt, max_new_tokens=800, temp=0.7, top_p=0.9, rep_penalty=1.2):
    tokens = enc.encode(prompt)
    generated = list(tokens)
    
    print(f"\n==================================================")
    print(f"PROMPT:\n{prompt.strip()}")
    print("==================================================")
    print("RESPONSE:\n", end="", flush=True)
    
    t0 = time.time()
    with torch.no_grad():
        inp = torch.tensor([tokens], dtype=torch.long)
        out = model(inp, past_key_values=None, use_cache=True)
        logits = out[0] if isinstance(out, tuple) else out
        past_kv = out[1] if isinstance(out, tuple) else None
        curr_logits = logits[:, -1, :].clone()
        
        for i in range(max_new_tokens):
            if rep_penalty > 1.0:
                for tid in set(generated[-40:]):
                    if curr_logits[0, tid] < 0:
                        curr_logits[0, tid] *= rep_penalty
                    else:
                        curr_logits[0, tid] /= rep_penalty

            if temp > 0:
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
            else:
                next_tok = torch.argmax(curr_logits, dim=-1).item()

            generated.append(next_tok)
            word_bytes = enc.decode_bytes([next_tok])
            word_str = word_bytes.decode('utf-8', errors='ignore')
            print(word_str, end="", flush=True)

            if next_tok == 50256:
                break
                
            next_inp = torch.tensor([[next_tok]], dtype=torch.long)
            out = model(next_inp, past_key_values=past_kv, use_cache=True)
            curr_logits = (out[0][:, -1, :] if isinstance(out, tuple) else out[:, -1, :]).clone()
            past_kv = out[1] if isinstance(out, tuple) else None

    t1 = time.time()
    gen_len = len(generated) - len(tokens)
    tps = gen_len / max(0.001, (t1 - t0))
    print(f"\n\n--------------------------------------------------")
    print(f"[{gen_len} tokens generated | Speed: {tps:.1f} tok/s | Total Time: {t1-t0:.1f}s]")

prompts = [
    "<|user|>\nPlease introduce yourself in detail. Explain your architecture, your 34 Council Experts, your 9-Vector cognitive decomposition, and how you assist developers.\n<|assistant|>\n",
    "<|user|>\nWrite a complete guide on how to build a scalable web application in Python.\n<|assistant|>\n"
]

for p in prompts:
    generate_long_form(p, max_new_tokens=600, temp=0.7, top_p=0.9, rep_penalty=1.2)
