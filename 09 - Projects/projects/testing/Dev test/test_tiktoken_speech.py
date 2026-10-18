import os
import sys
import torch
import torch.nn.functional as F
import tiktoken

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

REPO_ROOT = r"C:\02_QUILLAN"
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from _dev.quillan_v8_saturated import QuillanRoninSovereign, QuillanArchConfig

print("==================================================================")
print("   👑 QUILLAN-RONIN v5.3.1 — SPEECH GENERATION TEST (tiktoken gpt2)")
print("==================================================================")

enc = tiktoken.get_encoding("gpt2")
cfg = QuillanArchConfig(hidden_dim=1024, ffn_dim=2048, num_experts=34, text_only=True, eggroll_rank=256)
model = QuillanRoninSovereign(cfg).to("cpu")

ckpt_path = r"C:\02_QUILLAN\checkpoints\checkpoints_sft\quillan_final_best.pt"
print(f"[*] Loading checkpoint from: {ckpt_path}")
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
sd = ckpt.get("model_state_dict", ckpt)
model_sd = model.state_dict()
copied = 0
for k, v in sd.items():
    if k in model_sd and v.shape == model_sd[k].shape:
        model_sd[k].copy_(v)
        copied += 1
model.load_state_dict(model_sd)
print(f"[+] Loaded {copied}/{len(model_sd)} layers (Step: {ckpt.get('step')}, Loss: {ckpt.get('loss'):.4f})")
model.eval()

def test_speech(prompt, max_tokens=100, temp=0.8, top_p=0.9, rep_penalty=1.3):
    tokens = enc.encode(prompt)
    generated = list(tokens)
    
    print(f"\nPROMPT: {prompt.strip()}")
    print("RESPONSE: ", end="", flush=True)
    
    with torch.no_grad():
        for _ in range(max_tokens):
            inp = torch.tensor([generated[-256:]], dtype=torch.long)
            logits = model(inp)[:, -1, :].clone()
            
            # Repetition penalty
            if rep_penalty > 1.0:
                for tid in set(generated[-64:]):
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
                next_tok = torch.multinomial(probs, num_samples=1).item()
            else:
                next_tok = torch.argmax(logits, dim=-1).item()

            generated.append(next_tok)
            word = enc.decode([next_tok])
            print(word, end="", flush=True)
            if next_tok == 50256:
                print(" <EOS>")
                break
    print("\n" + "-" * 50)

prompts = [
    "<|user|>\nHello! Who are you?\n<|assistant|>\n",
    "System: You are Quillan, a sovereign AI assistant.\nUser: What is your name?\nAssistant: ",
    "The secret of artificial intelligence is",
    "def add_numbers(a, b):"
]

for p in prompts:
    test_speech(p, max_tokens=70, temp=0.8, top_p=0.92, rep_penalty=1.3)
