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
print("   👑 QUILLAN-RONIN v5.3.1 - HIGH-SPEED INFERENCE & SPEECH VERIFICATION")
print("==================================================================")

enc = tiktoken.get_encoding("gpt2")
cfg = QuillanArchConfig(hidden_dim=1024, ffn_dim=2048, num_experts=34, text_only=True, eggroll_rank=256)
model = QuillanRoninSovereign(cfg).to("cpu")

possible_ckpts = [
    r"C:\02_QUILLAN\checkpoints\checkpoints_sft\quillan_final_latest.pt",
    r"C:\02_QUILLAN\checkpoints\checkpoints_sft\quillan_final_best.pt",
    r"C:\02_QUILLAN\05_Training\quillan_final_latest.pt"
]

loaded_path = None
for cp in possible_ckpts:
    if os.path.exists(cp):
        print(f"[*] Loading checkpoint: {cp}")
        try:
            ckpt = torch.load(cp, map_location="cpu", weights_only=False)
            sd = ckpt.get("model_state_dict", ckpt)
            model_sd = model.state_dict()
            copied = 0
            for k, v in sd.items():
                if k in model_sd and v.shape == model_sd[k].shape:
                    model_sd[k].copy_(v)
                    copied += 1
            model.load_state_dict(model_sd)
            step_val = ckpt.get('step', 'N/A') if isinstance(ckpt, dict) else 'N/A'
            loss_val = ckpt.get('loss', 'N/A') if isinstance(ckpt, dict) else 'N/A'
            print(f"[+] Loaded {copied}/{len(model_sd)} parameter layers (Step: {step_val}, Loss: {loss_val})")
            loaded_path = cp
            break
        except Exception as e:
            print(f"[!] Error loading {cp}: {e}")

model.eval()

def generate(prompt, max_tokens=100, temp=0.7, top_p=0.9, rep_penalty=1.2):
    tokens = enc.encode(prompt)
    generated = list(tokens)
    eos_id = 50256
    
    print(f"\nPROMPT: {prompt.strip()}")
    print("RESPONSE: ", end="", flush=True)
    
    with torch.no_grad():
        for step_i in range(max_tokens):
            inp = torch.tensor([generated[-256:]], dtype=torch.long)
            logits = model(inp)[:, -1, :].clone()
            
            # Repetition penalty
            if rep_penalty > 1.0:
                for token_id in set(generated[-64:]):
                    if logits[0, token_id] < 0:
                        logits[0, token_id] *= rep_penalty
                    else:
                        logits[0, token_id] /= rep_penalty

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
            decoded = enc.decode([next_tok])
            print(decoded, end="", flush=True)
            if next_tok == eos_id or "<|end|>" in decoded:
                break
    print("\n" + "-" * 50)

test_prompts = [
    "<|user|>\nHello! Who are you?\n<|assistant|>\n",
    "<|user|>\nWhat can you help me with?\n<|assistant|>\n",
    "<|user|>\nExplain quantum physics in simple terms.\n<|assistant|>\n",
    "<|user|>\nWrite a python function to calculate factorial.\n<|assistant|>\n"
]

for p in test_prompts:
    generate(p, max_tokens=70, temp=0.6, top_p=0.88, rep_penalty=1.2)
