import torch
import torch.nn.functional as F
import tiktoken
import os
import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

REPO_ROOT = r"C:\02_QUILLAN"
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


from _dev.quillan_v8_saturated import QuillanRoninSovereign, QuillanArchConfig

enc = tiktoken.get_encoding("gpt2")
cfg = QuillanArchConfig(hidden_dim=1024, ffn_dim=2048, num_experts=34, text_only=True, eggroll_rank=256)
model = QuillanRoninSovereign(cfg).to("cpu")


ckpt_path = r"C:\02_QUILLAN\checkpoints\checkpoints_sft\quillan_sft_v3_best.pt"
if os.path.exists(ckpt_path):
    print(f"[*] Loading clean trained SFT best checkpoint from {ckpt_path}...")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt.get("model_state_dict", ckpt)
    model_sd = model.state_dict()
    copied = 0
    for k, v in sd.items():
        if k in model_sd and v.shape == model_sd[k].shape:
            model_sd[k].copy_(v)
            copied += 1
    model.load_state_dict(model_sd)
    print(f"[+] Loaded {copied} layer weights successfully!")
else:
    print("[!] No checkpoint found, using initial model weights")



model.eval()

def generate_text(prompt, max_tokens=60, temp=0.7, top_p=0.9):
    tokens = enc.encode(prompt)
    print(f"\n--- PROMPT ---\n{prompt}\n--- GENERATED RESPONSE ---")
    
    with torch.no_grad():
        for _ in range(max_tokens):
            inp = torch.tensor([tokens[-512:]], dtype=torch.long)
            logits = model(inp)[:, -1, :]
            
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
                
            tokens.append(next_tok)
            decoded = enc.decode([next_tok])
            print(decoded, end="", flush=True)
            if next_tok == 50256: # EOS
                break
    print("\n--------------------------")

generate_text("<|user|>\nHello! Who are you?\n<|assistant|>\n")
generate_text("<|user|>\nExplain quantum physics in one sentence.\n<|assistant|>\n")
