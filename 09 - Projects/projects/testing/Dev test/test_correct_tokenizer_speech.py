import os
import sys
import torch
import torch.nn.functional as F
from pathlib import Path
from tokenizers import Tokenizer

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

REPO_ROOT = Path(r"C:\02_QUILLAN")
sys.path.insert(0, str(REPO_ROOT))

from _dev.quillan_v8_saturated import QuillanRoninSovereign, QuillanArchConfig

tok_path = r"C:\02_QUILLAN\quillan_bpe_tokenizer_hf\tokenizer.json"
tok = Tokenizer.from_file(tok_path)

print("==================================================================")
print("   👑 QUILLAN-RONIN v5.3.1 - MATCHING TOKENIZER SPEECH TEST")
print("==================================================================")

cfg = QuillanArchConfig(hidden_dim=1024, ffn_dim=2048, num_experts=34, text_only=True, eggroll_rank=256)
model = QuillanRoninSovereign(cfg).to("cpu")

ckpt_path = r"C:\02_QUILLAN\checkpoints\checkpoints_sft\quillan_final_best.pt"
if not os.path.exists(ckpt_path):
    ckpt_path = r"C:\02_QUILLAN\checkpoints\checkpoints_sft\quillan_final_latest.pt"

print(f"[*] Loading state from: {ckpt_path}")
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
sd = ckpt.get("model_state_dict", ckpt)
msd = model.state_dict()
for k, v in sd.items():
    if k in msd and v.shape == msd[k].shape:
        msd[k].copy_(v)
model.load_state_dict(msd)
model.eval()

def generate_speech(prompt, max_tokens=100, temp=0.7, top_p=0.9, rep_penalty=1.2):
    encoded_ids = tok.encode(prompt).ids
    tokens = list(encoded_ids)
    
    print(f"\nPROMPT: {prompt.strip()}")
    print("RESPONSE: ", end="", flush=True)
    
    with torch.no_grad():
        for _ in range(max_tokens):
            inp = torch.tensor([tokens[-256:]], dtype=torch.long)
            logits = model(inp)[:, -1, :].clone()
            
            # Repetition penalty
            if rep_penalty > 1.0:
                for tid in set(tokens[-40:]):
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

            tokens.append(next_tok)
            word = tok.decode([next_tok])
            print(word, end="", flush=True)
            if next_tok in [0, 1, 2] or "user:" in word or "human:" in word:
                break
    print("\n" + "-" * 50)

test_prompts = [
    "system: You are Quillan, a sovereign AI assistant.\nuser: Hello! Who are you?\nassistant: ",
    "user: What can you help me with?\nassistant: ",
    "system: You are a science tutor.\nuser: Explain quantum physics in simple terms.\nassistant: ",
    "user: Write a Python function to add two numbers.\nassistant: "
]

for p in test_prompts:
    generate_speech(p, max_tokens=80, temp=0.6, top_p=0.9, rep_penalty=1.2)
