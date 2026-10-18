import os
import sys
import time
import math
import json
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
print("   👑 QUILLAN-RONIN v5.3.1 — DIALOGUE SFT ALIGNMENT PASS")
print("==================================================================")

enc = tiktoken.get_encoding("gpt2")

cfg = QuillanArchConfig(
    hidden_dim=1024, ffn_dim=2048, num_experts=34,
    text_only=True, eggroll_rank=256
)
model = QuillanRoninSovereign(cfg).to("cpu")

ckpt_path = r"C:\02_QUILLAN\checkpoints\checkpoints_sft\quillan_final_best.pt"
if not os.path.exists(ckpt_path):
    ckpt_path = r"C:\02_QUILLAN\checkpoints\checkpoints_sft\quillan_final_latest.pt"

print(f"[*] Loading base checkpoint: {ckpt_path}")
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

# Load Dialogue Datasets
dialogue_samples = []

# 1. Samurai Seed Dataset
seed_path = r"C:\02_QUILLAN\training_data\Quillan_Ronin_v5.3.1_Samurai_Training_Seed_Dataset.jsonl"
if os.path.exists(seed_path):
    with open(seed_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            obj = json.loads(line.strip())
            q = obj.get("question", "")
            ans = obj.get("final_output", "")
            if q and ans:
                full_txt = f"<|user|>\n{q}\n<|assistant|>\n{ans}\n<|end|>"
                prompt_txt = f"<|user|>\n{q}\n<|assistant|>\n"
                
                toks = enc.encode(full_txt)
                p_toks = enc.encode(prompt_txt)
                
                # Target masking: -100 for prompt tokens
                labs = [-100] * len(p_toks) + toks[len(p_toks):]
                if len(toks) > 32:
                    dialogue_samples.append((toks[:512], labs[:512]))

# 2. GPT 5.5 Distilled Dataset
distilled_path = r"C:\02_QUILLAN\training_data\GPT_5.5_Distilled.jsonl"
if os.path.exists(distilled_path):
    with open(distilled_path, "r", encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f):
            if i >= 500: break # Grab top 500 clean dialogue pairs
            if not line.strip(): continue
            try:
                obj = json.loads(line.strip())
                if "messages" in obj:
                    msgs = obj["messages"]
                    full_txt = ""
                    for m in msgs:
                        role = m.get("role", "user")
                        content = m.get("content", "")
                        full_txt += f"<|{role}|>\n{content}\n"
                    full_txt += "<|end|>"
                    toks = enc.encode(full_txt)
                    if len(toks) > 32:
                        dialogue_samples.append((toks[:512], toks[:512]))
                elif "prompt" in obj and "response" in obj:
                    full_txt = f"<|user|>\n{obj['prompt']}\n<|assistant|>\n{obj['response']}\n<|end|>"
                    toks = enc.encode(full_txt)
                    if len(toks) > 32:
                        dialogue_samples.append((toks[:512], toks[:512]))
            except Exception:
                pass

print(f"[DATA] Prepared {len(dialogue_samples)} clean target-masked dialogue samples.")

# Fine-Tuning Hyperparameters
ALIGN_STEPS = 150
LR = 5e-6
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)

print(f"\n[ALIGNMENT] Running {ALIGN_STEPS} Dialogue Target-Masked SFT Steps (LR={LR})...")

model.train()
t0 = time.time()
best_align_loss = 999.0

for step in range(1, ALIGN_STEPS + 1):
    optimizer.zero_grad()
    
    # Batch size 2
    toks1, labs1 = dialogue_samples[(step * 2 - 2) % len(dialogue_samples)]
    toks2, labs2 = dialogue_samples[(step * 2 - 1) % len(dialogue_samples)]
    
    sl1 = min(512, len(toks1))
    sl2 = min(512, len(toks2))
    
    x1 = torch.tensor([toks1[:sl1]], dtype=torch.long)
    y1 = torch.tensor([labs1[:sl1]], dtype=torch.long)
    x2 = torch.tensor([toks2[:sl2]], dtype=torch.long)
    y2 = torch.tensor([labs2[:sl2]], dtype=torch.long)
    
    logits1, aux1 = model(x1)
    loss1 = F.cross_entropy(logits1[..., :-1, :].contiguous().view(-1, cfg.vocab_size), y1[..., 1:].contiguous().view(-1), ignore_index=-100) + aux1
    
    logits2, aux2 = model(x2)
    loss2 = F.cross_entropy(logits2[..., :-1, :].contiguous().view(-1, cfg.vocab_size), y2[..., 1:].contiguous().view(-1), ignore_index=-100) + aux2
    
    total_loss = (loss1 + loss2) / 2.0
    total_loss.backward()
    
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    
    if total_loss.item() < best_align_loss:
        best_align_loss = total_loss.item()
        
    if step % 25 == 0 or step == 1:
        elapsed = time.time() - t0
        sps = elapsed / step
        print(f"  alignment step {step:3d}/{ALIGN_STEPS}  loss={total_loss.item():.4f}  best={best_align_loss:.4f}  ({sps:.1f}s/st)", flush=True)

# Save Aligned Checkpoint
save_path = r"C:\02_QUILLAN\checkpoints\checkpoints_sft\quillan_fluent_aligned.pt"
torch.save({
    'model_state_dict': model.state_dict(),
    'step': ALIGN_STEPS,
    'loss': best_align_loss,
    'version': 'quillan-v5.3.1-fluent-aligned'
}, save_path)

print(f"\n[SAVE] Aligned checkpoint saved to: {save_path}")

# Test Speech Generation with Aligned Checkpoint
model.eval()

def generate_speech(prompt, max_tokens=80, temp=0.7, top_p=0.9, rep_penalty=1.25):
    tokens = enc.encode(prompt)
    generated = list(tokens)
    
    print(f"\nPROMPT: {prompt.strip()}")
    print("RESPONSE: ", end="", flush=True)
    
    with torch.no_grad():
        for _ in range(max_tokens):
            inp = torch.tensor([generated[-256:]], dtype=torch.long)
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
            word = enc.decode([next_tok])
            print(word, end="", flush=True)
            if next_tok == 50256 or "<|end|>" in word:
                break
    print("\n" + "-" * 50)

print("\n=== FINAL FLUENT ALIGNED SPEECH TEST ===")
test_prompts = [
    "<|user|>\nHello! Who are you?\n<|assistant|>\n",
    "<|user|>\nWhat can you help me with?\n<|assistant|>\n",
    "<|user|>\nExplain quantum entanglement in simple terms.\n<|assistant|>\n",
    "<|user|>\nWrite a Python function to add two numbers.\n<|assistant|>\n"
]

for p in test_prompts:
    generate_speech(p, max_tokens=70, temp=0.7, top_p=0.9, rep_penalty=1.25)
