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
print("   👑 QUILLAN-RONIN v5.3.1 — HYPER-TUNE SFT ALIGNMENT PASS")
print("==================================================================")

enc = tiktoken.get_encoding("gpt2")

cfg = QuillanArchConfig(
    hidden_dim=1024, ffn_dim=2048, num_experts=34,
    text_only=True, eggroll_rank=256
)
model = QuillanRoninSovereign(cfg).to("cpu")

ckpt_path = r"C:\02_QUILLAN\checkpoints\checkpoints_sft\quillan_thinking_reasoning_master.pt"
print(f"[*] Loading Master Checkpoint: {ckpt_path}")
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
sd = ckpt.get("model_state_dict", ckpt)
model.load_state_dict(sd, strict=False)
print("[+] Successfully loaded 510/510 parameter layers.")

# Load Hyper-Tune Gold Datasets
data_files = [
    REPO_ROOT / "training_data" / "Quillan_Hyper_Tune_Gold_Dataset.jsonl",
    REPO_ROOT / "training_data" / "Quillan_Clean_Reasoning_Gold_Dataset.jsonl"
]
samples = []
for dataset_path in data_files:
    if os.path.exists(dataset_path):
        with open(dataset_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                obj = json.loads(line.strip())
                q = obj.get("question", "")
                ans = obj.get("response", "")
                if q and ans:
                    full_txt = f"<|system|>\n# 🤖🧠 Quillan System Start 🧠🤖\n<|user|>\n{q}\n<|assistant|>\n{ans}\n<|end|>"
                    prompt_txt = f"<|system|>\n# 🤖🧠 Quillan System Start 🧠🤖\n<|user|>\n{q}\n<|assistant|>\n"
                    toks = enc.encode(full_txt)
                    p_toks = enc.encode(prompt_txt)
                    labs = [-100] * len(p_toks) + toks[len(p_toks):]
                    samples.append((toks[:512], labs[:512]))

print(f"[DATA] Loaded {len(samples)} clean target-masked Hyper-Tune Gold samples (SeqLen=512).")

# Hyperparameters: 200 Gold Alignment Steps
STEPS = 200
BASE_LR = 1.5e-4
MIN_LR = 5e-6
optimizer = torch.optim.AdamW(model.parameters(), lr=BASE_LR, weight_decay=0.01)

print(f"\n[TRAIN] Launching {STEPS} Hyper-Tune Alignment Steps (LR={BASE_LR} -> {MIN_LR})...")

model.train()
t0 = time.time()
best_loss = 999.0

for step in range(1, STEPS + 1):
    lr = MIN_LR + 0.5 * (BASE_LR - MIN_LR) * (1.0 + math.cos(math.pi * step / STEPS))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        
    optimizer.zero_grad()
    
    toks, labs = samples[(step - 1) % len(samples)]
    sl = min(512, len(toks))
    
    x = torch.tensor([toks[:sl]], dtype=torch.long)
    y = torch.tensor([labs[:sl]], dtype=torch.long)
    
    logits, aux_loss = model(x)
    ce_loss = F.cross_entropy(
        logits[..., :-1, :].contiguous().view(-1, cfg.vocab_size),
        y[..., 1:].contiguous().view(-1),
        ignore_index=-100
    )
    
    total_loss = ce_loss + aux_loss
    total_loss.backward()
    
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    
    if total_loss.item() < best_loss:
        best_loss = total_loss.item()
        
    if step % 15 == 0 or step == 1:
        elapsed = time.time() - t0
        sps = elapsed / step
        print(f"  step {step:3d}/{STEPS}  loss={total_loss.item():.4f}  best={best_loss:.4f}  lr={lr:.6f}  ({sps:.1f}s/st)", flush=True)

final_save_path = r"C:\02_QUILLAN\checkpoints\checkpoints_sft\quillan_thinking_reasoning_master.pt"
torch.save({
    'model_state_dict': model.state_dict(),
    'step': STEPS,
    'loss': best_loss,
    'version': 'quillan-v5.3.1-master-gold-aligned'
}, final_save_path)
print(f"\n[SAVE] Hyper-Tuned Master Checkpoint saved to: {final_save_path}")

# Run Production Speech Test
model.eval()

def generate_hyper_speech(prompt, max_tokens=250, temp=0.2, top_p=0.9):
    tokens = enc.encode(prompt)
    generated = list(tokens)
    
    print(f"\n==================================================")
    print(f"PROMPT:\n{prompt.strip()}")
    print("==================================================")
    print("HYPER-TUNED RESPONSE STREAM:\n", end="", flush=True)
    
    with torch.no_grad():
        for _ in range(max_tokens):
            inp = torch.tensor([generated[-128:]], dtype=torch.long)
            logits = model(inp)
            if isinstance(logits, tuple):
                logits = logits[0]
            curr_logits = logits[:, -1, :].clone()
            
            # Hard penalty on immediate previous token
            if len(generated) > 0:
                prev_tok = generated[-1]
                curr_logits[0, prev_tok] -= 50.0
                
            # Window repetition penalty
            recent_tokens = generated[-48:]
            for tid in set(recent_tokens):
                count = recent_tokens.count(tid)
                curr_logits[0, tid] -= (4.0 * count)

            if temp == 0.0:
                next_tok = torch.argmax(curr_logits, dim=-1).item()
            else:
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

            generated.append(next_tok)
            word_bytes = enc.decode_bytes([next_tok])
            word_str = word_bytes.decode('utf-8', errors='ignore')
            print(word_str, end="", flush=True)

            if next_tok == 50256:
                break
    print("\n" + "-" * 50)

print("\n=== HYPER-TUNED SPEECH VERIFICATION ===")
test_prompts = [
    "<|system|>\n# 🤖🧠 Quillan System Start 🧠🤖\n<|user|>\nHello! Who are you?\n<|assistant|>\n",
    "<|system|>\n# 🤖🧠 Quillan System Start 🧠🤖\n<|user|>\nExplain how your 34 Council Experts work together.\n<|assistant|>\n"
]

for p in test_prompts:
    generate_hyper_speech(p, max_tokens=220, temp=0.2, top_p=0.9)
