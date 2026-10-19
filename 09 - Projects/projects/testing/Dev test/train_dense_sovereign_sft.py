#!/usr/bin/env python3
"""
👑 QUILLAN-RONIN v5.3.1 — 12-LAYER DENSE SOVEREIGN INTACT PAIR SFT
Fine-tunes the 12-Layer Internally Dense Sovereign Transformer (1024-dim, 16 heads)
with 34 Council MoE adapters on 11,009 intact (Prompt, <think>, <output>) pairs.
Uses natural dialogue tags:
  Prompt: "User: {question}\nAssistant:\n<think>"
  Target: "{reasoning}\n</think>\n<output>\n{answer}\n</output><|endoftext|>"
"""

import os, sys, time, json, math, random, psutil, torch, torch.nn.functional as F, tiktoken
from pathlib import Path

p = psutil.Process()
try:
    p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
except Exception:
    pass

torch.set_num_threads(4)
torch.set_num_interop_threads(2)

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

SCRATCH_DIR = Path(r"C:\Users\Admin\.gemini\antigravity-ide\brain\47d0b748-384a-4cc6-bcb6-f13c52f1d9d9\scratch")
sys.path.insert(0, str(SCRATCH_DIR))

from quillan_v9_dense_sovereign import QuillanDenseSovereign, QuillanDenseConfig

REPO_ROOT = Path(r"C:\02_QUILLAN")
DATA_DIR = REPO_ROOT / "training_data"
ckpt_path = REPO_ROOT / "checkpoints" / "checkpoints_sft" / "quillan_thinking_reasoning_master.pt"

print("==================================================================", flush=True)
print("   👑 QUILLAN-RONIN v5.3.1 — 12-LAYER DENSE SOVEREIGN SFT ENGINE", flush=True)
print("   [12 Dense Layers | 34 Council MoE Adapters | Intact Pairs]", flush=True)
print("==================================================================", flush=True)

enc = tiktoken.get_encoding("gpt2")

pt_file = DATA_DIR / "intact_pair_dataset.pt"
print(f"[*] Loading intact pair dataset: {pt_file.name}...", flush=True)
ds = torch.load(pt_file, map_location="cpu", weights_only=False)
input_ids_tensor = ds["input_ids"]
labels_tensor = ds["labels"]
num_samples = ds["num_samples"]
MAX_SEQ_LEN = ds["max_seq_len"]
print(f"[+] Loaded {num_samples:,} intact pairs (SeqLen={MAX_SEQ_LEN})!\n", flush=True)

# Load 12-layer Dense Model
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
model = QuillanDenseSovereign(ckpt["cfg"]).to("cpu")
model.load_state_dict(ckpt["model_state_dict"])
print("[+] Successfully loaded 12-layer Dense Sovereign Model!\n", flush=True)

# Train adapters and attention projections
STEPS = 500
BASE_LR = 2.0e-5
MIN_LR = 1.0e-6

optimizer = torch.optim.AdamW(model.parameters(), lr=BASE_LR, weight_decay=0.01)

def eval_dialogue(step_num):
    model.eval()
    test_queries = [
        "User: Hello! What are your capabilities?\nAssistant:",
        "User: Write a Python function to check if a string is a palindrome.\nAssistant:\ndef is_palindrome(s):",
        "User: What is the key difference between SIGTERM and SIGKILL in Linux?\nAssistant:"
    ]
    print(f"\n{'='*65}", flush=True)
    print(f"  [12-LAYER DENSE DIALOGUE EVALUATION @ STEP {step_num}]", flush=True)
    print(f"{'='*65}", flush=True)
    for tq in test_queries:
        toks = enc.encode(tq)
        out = model.generate(toks, max_tokens=45, temp=0.65, top_p=0.85, repetition_penalty=1.15)
        ans = enc.decode(out).strip()
        print(f"  PROMPT:\n{tq}", flush=True)
        print(f"  RESPONSE:\n{ans}", flush=True)
        print(f"{'-'*65}", flush=True)
    print(f"{'='*65}\n", flush=True)
    model.train()

print(f"[TRAIN] Launching 12-Layer Dense SFT ({STEPS} steps, LR={BASE_LR} -> {MIN_LR})...\n", flush=True)

model.train()
t_train = time.time()
best_loss = 999.0

for step in range(1, STEPS + 1):
    lr = MIN_LR + 0.5 * (BASE_LR - MIN_LR) * (1.0 + math.cos(math.pi * step / STEPS))
    for pg in optimizer.param_groups:
        pg['lr'] = lr

    idx = random.randint(0, num_samples - 1)
    x_batch = input_ids_tensor[idx].unsqueeze(0)
    y_batch = labels_tensor[idx].unsqueeze(0)

    optimizer.zero_grad()
    logits = model(x_batch)
    loss = F.cross_entropy(logits.view(-1, 50257), y_batch.view(-1), ignore_index=-100)
    loss.backward()

    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    val = loss.item()
    if val < best_loss: best_loss = val

    if step % 25 == 0 or step == 1:
        elapsed = time.time() - t_train
        sps = elapsed / step
        eta_m = sps * (STEPS - step) / 60.0
        print(f"  step {step:3d}/{STEPS}  target_loss={val:.4f}  best={best_loss:.4f}  lr={lr:.7f}  ({sps:.1f}s/st, ETA {eta_m:.1f}m)", flush=True)

    if step % 100 == 0:
        eval_dialogue(step)
        torch.save({
            'model_state_dict': model.state_dict(),
            'cfg': ckpt["cfg"],
            'step': step,
            'loss': best_loss,
            'version': 'quillan-v5.3.1-dense-12layer-master'
        }, ckpt_path)
        print(f"  [CHECKPOINT] Auto-saved 12-layer master model at step {step}.\n", flush=True)

# Final Save
torch.save({
    'model_state_dict': model.state_dict(),
    'cfg': ckpt["cfg"],
    'step': STEPS,
    'loss': best_loss,
    'version': 'quillan-v5.3.1-dense-12layer-master-final'
}, ckpt_path)

print(f"\n[DONE] 🏆 12-Layer Dense Training Complete! Best Target Loss: {best_loss:.4f} in {(time.time()-t_train)/60:.1f}m\n", flush=True)
eval_dialogue(STEPS)
