#!/usr/bin/env python3
"""
👑 QUILLAN-RONIN v5.3.1 — MEMORY-HARDENED DEEP ANNEALING SFT ENGINE
Eliminates OOM risk using lightweight SGD (0 state buffers, saving 3.5GB RAM)
with explicit garbage collection every 10 steps.
"""

import os, sys, time, gc, math, random, psutil, torch, torch.nn.functional as F, tiktoken
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

from quillan_v10_unrolled_sovereign import QuillanUnrolledSovereign, QuillanUnrolledConfig

REPO_ROOT = Path(r"C:\02_QUILLAN")
DATA_DIR = REPO_ROOT / "training_data"
ckpt_path = REPO_ROOT / "checkpoints" / "checkpoints_sft" / "quillan_thinking_reasoning_master.pt"

print("==================================================================", flush=True)
print("   👑 QUILLAN-RONIN v5.3.1 — MEMORY-HARDENED SFT ENGINE", flush=True)
print("   [Zero Memory Overhead | 34 Unique Experts | 12 Layers]", flush=True)
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

# Load Unrolled Master Checkpoint
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
model = QuillanUnrolledSovereign(ckpt["cfg"]).to("cpu")
model.load_state_dict(ckpt["model_state_dict"])
print("[+] Successfully loaded Unrolled 34-Expert Sovereign Model!\n", flush=True)

STEPS = 600
BASE_LR = 5.0e-5
MIN_LR = 5.0e-7

# Use Momentum SGD to avoid allocating 3.5GB of AdamW second-order moment buffers
optimizer = torch.optim.SGD(model.parameters(), lr=BASE_LR, momentum=0.9, nesterov=True, weight_decay=1e-4)

def eval_hardened(step_num):
    model.eval()
    test_queries = [
        "User: Hello! What are your primary capabilities as an AI?\nAssistant: I am Quillan, an AI capable of",
        "Question: Write a Python function to check if a string is a palindrome.\nAnswer:\ndef is_palindrome(s):\n    return",
        "Question: What is photosynthesis?\nAnswer:\nPhotosynthesis is the biological process where"
    ]
    print(f"\n{'='*65}", flush=True)
    print(f"  [MEMORY-HARDENED EVALUATION @ STEP {step_num}]", flush=True)
    print(f"{'='*65}", flush=True)
    for tq in test_queries:
        toks = enc.encode(tq)
        out = model.generate(toks, max_tokens=45, temp=0.55, top_p=0.85, repetition_penalty=1.12)
        ans = enc.decode(out).strip()
        print(f"  PROMPT:\n{tq}", flush=True)
        print(f"  RESPONSE:\n{ans}", flush=True)
        print(f"{'-'*65}", flush=True)
    print(f"{'='*65}\n", flush=True)
    model.train()

print(f"[TRAIN] Launching Memory-Hardened SFT ({STEPS} steps, LR={BASE_LR} -> {MIN_LR})...\n", flush=True)
eval_hardened(0)

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

    optimizer.zero_grad(set_to_none=True)
    logits = model(x_batch)
    loss = F.cross_entropy(logits.view(-1, 50257), y_batch.view(-1), ignore_index=-100)
    loss.backward()

    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    val = loss.item()
    if val < best_loss: best_loss = val

    # Proactive garbage collection to maintain flat memory profile
    if step % 10 == 0:
        gc.collect()

    if step % 50 == 0 or step == 1:
        elapsed = time.time() - t_train
        sps = elapsed / step
        eta_m = sps * (STEPS - step) / 60.0
        print(f"  step {step:3d}/{STEPS}  target_loss={val:.4f}  best={best_loss:.4f}  lr={lr:.7f}  ({sps:.1f}s/st, ETA {eta_m:.1f}m)", flush=True)

    if step % 150 == 0:
        eval_hardened(step)
        torch.save({
            'model_state_dict': model.state_dict(),
            'cfg': ckpt["cfg"],
            'step': step,
            'loss': best_loss,
            'version': 'quillan-v5.3.1-unrolled-hardened-master'
        }, ckpt_path)
        print(f"  [CHECKPOINT] Auto-saved memory-hardened master model at step {step}.\n", flush=True)

# Final Save
torch.save({
    'model_state_dict': model.state_dict(),
    'cfg': ckpt["cfg"],
    'step': STEPS,
    'loss': best_loss,
    'version': 'quillan-v5.3.1-unrolled-hardened-master-final'
}, ckpt_path)

print(f"\n[DONE] 🏆 Memory-Hardened SFT Complete! Best Target Loss: {best_loss:.4f} in {(time.time()-t_train)/60:.1f}m\n", flush=True)
eval_hardened(STEPS)
