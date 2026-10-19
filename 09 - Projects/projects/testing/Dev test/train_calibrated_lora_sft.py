#!/usr/bin/env python3
"""
👑 QUILLAN-RONIN v5.3.1 — CALIBRATED LOW-LR SOVEREIGN SFT ENGINE
Trains the 12-layer unrolled 34-expert sovereign model on the multi-format gold dataset
using ultra-gentle AdamW(2e-5) with weight decay(0.05) to preserve 100% baseline fluency
while learning domain-specific ground truths.
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
print("   👑 CALIBRATED LOW-LR SOVEREIGN SFT ENGINE", flush=True)
print("   [LR=2e-5 | Weight Decay=0.05 | 34 Experts | 12 Layers]", flush=True)
print("==================================================================\n", flush=True)

enc = tiktoken.get_encoding("gpt2")

# 1. Re-initialize Pure Zero-Delta Baseline Weights
cfg = QuillanUnrolledConfig()
model = QuillanUnrolledSovereign(cfg).to("cpu")

from transformers import GPT2LMHeadModel
print("[*] Grounding foundational backbone weights...", flush=True)
gpt2 = GPT2LMHeadModel.from_pretrained("gpt2-medium")
sd = gpt2.state_dict()

model.wte.weight.data.copy_(sd['transformer.wte.weight'])
model.wpe.weight.data.copy_(sd['transformer.wpe.weight'])
model.ln_f.weight.data.copy_(sd['transformer.ln_f.weight'])
model.ln_f.bias.data.copy_(sd['transformer.ln_f.bias'])

for i in range(12):
    pfx = f'transformer.h.{i}.'
    b = model.h[i]
    b.ln_1.weight.data.copy_(sd[pfx + 'ln_1.weight'])
    b.ln_1.bias.data.copy_(sd[pfx + 'ln_1.bias'])
    b.attn.c_attn.weight.data.copy_(sd[pfx + 'attn.c_attn.weight'])
    b.attn.c_attn.bias.data.copy_(sd[pfx + 'attn.c_attn.bias'])
    b.attn.c_proj.weight.data.copy_(sd[pfx + 'attn.c_proj.weight'])
    b.attn.c_proj.bias.data.copy_(sd[pfx + 'attn.c_proj.bias'])
    b.ln_2.weight.data.copy_(sd[pfx + 'ln_2.weight'])
    b.ln_2.bias.data.copy_(sd[pfx + 'ln_2.bias'])
    b.moe.c_fc.weight.data.copy_(sd[pfx + 'mlp.c_fc.weight'])
    b.moe.c_fc.bias.data.copy_(sd[pfx + 'mlp.c_fc.bias'])
    b.moe.c_proj.weight.data.copy_(sd[pfx + 'mlp.c_proj.weight'])
    b.moe.c_proj.bias.data.copy_(sd[pfx + 'mlp.c_proj.bias'])

print("[+] Grounded foundational weights successfully!\n", flush=True)

# 2. Load Dataset
pt_file = DATA_DIR / "omniformat_gold_dataset.pt"
print(f"[*] Loading target-masked dataset: {pt_file.name}...", flush=True)
ds = torch.load(pt_file, map_location="cpu", weights_only=False)
input_ids_tensor = ds["input_ids"]
labels_tensor = ds["labels"]
num_samples = ds["num_samples"]
MAX_SEQ_LEN = ds["max_seq_len"]
print(f"[+] Loaded {num_samples:,} target-masked samples!\n", flush=True)

# 3. Isolate Trainable Adapter Parameters (Freeze dense backbone)
adapter_params = []
for name, param in model.named_parameters():
    if any(k in name for k in ['expert_A', 'expert_B', 'swarm', 'prism', 'q1_bridge', 'q2_bridge', 'ingest_gate', 'router', 'moe_gate']):
        param.requires_grad = True
        adapter_params.append(param)
    else:
        param.requires_grad = False

print(f"[+] Isolated {len(adapter_params)} trainable adapter parameters across 34 experts and swarms!\n", flush=True)

STEPS = 120
BASE_LR = 2.0e-5
MIN_LR = 1.0e-6

optimizer = torch.optim.AdamW(adapter_params, lr=BASE_LR, weight_decay=0.05)

def eval_milestone(step_num):
    model.eval()
    test_queries = [
        ("Photosynthesis", "Photosynthesis is the biological process where green plants convert sunlight,"),
        ("Python Palindrome", "def is_palindrome(s):\n    \"\"\"Check if string s is a palindrome.\"\"\"\n    return"),
        ("Linux Signals", "In Linux, the difference between SIGTERM and SIGKILL is that SIGTERM")
    ]
    print(f"\n{'='*65}", flush=True)
    print(f"  [CALIBRATED SFT MILESTONE EVALUATION @ STEP {step_num}]", flush=True)
    print(f"{'='*65}", flush=True)
    for cat, prompt in test_queries:
        toks = enc.encode(prompt)
        t0 = time.time()
        out = model.generate(toks, max_tokens=40, temp=0.7, top_k=50, top_p=0.90, repetition_penalty=1.05)
        ans = enc.decode(out).strip()
        print(f"  PROMPT [{cat}]:\n{prompt}", flush=True)
        print(f"  GENERATED ({time.time()-t0:.2f}s):\n{ans}", flush=True)
        print(f"{'-'*65}", flush=True)
    print(f"{'='*65}\n", flush=True)
    model.train()

print(f"[TRAIN] Launching Calibrated SFT ({STEPS} steps, LR={BASE_LR} -> {MIN_LR})...\n", flush=True)

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

    torch.nn.utils.clip_grad_norm_(adapter_params, 0.5)
    optimizer.step()

    val = loss.item()
    if val < best_loss: best_loss = val

    if step % 10 == 0:
        gc.collect()

    if step % 20 == 0 or step == 1:
        elapsed = time.time() - t_train
        sps = elapsed / step
        eta_m = sps * (STEPS - step) / 60.0
        print(f"  step {step:3d}/{STEPS}  target_loss={val:.4f}  best={best_loss:.4f}  lr={lr:.7f}  ({sps:.2f}s/st, ETA {eta_m:.1f}m)", flush=True)

    if step % 40 == 0:
        eval_milestone(step)
        torch.save({
            'model_state_dict': model.state_dict(),
            'cfg': cfg,
            'step': step,
            'loss': best_loss,
            'version': 'quillan-v5.3.1-unrolled-calibrated-master'
        }, ckpt_path)
        print(f"  [CHECKPOINT] Auto-saved calibrated master checkpoint at step {step}.\n", flush=True)

# Final Save
torch.save({
    'model_state_dict': model.state_dict(),
    'cfg': cfg,
    'step': STEPS,
    'loss': best_loss,
    'version': 'quillan-v5.3.1-unrolled-calibrated-master-final'
}, ckpt_path)

print(f"\n[DONE] 🏆 Calibrated SFT Complete! Best Target Loss: {best_loss:.4f} in {(time.time()-t_train)/60:.1f}m\n", flush=True)

# 10-Question Benchmark
print("==================================================================", flush=True)
print("   👑 10-QUESTION MASTER BENCHMARK SUITE", flush=True)
print("==================================================================\n", flush=True)

model.eval()

BENCHMARK_QUESTIONS = [
    ("Identity", "User: Hello! What are your primary capabilities as an AI?\nAssistant: I am Quillan, an AI assistant capable of"),
    ("Python Palindrome", "Question: Write a Python function to check if a string is a palindrome.\nAnswer:\ndef is_palindrome(s):\n    \"\"\"Check if s is palindrome.\"\"\"\n    return"),
    ("Science (Photosynthesis)", "Question: What is photosynthesis?\nAnswer:\nPhotosynthesis is the biological process where green plants convert sunlight, carbon dioxide, and water into"),
    ("Linux DevOps", "Question: What is the difference between SIGTERM (15) and SIGKILL (9) in Linux?\nAnswer:\nIn Linux, the key difference between SIGTERM and SIGKILL is that SIGTERM"),
    ("Logic / Deduction", "Question: If all roses are flowers and some flowers fade quickly, do all roses fade quickly?\nAnswer:\nLogical Analysis: Not necessarily, because only some flowers fade quickly, meaning"),
    ("Hardware Acceleration", "Question: How does BitNet 1.58-bit ternary quantization reduce memory bandwidth?\nAnswer:\nBitNet ternary quantization replaces 16-bit floating point matrix multiplications with"),
    ("Database / LanceDB", "Question: How does vector indexing in LanceDB achieve sub-millisecond retrieval?\nAnswer:\nLanceDB utilizes IVF-PQ vector indexing to search high-dimensional embeddings by"),
    ("API Design", "Question: What are the core architectural constraints of RESTful APIs?\nAnswer:\nRESTful API architecture enforces stateless communication, client-server separation, and"),
    ("Security & CWE", "Question: How do parameterized queries prevent SQL injection vulnerabilities?\nAnswer:\nParameterized queries separate user data from executable SQL commands so that"),
    ("Creative Synthesis", "Question: Synthesize the relationship between entropy in physics and information theory.\nAnswer:\nBoth thermodynamic entropy and Shannon information entropy quantify the amount of uncertainty and disorder in")
]

for idx, (cat, prompt) in enumerate(BENCHMARK_QUESTIONS, 1):
    toks = enc.encode(prompt)
    t_start = time.time()
    out = model.generate(toks, max_tokens=50, temp=0.7, top_k=50, top_p=0.90, repetition_penalty=1.05)
    elapsed = time.time() - t_start
    gen_text = enc.decode(out).strip()
    
    print(f"[{idx}/10] CATEGORY: {cat}", flush=True)
    print(f"PROMPT:\n{prompt}", flush=True)
    print(f"GENERATED ({elapsed:.2f}s, {len(out)/max(0.001, elapsed):.1f} tok/s):\n{gen_text}", flush=True)
    print(f"{'-'*65}\n", flush=True)

print("==================================================================", flush=True)
print("   🏆 MASTER BENCHMARK EVALUATION 100% COMPLETE", flush=True)
print("==================================================================", flush=True)
