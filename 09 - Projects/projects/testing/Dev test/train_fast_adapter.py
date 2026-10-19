#!/usr/bin/env python3
"""
👑 QUILLAN-RONIN v5.3.1 — ULTRA-FAST FOCUSED ADAPTER TRAINER
Freezes massive 34-expert base matrices, trains LoRA adapters + Norms + Decoders.
Runs at ~2-3s per step on CPU to rapidly drive loss from 6.6 down to < 2.0.
"""
import os, sys, time, math, json, random, gc, torch
import torch.nn.functional as F
import tiktoken
from pathlib import Path

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

REPO_ROOT = Path(r"C:\02_QUILLAN")
sys.path.insert(0, str(REPO_ROOT))

from _dev.quillan_v8_saturated import QuillanRoninSovereign, QuillanArchConfig

enc = tiktoken.get_encoding("gpt2")

print("==================================================================")
print("   👑 QUILLAN-RONIN v5.3.1 — HIGH-SPEED ADAPTER ALIGNMENT")
print("==================================================================")

cfg = QuillanArchConfig(
    hidden_dim=1024, ffn_dim=2048, num_experts=34,
    vocab_size=50257, text_only=True, eggroll_rank=256
)
model = QuillanRoninSovereign(cfg).to("cpu")

ckpt_path = REPO_ROOT / "checkpoints" / "checkpoints_sft" / "quillan_thinking_reasoning_master.pt"
print(f"[*] Loading checkpoint: {ckpt_path.name}")
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
sd = ckpt.get("model_state_dict", ckpt)
model.load_state_dict(sd, strict=False)
print(f"[+] Base loaded successfully.")

# ─── FREEZE BASE & UNFREEZE ADAPTERS + NORMS + INGESTION ─────────────────────
trainable_params = []
frozen_count = 0
active_count = 0

for name, param in model.named_parameters():
    # Train LoRA adapters, LayerNorms, Ingestion bridge, and output heads
    if any(k in name for k in ['lora_', 'norm', 'final_norm', 'pre_final_norm', 'ingest', 'txt_dec', 'quillan_gate']):
        param.requires_grad = True
        trainable_params.append(param)
        active_count += param.numel()
    else:
        param.requires_grad = False
        frozen_count += param.numel()

print(f"[PARAMS] Frozen Base Weights: {frozen_count / 1e6:.1f}M")
print(f"[PARAMS] Active Trainable Weights: {active_count / 1e6:.1f}M (Fast CPU Convergence)")

# ─── LOAD DATASET ─────────────────────────────────────────────────────────────
SEQ_LEN = 192
data_dir = REPO_ROOT / "training_data"

dataset_configs = [
    ("Quillan_Clean_Reasoning_Gold_Dataset.jsonl", 4),
    ("Quillan_Hyper_Tune_Gold_Dataset.jsonl", 6),
    ("Quillan_Explanatory_Prose_Dataset.jsonl", 6),
    ("Quillan_General_Knowledge_Dataset.jsonl", 6),
    ("code_train.jsonl", 1),
    ("instruct_train.jsonl", 1),
]

training_samples = []
for fname, repeat in dataset_configs:
    fpath = data_dir / fname
    if not fpath.exists(): continue
    count = 0
    with open(fpath, "r", encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if not line_str: continue
            try:
                obj = json.loads(line_str)
            except Exception:
                continue
            q = (obj.get("question") or obj.get("instruction") or obj.get("prompt") or "")
            ans = (obj.get("response") or obj.get("answer") or obj.get("output") or "")
            if not q and not ans and "messages" in obj:
                for m in obj["messages"]:
                    if m.get("role") == "user": q = m.get("content", "")
                    elif m.get("role") == "assistant": ans = m.get("content", "").split("</think>")[-1].strip()
            if not q or not ans: continue
            
            full_txt = f"<|system|>\n# 🤖🧠 Quillan System Start 🧠🤖\n<|user|>\n{q}\n<|assistant|>\n{ans}\n<|end|>"
            prompt_txt = f"<|system|>\n# 🤖🧠 Quillan System Start 🧠🤖\n<|user|>\n{q}\n<|assistant|>\n"
            toks = enc.encode(full_txt)[:SEQ_LEN]
            p_toks = enc.encode(prompt_txt)
            labs = [-100] * len(p_toks) + toks[len(p_toks):]
            labs = labs[:SEQ_LEN]
            if len(toks) < 12: continue
            for _ in range(repeat):
                training_samples.append((toks, labs))
            count += 1

random.seed(42)
random.shuffle(training_samples)
print(f"[DATA] Loaded {len(training_samples)} High-Density Samples (seq_len={SEQ_LEN})")

# ─── TRAINING LOOP ────────────────────────────────────────────────────────────
STEPS = 600
BASE_LR = 8e-5
MIN_LR = 5e-6
optimizer = torch.optim.AdamW(trainable_params, lr=BASE_LR, weight_decay=0.01)

def eval_sentence():
    model.eval()
    q = "What is 2 + 2?"
    p = f"<|system|>\n# 🤖🧠 Quillan System Start 🧠🤖\n<|user|>\n{q}\n<|assistant|>\n"
    toks = enc.encode(p)
    gen = list(toks)
    with torch.no_grad():
        for _ in range(50):
            inp = torch.tensor([gen[-128:]], dtype=torch.long)
            logits = model(inp)
            if isinstance(logits, tuple): logits = logits[0]
            curr = logits[:, -1, :].clone()
            if len(gen) > 0: curr[0, gen[-1]] -= 50.0
            next_tok = torch.argmax(curr, dim=-1).item()
            gen.append(next_tok)
            if next_tok == 50256: break
    out = enc.decode(gen[len(toks):])
    print(f"\n  [LIVE EVAL] Q: '{q}' -> Response: {out.strip()[:180]}\n", flush=True)
    model.train()

print(f"\n[TRAIN] Launching Fast Adapter SFT ({STEPS} steps, LR={BASE_LR} -> {MIN_LR})...\n")
model.train()
t0 = time.time()
best_loss = 999.0
idx = 0

for step in range(1, STEPS + 1):
    lr = MIN_LR + 0.5 * (BASE_LR - MIN_LR) * (1.0 + math.cos(math.pi * step / STEPS))
    for pg in optimizer.param_groups:
        pg['lr'] = lr
        
    optimizer.zero_grad()
    toks, labs = training_samples[idx % len(training_samples)]
    idx += 1
    if idx % len(training_samples) == 0:
        random.shuffle(training_samples)
        
    x = torch.tensor([toks], dtype=torch.long)
    y = torch.tensor([labs], dtype=torch.long)
    
    logits, aux = model(x)
    loss = F.cross_entropy(
        logits[..., :-1, :].contiguous().view(-1, cfg.vocab_size),
        y[..., 1:].contiguous().view(-1),
        ignore_index=-100
    )
    total_loss = loss + 0.002 * aux
    total_loss.backward()
    
    torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
    optimizer.step()
    
    val = total_loss.item()
    if val < best_loss:
        best_loss = val
        
    if step % 20 == 0 or step == 1:
        elapsed = time.time() - t0
        sps = elapsed / step
        eta_m = sps * (STEPS - step) / 60.0
        print(f"  step {step:3d}/{STEPS}  loss={val:.4f}  best={best_loss:.4f}  lr={lr:.6f}  ({sps:.1f}s/st, ETA {eta_m:.1f}m)", flush=True)
        
    if step % 100 == 0:
        eval_sentence()
        torch.save({
            'model_state_dict': model.state_dict(),
            'step': step, 'loss': best_loss,
            'version': 'quillan-v5.3.1-adapter-sft'
        }, ckpt_path)

# Final Save
torch.save({
    'model_state_dict': model.state_dict(),
    'step': STEPS, 'loss': best_loss,
    'version': 'quillan-v5.3.1-master-adapter'
}, ckpt_path)

print(f"\n[DONE] 🏆 Fast Training Complete! Final Best Loss: {best_loss:.4f} in {(time.time()-t0)/60:.1f}m")
eval_sentence()
