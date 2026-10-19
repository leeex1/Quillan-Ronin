#!/usr/bin/env python3
"""
👑 QUILLAN-RONIN v5.3.1 — FAST FLUENCY ALIGNMENT SFT
Optimized for CPU: short seqlen, grad_accum=2, 300 steps
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
print("   👑 QUILLAN-RONIN v5.3.1 — FAST FLUENCY ALIGNMENT SFT")
print("==================================================================")

cfg = QuillanArchConfig(
    hidden_dim=1024, ffn_dim=2048, num_experts=34,
    vocab_size=50257, text_only=True, eggroll_rank=256
)
model = QuillanRoninSovereign(cfg).to("cpu")

# Restore base from hyper_tuned (the copy we made)
ckpt_path = REPO_ROOT / "checkpoints" / "checkpoints_sft" / "quillan_thinking_reasoning_master.pt"
save_path = REPO_ROOT / "checkpoints" / "checkpoints_sft" / "quillan_fluency_aligned.pt"
print(f"[*] Loading: {ckpt_path.name}")
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
sd = ckpt.get("model_state_dict", ckpt)
model.load_state_dict(sd, strict=False)
print(f"[+] Loaded (Step: {ckpt.get('step','N/A')}, Loss: {ckpt.get('loss','N/A')})")

# ─── Load Gold Datasets ──────────────────────────────────────────────────────
SEQ_LEN = 256  # Shorter for CPU speed

gold_samples = []
datasets = [
    ("Quillan_Clean_Reasoning_Gold_Dataset.jsonl", 2),
    ("Quillan_Hyper_Tune_Gold_Dataset.jsonl", 3),
    ("Quillan_Explanatory_Prose_Dataset.jsonl", 3),
    ("Quillan_General_Knowledge_Dataset.jsonl", 3),
]

for fname, repeat in datasets:
    fpath = REPO_ROOT / "training_data" / fname
    if not fpath.exists():
        continue
    count = 0
    with open(fpath, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line.strip())
            q = obj.get("question", obj.get("instruction", ""))
            ans = obj.get("response", obj.get("answer", ""))
            if not q or not ans:
                continue
            
            full = f"<|system|>\n# 🤖🧠 Quillan System Start 🧠🤖\n<|user|>\n{q}\n<|assistant|>\n{ans}\n<|end|>"
            prompt = f"<|system|>\n# 🤖🧠 Quillan System Start 🧠🤖\n<|user|>\n{q}\n<|assistant|>\n"
            
            toks = enc.encode(full)[:SEQ_LEN]
            p_toks = enc.encode(prompt)
            labs = [-100] * len(p_toks) + toks[len(p_toks):]
            labs = labs[:SEQ_LEN]
            
            if len(toks) < 10:
                continue
            for _ in range(repeat):
                gold_samples.append((toks, labs))
            count += 1
    print(f"  [DATA] {fname}: {count} × {repeat}")

random.shuffle(gold_samples)
print(f"[DATA] Total: {len(gold_samples)} samples (seqlen={SEQ_LEN})")

# ─── Training ────────────────────────────────────────────────────────────────
STEPS = 300
GRAD_ACCUM = 2
BASE_LR = 5e-5
MIN_LR = 5e-6

optimizer = torch.optim.AdamW(model.parameters(), lr=BASE_LR, weight_decay=0.01)

def quick_eval(step_num):
    model.eval()
    prompt = "<|system|>\n# 🤖🧠 Quillan System Start 🧠🤖\n<|user|>\nHello! Who are you?\n<|assistant|>\n"
    toks = enc.encode(prompt)
    gen = list(toks)
    with torch.no_grad():
        for _ in range(80):
            inp = torch.tensor([gen[-256:]], dtype=torch.long)
            logits = model(inp)
            if isinstance(logits, tuple):
                logits = logits[0]
            curr = logits[:, -1, :].clone()
            if len(gen) > 0:
                curr[0, gen[-1]] -= 50.0
            recent = gen[-32:]
            for tid in set(recent):
                curr[0, tid] -= 3.0 * recent.count(tid)
            next_tok = torch.argmax(curr, dim=-1).item()
            gen.append(next_tok)
            if next_tok == 50256:
                break
    out = enc.decode(gen[len(toks):])
    print(f"  [EVAL@{step_num}] {out[:250]}", flush=True)
    model.train()

print(f"\n[TRAIN] {STEPS} steps, accum={GRAD_ACCUM}, LR={BASE_LR}→{MIN_LR}")

model.train()
t0 = time.time()
best_loss = 999.0
idx = 0

for step in range(1, STEPS + 1):
    lr = MIN_LR + 0.5 * (BASE_LR - MIN_LR) * (1.0 + math.cos(math.pi * step / STEPS))
    for pg in optimizer.param_groups:
        pg['lr'] = lr
    
    optimizer.zero_grad()
    accum_loss = 0.0
    
    for _ in range(GRAD_ACCUM):
        toks, labs = gold_samples[idx % len(gold_samples)]
        idx += 1
        if idx % len(gold_samples) == 0:
            random.shuffle(gold_samples)
        
        x = torch.tensor([toks], dtype=torch.long)
        y = torch.tensor([labs], dtype=torch.long)
        
        logits, aux = model(x)
        loss = F.cross_entropy(
            logits[..., :-1, :].contiguous().view(-1, cfg.vocab_size),
            y[..., 1:].contiguous().view(-1),
            ignore_index=-100
        )
        total = (loss + 0.01 * aux) / GRAD_ACCUM
        total.backward()
        accum_loss += total.item() * GRAD_ACCUM
    
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    
    if accum_loss < best_loss:
        best_loss = accum_loss
    
    if step % 5 == 0 or step == 1:
        elapsed = time.time() - t0
        sps = elapsed / step
        eta = sps * (STEPS - step) / 60
        print(f"  step {step:3d}/{STEPS}  loss={accum_loss:.4f}  best={best_loss:.4f}  lr={lr:.6f}  ({sps:.1f}s/st, ETA {eta:.0f}m)", flush=True)
    
    if step % 50 == 0:
        quick_eval(step)
    
    if step % 100 == 0:
        torch.save({
            'model_state_dict': model.state_dict(),
            'step': step, 'loss': best_loss,
            'version': 'quillan-v5.3.1-fluency-aligned'
        }, save_path)
        print(f"  [SAVE] {save_path.name} (step {step})", flush=True)

# Final save
torch.save({
    'model_state_dict': model.state_dict(),
    'step': STEPS, 'loss': best_loss,
    'version': 'quillan-v5.3.1-fluency-aligned-final'
}, save_path)

print(f"\n[DONE] 🏆 Best loss: {best_loss:.4f}, Time: {(time.time()-t0)/60:.1f}m")
print(f"[SAVE] {save_path.name}")
print("\n[FINAL EVAL]")
quick_eval(STEPS)
