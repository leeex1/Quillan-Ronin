#!/usr/bin/env python3
"""
👑 QUILLAN-RONIN v5.3.1 — DIRECT CONVERSATIONAL ENGLISH SFT
Trains the normalized spiderweb architecture to produce direct, fluent English instruction-following responses.
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
print("   👑 QUILLAN-RONIN v5.3.1 — DIRECT CONVERSATIONAL ENGLISH SFT")
print("==================================================================")

cfg = QuillanArchConfig(
    hidden_dim=1024, ffn_dim=2048, num_experts=34,
    vocab_size=50257, text_only=True, eggroll_rank=256
)
model = QuillanRoninSovereign(cfg).to("cpu")

ckpt_path = REPO_ROOT / "checkpoints" / "checkpoints_sft" / "quillan_thinking_reasoning_master.pt"
print(f"[*] Loading Master Model: {ckpt_path.name}")
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
sd = ckpt.get("model_state_dict", ckpt)
model.load_state_dict(sd, strict=False)
print(f"[+] Loaded successfully.")

# 100% parameter gradient updates
for param in model.parameters():
    param.requires_grad = True

# ─── DATASET INGESTION: CLEAN CONVERSATIONAL ENGLISH ─────────────────────────
SEQ_LEN = 192
data_dir = REPO_ROOT / "training_data"

dataset_configs = [
    ("Quillan_Clean_Reasoning_Gold_Dataset.jsonl", 3),
    ("Quillan_Explanatory_Prose_Dataset.jsonl", 6),
    ("Quillan_General_Knowledge_Dataset.jsonl", 6),
    ("instruct_train.jsonl", 1),
    ("full_train.jsonl", 1),
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
            try: obj = json.loads(line_str)
            except: continue
            
            q = (obj.get("question") or obj.get("instruction") or obj.get("prompt") or "")
            ans = (obj.get("response") or obj.get("answer") or obj.get("output") or "")
            if not q and not ans and "messages" in obj:
                for m in obj["messages"]:
                    if m.get("role") == "user": q = m.get("content", "")
                    elif m.get("role") == "assistant": ans = m.get("content", "").split("</think>")[-1].strip()
            if not q or not ans: continue
            
            # Strip any residual system banners from response so model outputs direct English
            for banner in ["# 🤖🧠 Quillan System Start 🧠🤖", "# 🤖", "# 🧠", "# Quillan System Start", "👑 QUILLAN-RONIN"]:
                if ans.startswith(banner):
                    ans = ans.replace(banner, "").strip()
            if not ans: continue
            
            # Clean Conversational Prompt Format
            full_txt = f"<|system|>\nYou are Quillan-Ronin, an advanced conversational AI assistant.\n<|user|>\n{q}\n<|assistant|>\n{ans}\n<|end|>"
            prompt_txt = f"<|system|>\nYou are Quillan-Ronin, an advanced conversational AI assistant.\n<|user|>\n{q}\n<|assistant|>\n"
            
            toks = enc.encode(full_txt)[:SEQ_LEN]
            p_toks = enc.encode(prompt_txt)
            labs = [-100] * len(p_toks) + toks[len(p_toks):]
            labs = labs[:SEQ_LEN]
            
            if len(toks) < 15: continue
            for _ in range(repeat):
                training_samples.append((toks, labs))
            count += 1

random.seed(42)
random.shuffle(training_samples)
print(f"[DATA] Ingested {len(training_samples)} High-Quality Conversational English Samples (seq_len={SEQ_LEN})\n")

# ─── TRAINING LOOP ────────────────────────────────────────────────────────────
STEPS = 350
BASE_LR = 3e-5
MIN_LR = 2e-6

optimizer = torch.optim.AdamW(model.parameters(), lr=BASE_LR, weight_decay=0.01)

def quick_eval(step_num):
    model.eval()
    test_q = "Explain what a function is in Python in one simple sentence."
    prompt_str = f"<|system|>\nYou are Quillan-Ronin, an advanced conversational AI assistant.\n<|user|>\n{test_q}\n<|assistant|>\n"
    toks = enc.encode(prompt_str)
    gen = list(toks)
    with torch.no_grad():
        for _ in range(50):
            inp = torch.tensor([gen[-192:]], dtype=torch.long)
            logits = model(inp)
            if isinstance(logits, tuple): logits = logits[0]
            curr = logits[:, -1, :].clone()
            if len(gen) > 0: curr[0, gen[-1]] -= 50.0
            next_tok = torch.argmax(curr, dim=-1).item()
            gen.append(next_tok)
            if next_tok == 50256: break
    out = enc.decode(gen[len(toks):])
    print(f"\n  [EVAL @ step {step_num}] Prompt: '{test_q}'\n  Response: {out.strip()[:180]}\n", flush=True)
    model.train()

print(f"[TRAIN] Launching Conversational English SFT ({STEPS} steps, LR={BASE_LR} -> {MIN_LR})...\n")

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
    
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    
    val = total_loss.item()
    if val < best_loss:
        best_loss = val
        
    if step % 10 == 0 or step == 1:
        elapsed = time.time() - t0
        sps = elapsed / step
        eta_m = sps * (STEPS - step) / 60.0
        print(f"  step {step:3d}/{STEPS}  loss={val:.4f}  best={best_loss:.4f}  lr={lr:.6f}  ({sps:.1f}s/st, ETA {eta_m:.1f}m)", flush=True)
        
    if step % 50 == 0:
        quick_eval(step)
        torch.save({
            'model_state_dict': model.state_dict(),
            'step': step, 'loss': best_loss,
            'version': 'quillan-v5.3.1-conversational-master'
        }, ckpt_path)
        print(f"  [CHECKPOINT] Auto-saved master model at step {step}.", flush=True)

# Final Save
torch.save({
    'model_state_dict': model.state_dict(),
    'step': STEPS, 'loss': best_loss,
    'version': 'quillan-v5.3.1-conversational-master-final'
}, ckpt_path)

print(f"\n[DONE] 🏆 Conversational English Training Complete! Best Loss: {best_loss:.4f} in {(time.time()-t0)/60:.1f}m")
quick_eval(STEPS)
