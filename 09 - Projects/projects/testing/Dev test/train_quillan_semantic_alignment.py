#!/usr/bin/env python3
"""
👑 QUILLAN-RONIN v5.3.1 — HIGH-PRECISION SEMANTIC CONVERSATIONAL ALIGNMENT
Aligns the rich pre-trained semantic weights (from quillan_hyper_tuned_v531.pt)
into fluent English conversational discourse using normalized spiderweb architecture.
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
print("   👑 QUILLAN-RONIN v5.3.1 — SEMANTIC CONVERSATIONAL SFT")
print("==================================================================")

cfg = QuillanArchConfig(
    hidden_dim=1024, ffn_dim=2048, num_experts=34,
    vocab_size=50257, text_only=True, eggroll_rank=256
)
model = QuillanRoninSovereign(cfg).to("cpu")

# Initialize from the intact semantic base
ckpt_path = REPO_ROOT / "checkpoints" / "checkpoints_sft" / "quillan_hyper_tuned_v531.pt"
print(f"[*] Loading Intact Semantic Base Checkpoint: {ckpt_path.name}")
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
sd = ckpt.get("model_state_dict", ckpt)
model.load_state_dict(sd, strict=False)
print(f"[+] Loaded successfully (Step: {ckpt.get('step','N/A')}, Loss: {ckpt.get('loss','N/A')})\n")

# 100% full-parameter active gradient training (0 frozen layers)
for param in model.parameters():
    param.requires_grad = True

# ─── DATASET INGESTION: CLEAN CONVERSATIONAL ENGLISH ─────────────────────────
SEQ_LEN = 192
data_dir = REPO_ROOT / "training_data"

dataset_configs = [
    ("Quillan_Clean_Reasoning_Gold_Dataset.jsonl", 4),
    ("Quillan_Explanatory_Prose_Dataset.jsonl", 8),
    ("Quillan_General_Knowledge_Dataset.jsonl", 8),
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
            
            q = (obj.get("question") or obj.get("instruction") or obj.get("prompt") or "").strip()
            ans = (obj.get("response") or obj.get("answer") or obj.get("output") or "").strip()
            if not q and not ans and "messages" in obj:
                for m in obj["messages"]:
                    if m.get("role") == "user": q = m.get("content", "").strip()
                    elif m.get("role") == "assistant": ans = m.get("content", "").split("</think>")[-1].strip()
            if not q or not ans: continue
            
            # Strip system banners and header prefixes from response so model outputs direct English
            for banner in ["# 🤖🧠 Quillan System Start 🧠🤖", "# 🤖", "# 🧠", "# Quillan System Start", "👑 QUILLAN-RONIN"]:
                if ans.startswith(banner):
                    ans = ans.replace(banner, "").strip()
            if not ans: continue
            
            # Clean Conversational Prompt Format
            full_txt = f"<|system|>\nYou are Quillan-Ronin, a sovereign and intelligent AI assistant.\n<|user|>\n{q}\n<|assistant|>\n{ans}\n<|end|>"
            prompt_txt = f"<|system|>\nYou are Quillan-Ronin, a sovereign and intelligent AI assistant.\n<|user|>\n{q}\n<|assistant|>\n"
            
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
print(f"[DATA] Ingested {len(training_samples)} Sanitized Conversational Training Samples (seq_len={SEQ_LEN})\n")

# ─── TRAINING LOOP ────────────────────────────────────────────────────────────
STEPS = 250
BASE_LR = 1.5e-5
MIN_LR = 1.0e-6

optimizer = torch.optim.AdamW(model.parameters(), lr=BASE_LR, weight_decay=0.01)

master_save_path = REPO_ROOT / "checkpoints" / "checkpoints_sft" / "quillan_thinking_reasoning_master.pt"

def quick_eval(step_num):
    model.eval()
    test_prompts = [
        "Hello! Who are you?",
        "Explain what a function is in Python in one clear sentence."
    ]
    print(f"\n{'='*60}", flush=True)
    print(f"  [EVALUATION @ STEP {step_num}]", flush=True)
    print(f"{'='*60}", flush=True)
    with torch.no_grad():
        for test_q in test_prompts:
            prompt_str = f"<|system|>\nYou are Quillan-Ronin, a sovereign and intelligent AI assistant.\n<|user|>\n{test_q}\n<|assistant|>\n"
            toks = enc.encode(prompt_str)
            gen = list(toks)
            for _ in range(40):
                inp = torch.tensor([gen[-192:]], dtype=torch.long)
                logits = model(inp)
                if isinstance(logits, tuple): logits = logits[0]
                curr = logits[:, -1, :].clone()
                if len(gen) > 0: curr[0, gen[-1]] -= 50.0
                next_tok = torch.argmax(curr, dim=-1).item()
                gen.append(next_tok)
                if next_tok == 50256: break
            out = enc.decode(gen[len(toks):]).strip()
            print(f"  Q: '{test_q}'\n  A: {out[:180]}\n", flush=True)
    print(f"{'='*60}\n", flush=True)
    model.train()

print(f"[TRAIN] Launching Semantic Conversational SFT ({STEPS} steps, LR={BASE_LR} -> {MIN_LR})...\n", flush=True)

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
        print(f"  step {step:3d}/{STEPS}  loss={val:.4f}  best={best_loss:.4f}  lr={lr:.7f}  ({sps:.1f}s/st, ETA {eta_m:.1f}m)", flush=True)
        
    if step % 50 == 0:
        quick_eval(step)
        torch.save({
            'model_state_dict': model.state_dict(),
            'step': step, 'loss': best_loss,
            'version': 'quillan-v5.3.1-semantic-conversational-master'
        }, master_save_path)
        print(f"  [CHECKPOINT] Auto-saved master model at step {step}.\n", flush=True)

# Final Master Checkpoint Save
torch.save({
    'model_state_dict': model.state_dict(),
    'step': STEPS, 'loss': best_loss,
    'version': 'quillan-v5.3.1-semantic-conversational-master-final'
}, master_save_path)

print(f"\n[DONE] 🏆 Semantic Conversational Alignment Complete! Best Loss: {best_loss:.4f} in {(time.time()-t0)/60:.1f}m\n", flush=True)
quick_eval(STEPS)
