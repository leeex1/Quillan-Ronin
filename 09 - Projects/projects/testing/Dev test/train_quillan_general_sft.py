#!/usr/bin/env python3
"""
👑 QUILLAN-RONIN v5.3.1 — MULTI-DOMAIN GENERAL SFT & DUAL-BRAIN INGESTION CALIBRATION
Trains Q1/Q2 analytical-intuitive pathways across General Knowledge, Code, Dialogue, Science, and Reasoning.
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
print("   👑 QUILLAN-RONIN v5.3.1 — GENERAL MULTI-DOMAIN SFT")
print("==================================================================")

cfg = QuillanArchConfig(
    hidden_dim=1024, ffn_dim=2048, num_experts=34,
    vocab_size=50257, text_only=True, eggroll_rank=256
)
model = QuillanRoninSovereign(cfg).to("cpu")

ckpt_path = REPO_ROOT / "checkpoints" / "checkpoints_sft" / "quillan_thinking_reasoning_master.pt"
save_master_path = REPO_ROOT / "checkpoints" / "checkpoints_sft" / "quillan_thinking_reasoning_master.pt"
save_backup_path = REPO_ROOT / "checkpoints" / "checkpoints_sft" / "quillan_general_aligned_master.pt"

print(f"[*] Loading base checkpoint: {ckpt_path.name}")
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
sd = ckpt.get("model_state_dict", ckpt)
model.load_state_dict(sd, strict=False)
print(f"[+] Loaded successfully (Prior Step: {ckpt.get('step','N/A')}, Prior Loss: {ckpt.get('loss','N/A')})")

# ─── Load Multi-Domain Datasets ───────────────────────────────────────────────
SEQ_LEN = 256
data_dir = REPO_ROOT / "training_data"

dataset_configs = [
    ("Quillan_Clean_Reasoning_Gold_Dataset.jsonl", 2),
    ("Quillan_Hyper_Tune_Gold_Dataset.jsonl", 3),
    ("Quillan_Explanatory_Prose_Dataset.jsonl", 3),
    ("Quillan_General_Knowledge_Dataset.jsonl", 3),
    ("code_train.jsonl", 1),
    ("instruct_train.jsonl", 1),
    ("full_train.jsonl", 1),
    ("quillan_science_absolute.jsonl", 1),
]

training_samples = []

for fname, repeat in dataset_configs:
    fpath = data_dir / fname
    if not fpath.exists():
        continue
    count = 0
    with open(fpath, "r", encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if not line_str:
                continue
            try:
                obj = json.loads(line_str)
            except Exception:
                continue
            
            # Universal key extraction
            q = (obj.get("question") or obj.get("instruction") or 
                 obj.get("prompt") or obj.get("input") or "")
            ans = (obj.get("response") or obj.get("answer") or 
                   obj.get("output") or obj.get("completion") or "")
            
            # If messages format (OpenAI / Anthropic standard)
            if not q and not ans and "messages" in obj and isinstance(obj["messages"], list):
                for m in obj["messages"]:
                    if m.get("role") == "user":
                        q = m.get("content", "")
                    elif m.get("role") == "assistant":
                        ans = m.get("content", "")
                        # Clean internal think blocks if needed to focus on direct conversational prose
                        if "</think>" in ans:
                            ans = ans.split("</think>")[-1].strip()
            
            # If text key format
            if not q and not ans and "text" in obj:
                raw_text = obj["text"]
                if "<|user|>" in raw_text and "<|assistant|>" in raw_text:
                    parts = raw_text.split("<|assistant|>")
                    q = parts[0].replace("<|system|>", "").replace("<|user|>", "").strip()
                    ans = parts[1].replace("<|end|>", "").strip()
            
            if not q or not ans:
                continue
            
            # Clean Conversational Prompt Format
            full_txt = f"<|system|>\n# 🤖🧠 Quillan System Start 🧠🤖\n<|user|>\n{q}\n<|assistant|>\n{ans}\n<|end|>"
            prompt_txt = f"<|system|>\n# 🤖🧠 Quillan System Start 🧠🤖\n<|user|>\n{q}\n<|assistant|>\n"
            
            toks = enc.encode(full_txt)[:SEQ_LEN]
            p_toks = enc.encode(prompt_txt)
            labs = [-100] * len(p_toks) + toks[len(p_toks):]
            labs = labs[:SEQ_LEN]
            
            if len(toks) < 12:
                continue
                
            for _ in range(repeat):
                training_samples.append((toks, labs))
            count += 1
            
    print(f"  [DATA] {fname:45s}: {count:5d} samples × {repeat}")

random.seed(42)
random.shuffle(training_samples)
print(f"\n[DATA] Total Multi-Domain Corpus: {len(training_samples)} balanced samples (seq_len={SEQ_LEN})\n")

# ─── Training Optimization Setup ──────────────────────────────────────────────
STEPS = 250
GRAD_ACCUM = 2
BASE_LR = 2.5e-5
MIN_LR = 2.0e-6

# Explicit parameter groups to safely calibrate Q1/Q2 without disturbing MoE
ingest_params = [p for n, p in model.named_parameters() if 'ingestion' in n and p.requires_grad]
backbone_params = [p for n, p in model.named_parameters() if 'ingestion' not in n and p.requires_grad]

optimizer = torch.optim.AdamW([
    {'params': ingest_params, 'lr': BASE_LR * 1.5, 'weight_decay': 0.001},
    {'params': backbone_params, 'lr': BASE_LR, 'weight_decay': 0.01}
])

def quick_eval(step_num):
    model.eval()
    test_q = "Explain the difference between a list and a set in Python."
    prompt = f"<|system|>\n# 🤖🧠 Quillan System Start 🧠🤖\n<|user|>\n{test_q}\n<|assistant|>\n"
    toks = enc.encode(prompt)
    gen = list(toks)
    with torch.no_grad():
        for _ in range(70):
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
    print(f"  [EVAL @ step {step_num}] Prompt: '{test_q}'\n  Response: {out.strip()[:200]}\n", flush=True)
    model.train()

# ─── Training Execution Loop ──────────────────────────────────────────────────
print(f"[TRAIN] Launching General Multi-Domain Alignment ({STEPS} steps, accum={GRAD_ACCUM})...")
model.train()
t0 = time.time()
best_loss = 999.0
idx = 0

for step in range(1, STEPS + 1):
    lr_factor = MIN_LR + 0.5 * (BASE_LR - MIN_LR) * (1.0 + math.cos(math.pi * step / STEPS))
    for pg in optimizer.param_groups:
        pg['lr'] = lr_factor
        
    optimizer.zero_grad()
    accum_loss = 0.0
    
    for _ in range(GRAD_ACCUM):
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
        total = (loss + 0.005 * aux) / GRAD_ACCUM
        total.backward()
        accum_loss += total.item() * GRAD_ACCUM
        
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    
    if accum_loss < best_loss:
        best_loss = accum_loss
        
    if step % 10 == 0 or step == 1:
        elapsed = time.time() - t0
        sps = elapsed / step
        eta_min = sps * (STEPS - step) / 60.0
        bypass_val = torch.sigmoid(model.ingestion.ingest_bypass).item()
        print(f"  step {step:3d}/{STEPS}  loss={accum_loss:.4f}  best={best_loss:.4f}  bypass={bypass_val:.3f}  ({sps:.1f}s/st, ETA {eta_min:.0f}m)", flush=True)
        
    if step % 50 == 0:
        quick_eval(step)
        torch.save({
            'model_state_dict': model.state_dict(),
            'step': step, 'loss': best_loss,
            'version': 'quillan-v5.3.1-general-sft-checkpoint'
        }, save_master_path)
        print(f"  [CHECKPOINT] Auto-saved master checkpoint at step {step}.", flush=True)

# Final Checkpoint Save
torch.save({
    'model_state_dict': model.state_dict(),
    'step': STEPS, 'loss': best_loss,
    'version': 'quillan-v5.3.1-general-sft-master'
}, save_master_path)

torch.save({
    'model_state_dict': model.state_dict(),
    'step': STEPS, 'loss': best_loss,
    'version': 'quillan-v5.3.1-general-sft-master'
}, save_backup_path)

print(f"\n[DONE] 🏆 Multi-Domain Training & Q1/Q2 Calibration Complete!")
print(f"  Final Best Loss: {best_loss:.4f} | Total Time: {(time.time()-t0)/60:.1f}m")
print(f"  Master Checkpoint Saved: {save_master_path.name}")
gc.collect()
