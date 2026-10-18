#!/usr/bin/env python3
"""
👑 QUILLAN-RONIN v5.3.1 — FLUENCY ALIGNMENT SFT TRAINER
Base: quillan_hyper_tuned_v531.pt (Step 300, Loss 0.305)
Goal: Tighten sentence structure while preserving semantic core
Strategy: Low LR, 100% Gold SFT, target-masked loss, proper shuffling
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
print("   👑 QUILLAN-RONIN v5.3.1 — FLUENCY ALIGNMENT SFT TRAINER")
print("==================================================================")

# ─── Model Setup ──────────────────────────────────────────────────────────────
cfg = QuillanArchConfig(
    hidden_dim=1024, ffn_dim=2048, num_experts=34,
    vocab_size=50257, text_only=True, eggroll_rank=256
)
model = QuillanRoninSovereign(cfg).to("cpu")

ckpt_path = REPO_ROOT / "checkpoints" / "checkpoints_sft" / "quillan_thinking_reasoning_master.pt"
print(f"[*] Loading base checkpoint: {ckpt_path.name}")
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
sd = ckpt.get("model_state_dict", ckpt)
model.load_state_dict(sd, strict=False)
print(f"[+] Loaded (Step: {ckpt.get('step','N/A')}, Loss: {ckpt.get('loss','N/A')})")

# ─── Dataset Loading ──────────────────────────────────────────────────────────
# Use ALL high-quality SFT datasets for maximum variety
jsonl_files = [
    ("Quillan_Clean_Reasoning_Gold_Dataset.jsonl", 3),    # 1503 × 3 = high priority
    ("Quillan_Hyper_Tune_Gold_Dataset.jsonl", 5),          # 150 × 5 = high priority
    ("Quillan_Explanatory_Prose_Dataset.jsonl", 5),        # 100 × 5 = high priority
    ("Quillan_General_Knowledge_Dataset.jsonl", 5),        # 100 × 5 = high priority
    ("instruct_train.jsonl", 1),                           # 7217 × 1 = variety
    ("full_train.jsonl", 1),                               # 8706 × 1 = variety
]

gold_samples = []
for fname, repeat in jsonl_files:
    fpath = REPO_ROOT / "training_data" / fname
    if not fpath.exists():
        print(f"  [SKIP] {fname} not found")
        continue
    file_count = 0
    with open(fpath, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line.strip())
            # Handle different JSONL formats
            q = obj.get("question", obj.get("instruction", obj.get("prompt", "")))
            ans = obj.get("response", obj.get("answer", obj.get("output", obj.get("completion", ""))))
            if not q or not ans:
                continue
            
            full_txt = f"<|system|>\n# 🤖🧠 Quillan System Start 🧠🤖\n<|user|>\n{q}\n<|assistant|>\n{ans}\n<|end|>"
            prompt_txt = f"<|system|>\n# 🤖🧠 Quillan System Start 🧠🤖\n<|user|>\n{q}\n<|assistant|>\n"
            
            toks = enc.encode(full_txt)
            p_toks = enc.encode(prompt_txt)
            # Target-masked: only compute loss on the assistant's response
            labs = [-100] * len(p_toks) + toks[len(p_toks):]
            
            # Truncate to 512
            toks = toks[:512]
            labs = labs[:512]
            
            if len(toks) < 10:
                continue
                
            for _ in range(repeat):
                gold_samples.append((toks, labs))
            file_count += 1
    print(f"  [DATA] {fname}: {file_count} samples × {repeat} = {file_count * repeat}")

random.shuffle(gold_samples)
print(f"\n[DATA] Total training pool: {len(gold_samples)} samples")

# ─── Training Config ──────────────────────────────────────────────────────────
# Conservative LR to preserve semantic core while teaching fluency
STEPS = 500
GRAD_ACCUM = 4
BASE_LR = 5e-5
MIN_LR = 5e-6
SAVE_EVERY = 50
EVAL_EVERY = 25

optimizer = torch.optim.AdamW(model.parameters(), lr=BASE_LR, weight_decay=0.01)

print(f"\n[TRAIN] Fluency Alignment SFT ({STEPS} steps, LR={BASE_LR} → {MIN_LR}, accum={GRAD_ACCUM})")
print(f"[TRAIN] Effective batch = {GRAD_ACCUM} samples/step")

# ─── Quick Inference Check Function ───────────────────────────────────────────
def quick_eval(model, step_num):
    model.eval()
    prompt = "<|system|>\n# 🤖🧠 Quillan System Start 🧠🤖\n<|user|>\nHello! Who are you?\n<|assistant|>\n"
    toks = enc.encode(prompt)
    generated = list(toks)
    with torch.no_grad():
        for _ in range(60):
            inp = torch.tensor([generated[-256:]], dtype=torch.long)
            logits = model(inp)
            if isinstance(logits, tuple):
                logits = logits[0]
            curr = logits[:, -1, :].clone()
            # Anti-stutter
            if len(generated) > 0:
                curr[0, generated[-1]] -= 50.0
            recent = generated[-32:]
            for tid in set(recent):
                curr[0, tid] -= 3.0 * recent.count(tid)
            # Greedy for reproducibility
            next_tok = torch.argmax(curr, dim=-1).item()
            generated.append(next_tok)
            if next_tok == 50256:
                break
    output = enc.decode(generated[len(toks):])
    print(f"  [EVAL step {step_num}] {output[:200]}")
    model.train()

# ─── Training Loop ────────────────────────────────────────────────────────────
model.train()
t0 = time.time()
best_loss = 999.0
sample_idx = 0

for step in range(1, STEPS + 1):
    # Cosine LR schedule
    lr = MIN_LR + 0.5 * (BASE_LR - MIN_LR) * (1.0 + math.cos(math.pi * step / STEPS))
    for pg in optimizer.param_groups:
        pg['lr'] = lr
    
    optimizer.zero_grad()
    accum_loss = 0.0
    
    for _ in range(GRAD_ACCUM):
        toks, labs = gold_samples[sample_idx % len(gold_samples)]
        sample_idx += 1
        
        # Reshuffle every epoch
        if sample_idx % len(gold_samples) == 0:
            random.shuffle(gold_samples)
        
        sl = min(512, len(toks))
        x = torch.tensor([toks[:sl]], dtype=torch.long)
        y = torch.tensor([labs[:sl]], dtype=torch.long)
        
        logits, aux_loss = model(x)
        ce_loss = F.cross_entropy(
            logits[..., :-1, :].contiguous().view(-1, cfg.vocab_size),
            y[..., 1:].contiguous().view(-1),
            ignore_index=-100
        )
        total_loss = (ce_loss + 0.01 * aux_loss) / GRAD_ACCUM
        total_loss.backward()
        accum_loss += total_loss.item() * GRAD_ACCUM
    
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    
    if accum_loss < best_loss:
        best_loss = accum_loss
    
    if step % 10 == 0 or step == 1:
        elapsed = time.time() - t0
        sps = elapsed / step
        print(f"  step {step:4d}/{STEPS}  loss={accum_loss:.4f}  best={best_loss:.4f}  lr={lr:.6f}  ({sps:.1f}s/st)", flush=True)
    
    # Periodic eval
    if step % EVAL_EVERY == 0:
        quick_eval(model, step)
    
    # Save checkpoint
    if step % SAVE_EVERY == 0:
        torch.save({
            'model_state_dict': model.state_dict(),
            'step': step,
            'loss': best_loss,
            'version': 'quillan-v5.3.1-fluency-aligned'
        }, ckpt_path)
        print(f"  [SAVE] Checkpoint saved (step {step}, best_loss={best_loss:.4f})", flush=True)

# ─── Final Save ───────────────────────────────────────────────────────────────
torch.save({
    'model_state_dict': model.state_dict(),
    'step': STEPS,
    'loss': best_loss,
    'version': 'quillan-v5.3.1-fluency-aligned-final'
}, ckpt_path)

print(f"\n[DONE] 🏆 Fluency Alignment Complete!")
print(f"  Final best loss: {best_loss:.4f}")
print(f"  Total time: {time.time() - t0:.0f}s")
print(f"  Saved to: {ckpt_path.name}")

# Final inference test
print("\n[FINAL EVAL]")
quick_eval(model, STEPS)

gc.collect()
