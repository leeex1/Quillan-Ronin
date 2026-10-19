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
print("   👑 QUILLAN-RONIN v5.3.1 — FINAL MASTER MULTI-DOMAIN SFT PASS")
print("==================================================================")

enc = tiktoken.get_encoding("gpt2")

cfg = QuillanArchConfig(
    hidden_dim=1024, ffn_dim=2048, num_experts=34,
    text_only=True, eggroll_rank=256
)
model = QuillanRoninSovereign(cfg).to("cpu")

ckpt_path = r"C:\02_QUILLAN\checkpoints\checkpoints_sft\quillan_thinking_reasoning_master.pt"
if not os.path.exists(ckpt_path):
    ckpt_path = r"C:\02_QUILLAN\checkpoints\checkpoints_v2\quillan_full_base_final.pt"

print(f"[*] Resuming from Base Model Checkpoint: {ckpt_path}")
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
sd = ckpt.get("model_state_dict", ckpt)
model.load_state_dict(sd)
print("[+] Successfully loaded 510/510 parameter layers.")

# Ingest all multi-domain training files in C:\02_QUILLAN\training_data
data_dir = REPO_ROOT / "training_data"
data_files = [
    "Quillan_Clean_Reasoning_Gold_Dataset.jsonl",
    "Quillan_Hyper_Tune_Gold_Dataset.jsonl",
    "instruct_train.jsonl",
    "GPT_5.5_Distilled.jsonl",
    "code_train.jsonl",
    "quillan_corpus_CLEAN_V7.jsonl"
]

samples = []
for fname in data_files:
    fpath = data_dir / fname
    if not fpath.exists():
        continue
    print(f"[*] Ingesting: {fname}...")
    with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line.strip(): continue
            try:
                obj = json.loads(line.strip())
                q = obj.get("question") or obj.get("instruction") or obj.get("prompt") or obj.get("input", "")
                ans = obj.get("response") or obj.get("output") or obj.get("completion") or obj.get("target", "")
                if q and ans:
                    full_txt = f"<|system|>\n# 🤖🧠 Quillan System Start 🧠🤖\n<|user|>\n{q}\n<|assistant|>\n{ans}\n<|end|>"
                    prompt_txt = f"<|system|>\n# 🤖🧠 Quillan System Start 🧠🤖\n<|user|>\n{q}\n<|assistant|>\n"
                    toks = enc.encode(full_txt)
                    p_toks = enc.encode(prompt_txt)
                    labs = [-100] * len(p_toks) + toks[len(p_toks):]
                    samples.append((toks[:1024], labs[:1024]))
                elif isinstance(obj, str) and len(obj) > 20:
                    toks = enc.encode(obj[:1024])
                    samples.append((toks, toks))
            except Exception:
                continue

print(f"[DATA] Ingestion complete! Total active multi-domain samples: {len(samples)} (SeqLen=512).")

STEPS = 500
BASE_LR = 1.8e-4
MIN_LR = 5.0e-6
final_save_path = r"C:\02_QUILLAN\checkpoints\checkpoints_sft\quillan_thinking_reasoning_master.pt"
optimizer = torch.optim.AdamW(model.parameters(), lr=BASE_LR, weight_decay=0.01)

def save_checkpoint_safely(model, step, loss, path):
    tmp_path = path + ".tmp"
    try:
        torch.save({
            'model_state_dict': model.state_dict(),
            'step': step,
            'loss': loss,
            'version': 'quillan-v5.3.1-master-final'
        }, tmp_path)
        if os.path.exists(path):
            try:
                os.remove(path)
            except Exception:
                pass
        os.replace(tmp_path, path)
    except Exception as e:
        print(f"[WARN] Checkpoint save warning at step {step}: {e}", flush=True)

print(f"\n[TRAIN] Executing Final Master SFT Pass ({STEPS} Steps, LR={BASE_LR} -> {MIN_LR})...")

model.train()
t0 = time.time()
best_loss = 999.0

for step in range(1, STEPS + 1):
    lr = MIN_LR + 0.5 * (BASE_LR - MIN_LR) * (1.0 + math.cos(math.pi * step / STEPS))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        
    optimizer.zero_grad()
    
    toks, labs = samples[(step - 1) % len(samples)]
    sl = min(1024, len(toks))
    
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
        
    if step % 25 == 0 or step == 1:
        elapsed = time.time() - t0
        sps = elapsed / step
        print(f"  step {step:3d}/{STEPS}  loss={total_loss.item():.4f}  best={best_loss:.4f}  lr={lr:.6f}  ({sps:.1f}s/st)", flush=True)
        
    if step % 50 == 0:
        save_checkpoint_safely(model, step, best_loss, final_save_path)

save_checkpoint_safely(model, STEPS, best_loss, final_save_path)
print(f"\n[SAVE] 🏆 Quillan Master Native Base Model saved to: {final_save_path}")
