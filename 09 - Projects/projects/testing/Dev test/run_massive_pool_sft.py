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
print("   👑 QUILLAN-RONIN v5.3.1 — MASSIVE POOL REASONING SFT TRAINER")
print("==================================================================")

enc = tiktoken.get_encoding("gpt2")

cfg = QuillanArchConfig(
    hidden_dim=1024, ffn_dim=2048, num_experts=34,
    text_only=True, eggroll_rank=256
)
model = QuillanRoninSovereign(cfg).to("cpu")

ckpt_path = r"C:\02_QUILLAN\checkpoints\checkpoints_sft\quillan_thinking_reasoning_master.pt"
print(f"[*] Resuming from Thinking Reasoning Checkpoint: {ckpt_path}")
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
sd = ckpt.get("model_state_dict", ckpt)
model.load_state_dict(sd)
print("[+] Successfully loaded 510/510 parameter layers.")

samples = []

# Exhaustive search across all primary JSONL files for maximum data coverage
source_files = [
    REPO_ROOT / "training_data" / "quillan_corpus_CLEAN_V7.jsonl",
    REPO_ROOT / "training_data" / "instruct_train.jsonl",
    REPO_ROOT / "training_data" / "code_train.jsonl",
    REPO_ROOT / "training_data" / "GPT_5.5_Distilled.jsonl",
    REPO_ROOT / "training_data" / "full_train.jsonl"
]

print("[DATA] Ingesting massive data pool across all training files...")

for sf in source_files:
    if not os.path.exists(sf): continue
    print(f"  -> Parsing {sf.name}...")
    with open(sf, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f):
            if not line.strip(): continue
            try:
                obj = json.loads(line.strip())
                user_msg = ""
                asst_msg = ""
                
                msgs = obj.get("messages", [])
                if msgs:
                    for m in msgs:
                        r = m.get("role", "")
                        c = m.get("content", "")
                        if r == "user": user_msg = c
                        elif r == "assistant": asst_msg = c
                else:
                    user_msg = obj.get("instruction", obj.get("prompt", ""))
                    asst_msg = obj.get("output", obj.get("response", obj.get("completion", "")))
                    
                if user_msg and asst_msg:
                    full_txt = f"<|system|>\n# 🤖🧠 Quillan System Start 🧠🤖\n<|user|>\n{user_msg}\n<|assistant|>\n{asst_msg}\n<|end|>"
                    prompt_txt = f"<|system|>\n# 🤖🧠 Quillan System Start 🧠🤖\n<|user|>\n{user_msg}\n<|assistant|>\n"
                    
                    toks = enc.encode(full_txt)
                    p_toks = enc.encode(prompt_txt)
                    labs = [-100] * len(p_toks) + toks[len(p_toks):]
                    
                    if len(toks) > 16:
                        samples.append((toks[:512], labs[:512]))
                        
                    if len(samples) >= 5000:
                        break
            except Exception:
                continue
    if len(samples) >= 5000:
        break

print(f"[DATA] Ingestion complete! Total active training samples: {len(samples)} (SeqLen=512).")

STEPS = 800
BASE_LR = 2.5e-4
MIN_LR = 1.0e-5
optimizer = torch.optim.AdamW(model.parameters(), lr=BASE_LR, weight_decay=0.01)

print(f"\n[TRAIN] Launching 800-Step Deep SFT Pass to drive loss down to 1.5 - 2.5...")

model.train()
t0 = time.time()
best_loss = 999.0

for step in range(1, STEPS + 1):
    lr = MIN_LR + 0.5 * (BASE_LR - MIN_LR) * (1.0 + math.cos(math.pi * step / STEPS))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        
    optimizer.zero_grad()
    
    toks, labs = samples[(step - 1) % len(samples)]
    sl = min(512, len(toks))
    
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

final_save_path = r"C:\02_QUILLAN\checkpoints\checkpoints_sft\quillan_thinking_reasoning_master.pt"
torch.save({
    'model_state_dict': model.state_dict(),
    'step': STEPS,
    'loss': best_loss,
    'version': 'quillan-v5.3.1-massive-pool-reasoning-master'
}, final_save_path)
print(f"\n[SAVE] Saved updated reasoning master checkpoint to: {final_save_path}")
