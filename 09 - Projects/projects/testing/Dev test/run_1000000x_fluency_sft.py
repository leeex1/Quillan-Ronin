import os
import sys
import time
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
print("   👑 QUILLAN-RONIN v5.3.1 — 1,000,000X FLUENCY SFT TRAINER")
print("==================================================================")

enc = tiktoken.get_encoding("gpt2")

cfg = QuillanArchConfig(
    hidden_dim=1024, ffn_dim=2048, num_experts=34,
    text_only=True, eggroll_rank=256
)
model = QuillanRoninSovereign(cfg).to("cpu")

init_ckpt_path = r"C:\02_QUILLAN\checkpoints\checkpoints_sft\quillan_gpt2_semantic_init.pt"
print(f"[*] Loading GPT-2 Semantic Backbone initialized checkpoint: {init_ckpt_path}")
ckpt = torch.load(init_ckpt_path, map_location="cpu", weights_only=False)
sd = ckpt.get("model_state_dict", ckpt)
model.load_state_dict(sd)
print("[+] Loaded semantic backbone model weights.")

# Freeze txt_emb and txt_dec so English vocabulary manifold stays 100% pristine
model.ingestion.txt_emb.weight.requires_grad = False
model.txt_dec.weight.requires_grad = False

# Prepare High-Quality Target-Masked Dialogue SFT Data
samples = []

# 1. Samurai Seed Dataset
seed_path = r"C:\02_QUILLAN\training_data\Quillan_Ronin_v5.3.1_Samurai_Training_Seed_Dataset.jsonl"
if os.path.exists(seed_path):
    with open(seed_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            obj = json.loads(line.strip())
            q = obj.get("question", "")
            ans = obj.get("final_output", "")
            if q and ans:
                full_txt = f"<|user|>\n{q}\n<|assistant|>\n{ans}\n<|end|>"
                prompt_txt = f"<|user|>\n{q}\n<|assistant|>\n"
                toks = enc.encode(full_txt)
                p_toks = enc.encode(prompt_txt)
                labs = [-100] * len(p_toks) + toks[len(p_toks):]
                samples.append((toks[:256], labs[:256]))

# 2. GPT 5.5 Distilled Dataset
distilled_path = r"C:\02_QUILLAN\training_data\GPT_5.5_Distilled.jsonl"
if os.path.exists(distilled_path):
    with open(distilled_path, "r", encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f):
            if i >= 200: break
            if not line.strip(): continue
            try:
                obj = json.loads(line.strip())
                if "prompt" in obj and "response" in obj:
                    q = obj["prompt"]
                    ans = obj["response"]
                    full_txt = f"<|user|>\n{q}\n<|assistant|>\n{ans}\n<|end|>"
                    prompt_txt = f"<|user|>\n{q}\n<|assistant|>\n"
                    toks = enc.encode(full_txt)
                    p_toks = enc.encode(prompt_txt)
                    labs = [-100] * len(p_toks) + toks[len(p_toks):]
                    samples.append((toks[:256], labs[:256]))
            except Exception:
                pass

print(f"[DATA] Prepared {len(samples)} clean target-masked SFT samples.")

# High Learning Rate for Fast MoE Alignment
STEPS = 80
LR = 5e-4
trainable_params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.AdamW(trainable_params, lr=LR, weight_decay=0.01)

print(f"\n[TRAIN] Running {STEPS} MoE Alignment Steps (LR={LR}, Trainable Params: {len(trainable_params)} tensors)...")

model.train()
t0 = time.time()
best_loss = 999.0

for step in range(1, STEPS + 1):
    optimizer.zero_grad()
    
    toks, labs = samples[(step - 1) % len(samples)]
    sl = min(256, len(toks))
    
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
    
    torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
    optimizer.step()
    
    if total_loss.item() < best_loss:
        best_loss = total_loss.item()
        
    if step % 5 == 0 or step == 1:
        elapsed = time.time() - t0
        sps = elapsed / step
        print(f"  step {step:2d}/{STEPS}  loss={total_loss.item():.4f}  best={best_loss:.4f}  ({sps:.1f}s/st)", flush=True)

save_path = r"C:\02_QUILLAN\checkpoints\checkpoints_sft\quillan_1000000x_fluent.pt"
torch.save({
    'model_state_dict': model.state_dict(),
    'step': STEPS,
    'loss': best_loss,
    'version': 'quillan-v5.3.1-1000000x-fluent'
}, save_path)
print(f"\n[SAVE] 1000000X Fluent Checkpoint saved to: {save_path}")

# Test Speech Generation
model.eval()

def generate_speech(prompt, max_tokens=100, temp=0.7, top_p=0.9):
    tokens = enc.encode(prompt)
    generated = list(tokens)
    
    print(f"\n==================================================")
    print(f"PROMPT:\n{prompt.strip()}")
    print("==================================================")
    print("RESPONSE:\n", end="", flush=True)
    
    with torch.no_grad():
        for _ in range(max_tokens):
            inp = torch.tensor([generated[-128:]], dtype=torch.long)
            logits = model(inp)
            if isinstance(logits, tuple):
                logits = logits[0]
            curr_logits = logits[:, -1, :].clone()
            
            for tid in set(generated[-32:]):
                curr_logits[0, tid] -= 3.0

            if generated[-1] == 220:
                curr_logits[0, 220] = float('-inf')

            if temp > 0:
                scaled_logits = curr_logits / temp
                probs = F.softmax(scaled_logits, dim=-1)
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                cum_probs = torch.cumsum(sorted_probs, dim=-1)
                sorted_indices_to_remove = cum_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                scaled_logits[indices_to_remove] = float('-inf')
                probs = F.softmax(scaled_logits, dim=-1)
                next_tok = torch.multinomial(probs, 1).item()
            else:
                next_tok = torch.argmax(curr_logits, dim=-1).item()

            generated.append(next_tok)
            word_bytes = enc.decode_bytes([next_tok])
            word_str = word_bytes.decode('utf-8', errors='ignore')
            print(word_str, end="", flush=True)

            if next_tok == 50256:
                break
    print("\n" + "-" * 50)

print("\n=== 1,000,000X FLUENCE MODEL SPEECH DEMO ===")
prompts = [
    "<|user|>\nHello! Who are you?\n<|assistant|>\n",
    "<|user|>\nWhat can you help me with?\n<|assistant|>\n"
]

for p in prompts:
    generate_speech(p, max_tokens=80, temp=0.7, top_p=0.9)
