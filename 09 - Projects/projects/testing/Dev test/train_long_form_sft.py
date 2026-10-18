import os
import sys
import time
import math
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
print("   👑 QUILLAN-RONIN v5.3.1 — LONG-FORM TARGET-MASKED SFT TRAINER")
print("==================================================================")

enc = tiktoken.get_encoding("gpt2")

cfg = QuillanArchConfig(
    hidden_dim=1024, ffn_dim=2048, num_experts=34,
    text_only=True, eggroll_rank=256
)
model = QuillanRoninSovereign(cfg).to("cpu")

ckpt_path = r"C:\02_QUILLAN\checkpoints\checkpoints_sft\quillan_fluent_aligned.pt"
if not os.path.exists(ckpt_path):
    ckpt_path = r"C:\02_QUILLAN\checkpoints\checkpoints_sft\quillan_final_best.pt"

print(f"[*] Loading model state from: {ckpt_path}")
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
sd = ckpt.get("model_state_dict", ckpt)
msd = model.state_dict()
copied = 0
for k, v in sd.items():
    if k in msd and v.shape == msd[k].shape:
        msd[k].copy_(v)
        copied += 1
model.load_state_dict(msd)
print(f"[+] Loaded {copied}/{len(msd)} parameter layers.")

data_path = r"C:\02_QUILLAN\training_data\long_form_sft_dataset.pt"
samples = torch.load(data_path, map_location="cpu", weights_only=False)
print(f"[DATA] Loaded {len(samples)} long-form target-masked SFT samples.")

# Hyperparameters
STEPS = 60
LR = 1.5e-5
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)

print(f"\n[TRAIN] Running {STEPS} Long-Form Target-Masked SFT Steps (LR={LR})...")

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
    
    # Router Entropy Regularization
    entropy_penalty = torch.tensor(0.0)
    if hasattr(model.moe, '_last_probs') and model.moe._last_probs is not None:
        p = model.moe._last_probs
        entropy = -torch.sum(p * torch.log(p + 1e-8), dim=-1).mean()
        entropy_penalty = -0.01 * entropy

    total_loss = ce_loss + aux_loss + entropy_penalty
    total_loss.backward()
    
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    
    if total_loss.item() < best_loss:
        best_loss = total_loss.item()
        
    if step % 5 == 0 or step == 1:
        elapsed = time.time() - t0
        sps = elapsed / step
        print(f"  step {step:2d}/{STEPS}  loss={total_loss.item():.4f}  best={best_loss:.4f}  ({sps:.1f}s/st)", flush=True)

save_path = r"C:\02_QUILLAN\checkpoints\checkpoints_sft\quillan_longform_best.pt"
torch.save({
    'model_state_dict': model.state_dict(),
    'step': STEPS,
    'loss': best_loss,
    'version': 'quillan-v5.3.1-longform-best'
}, save_path)
print(f"\n[SAVE] Long-form best checkpoint saved to: {save_path}")

# Test Long-Form Speech Generation
model.eval()

def generate_speech(prompt, max_tokens=150, temp=0.7, top_p=0.9, rep_penalty=1.3):
    tokens = enc.encode(prompt)
    generated = list(tokens)
    
    print(f"\n==================================================")
    print(f"PROMPT:\n{prompt.strip()}")
    print("==================================================")
    print("RESPONSE:\n", end="", flush=True)
    
    with torch.no_grad():
        for _ in range(max_tokens):
            inp = torch.tensor([generated[-128:]], dtype=torch.long)
            logits = model(inp)[:, -1, :].clone()
            
            for tid in set(generated[-32:]):
                if logits[0, tid] < 0:
                    logits[0, tid] *= rep_penalty
                else:
                    logits[0, tid] /= rep_penalty

            if generated[-1] == 220:
                logits[0, 220] = float('-inf')

            if temp > 0:
                logits = logits / temp
                probs = F.softmax(logits, dim=-1)
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                cum_probs = torch.cumsum(sorted_probs, dim=-1)
                sorted_indices_to_remove = cum_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float('-inf')
                probs = F.softmax(logits, dim=-1)
                next_tok = torch.multinomial(probs, 1).item()
            else:
                next_tok = torch.argmax(logits, dim=-1).item()

            generated.append(next_tok)
            word_bytes = enc.decode_bytes([next_tok])
            word_str = word_bytes.decode('utf-8', errors='ignore')
            print(word_str, end="", flush=True)

            if next_tok == 50256:
                break
    print("\n" + "-" * 50)

print("\n=== FINAL LONG-FORM MODEL SPEECH DEMO ===")
prompts = [
    "<|user|>\nPlease introduce yourself in detail. Explain your architecture, your 34 Council Experts, your 9-Vector cognitive decomposition, and how you assist developers.\n<|assistant|>\n",
    "<|user|>\nWrite a Python function to implement quicksort with step-by-step explanations.\n<|assistant|>\n"
]

for p in prompts:
    generate_speech(p, max_tokens=150, temp=0.7, top_p=0.9, rep_penalty=1.3)
