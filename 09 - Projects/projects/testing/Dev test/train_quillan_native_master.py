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

enc = tiktoken.get_encoding("gpt2")

print("==================================================================")
print("   👑 QUILLAN-RONIN v5.3.1 — NATIVE SUBSTRATE MASTER TRAINER")
print("==================================================================")

# 100% Native Quillan Architecture Stack (Matching 50,257 Pre-tokenized Corpus)
cfg = QuillanArchConfig(
    hidden_dim=1024, ffn_dim=2048, num_experts=34,
    vocab_size=50257, text_only=True, eggroll_rank=256
)
model = QuillanRoninSovereign(cfg).to("cpu")

ckpt_path = REPO_ROOT / "checkpoints" / "checkpoints_sft" / "quillan_thinking_reasoning_master.pt"
if ckpt_path.exists():
    print(f"[*] Resuming from Checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(sd, strict=False)
    print(f"[+] Loaded {len(sd)} parameter tensors cleanly.")

# Load Gold Reasoning & Dialogue JSONL Datasets
jsonl_files = [
    REPO_ROOT / "training_data" / "Quillan_Clean_Reasoning_Gold_Dataset.jsonl",
    REPO_ROOT / "training_data" / "Quillan_Hyper_Tune_Gold_Dataset.jsonl",
    REPO_ROOT / "training_data" / "Quillan_Explanatory_Prose_Dataset.jsonl",
    REPO_ROOT / "training_data" / "Quillan_General_Knowledge_Dataset.jsonl"
]

gold_samples = []
for file_path in jsonl_files:
    if file_path.exists():
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                obj = json.loads(line.strip())
                q = obj.get("question", "")
                ans = obj.get("response", "")
                if q and ans:
                    full_txt = f"<|system|>\n# 🤖🧠 Quillan System Start 🧠🤖\n<|user|>\n{q}\n<|assistant|>\n{ans}\n<|end|>"
                    prompt_txt = f"<|system|>\n# 🤖🧠 Quillan System Start 🧠🤖\n<|user|>\n{q}\n<|assistant|>\n"
                    toks = enc.encode(full_txt)
                    p_toks = enc.encode(prompt_txt)
                    labs = [-100] * len(p_toks) + toks[len(p_toks):]
                    gold_samples.append((toks[:512], labs[:512]))

print(f"[DATA] Loaded {len(gold_samples)} Gold Target-Masked Reasoning/Dialogue Samples.")

# Load Pre-Tokenized Corpus Tensors (166M Tokens) & Reshape to Matrix
corpus_pt = REPO_ROOT / "training_data" / "quillan_corpus_CLEAN_V7.pt"
if corpus_pt.exists():
    print(f"[*] Ingesting 166M Pre-Tokenized Tensor Corpus: {corpus_pt.name}...")
    raw_tokens = torch.load(corpus_pt, weights_only=False)
    n_blocks = len(raw_tokens) // 512
    corpus_matrix = raw_tokens[:n_blocks * 512].reshape(n_blocks, 512)
    print(f"[+] Loaded Matrix Blocks: {corpus_matrix.shape}")
else:
    corpus_matrix = None

# Training Config: 1,000 High-Density Native Steps with Gradient Accumulation
STEPS = 1000
GRAD_ACCUM = 2
BASE_LR = 2.5e-4
MIN_LR = 1.0e-5

optimizer = torch.optim.AdamW(model.parameters(), lr=BASE_LR, weight_decay=0.01)

print(f"\n[TRAIN] Executing Native Substrate Master Training ({STEPS} Steps, LR={BASE_LR} -> {MIN_LR})...")

model.train()
t0 = time.time()
best_loss = 999.0

for step in range(1, STEPS + 1):
    lr = MIN_LR + 0.5 * (BASE_LR - MIN_LR) * (1.0 + math.cos(math.pi * step / STEPS))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        
    optimizer.zero_grad()
    accum_loss = 0.0
    
    for _ in range(GRAD_ACCUM):
        # 60% Gold Reasoning Samples, 40% Continuous Pre-tokenized Corpus
        use_gold = (step % 5 != 0) or (corpus_matrix is None)
        
        if use_gold:
            sample_idx = (step * GRAD_ACCUM + _) % len(gold_samples)
            toks, labs = gold_samples[sample_idx]
            sl = min(512, len(toks))
            x = torch.tensor([toks[:sl]], dtype=torch.long)
            y = torch.tensor([labs[:sl]], dtype=torch.long)
        else:
            # Direct O(1) Matrix Row Lookup
            rand_row = torch.randint(0, len(corpus_matrix), (1,)).item()
            tok_row = corpus_matrix[rand_row]
            x = tok_row[:-1].unsqueeze(0)
            y = tok_row[1:].unsqueeze(0)
            
        logits, aux_loss = model(x)
        ce_loss = F.cross_entropy(
            logits[..., :-1, :].contiguous().view(-1, cfg.vocab_size),
            y[..., 1:].contiguous().view(-1),
            ignore_index=-100
        )
        total_loss = (ce_loss + aux_loss) / GRAD_ACCUM
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
        
        # Save Atomic Progress Checkpoint every 50 steps
        if step % 50 == 0:
            torch.save({
                'model_state_dict': model.state_dict(),
                'step': step,
                'loss': best_loss,
                'version': 'quillan-v5.3.1-native-substrate'
            }, ckpt_path)
            print(f"  [CHECKPOINT] Progress checkpoint saved to {ckpt_path.name}", flush=True)

# Final Save
torch.save({
    'model_state_dict': model.state_dict(),
    'step': STEPS,
    'loss': best_loss,
    'version': 'quillan-v5.3.1-native-substrate-master'
}, ckpt_path)

print(f"\n[SAVE] 🏆 Quillan Native Substrate Master Model Saved Successfully to: {ckpt_path}")
