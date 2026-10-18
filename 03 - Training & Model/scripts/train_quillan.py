import torch
import json
import os
import time
import signal
import sys
from pathlib import Path
from safetensors.torch import load_file, save_file

BASE = Path(r"C:\Users\Admin\Quillan-Ronin")
MODEL_PATH = BASE / "Quillan-v4.2-model" / "quillan_custom" / "model.safetensors"
DATA_DIR = BASE / "training_data"
CKPT_DIR = BASE / "quillan_checkpoints"
CKPT_DIR.mkdir(exist_ok=True)

# Emergency save handler - saves on crash/SIGINT
emergency_save_flag = False
def emergency_save(signum, frame):
    global emergency_save_flag
    emergency_save_flag = True
    print("\n[EMERGENCY] Signal received, saving checkpoint...")
    save_checkpoint(weights, "emergency")
    sys.exit(0)

signal.signal(signal.SIGINT, emergency_save)
signal.signal(signal.SIGTERM, emergency_save)

def save_checkpoint(state, name):
    path = CKPT_DIR / f"quillan_{name}.safetensors"
    # Save as float32 to avoid bf16 corruption issues
    clean = {}
    for k, v in state.items():
        clean[k] = v.detach().cpu().float() if v.dtype != torch.uint8 else v.cpu()
    save_file(clean, str(path))
    # Verify file is valid
    verify = load_file(str(path))
    assert len(verify) == len(state), f"Checkpoint verification failed! {len(verify)} != {len(state)}"
    print(f"  [SAVED] {path.name} ({os.path.getsize(path)/1024/1024:.0f} MB, verified)")
    return str(path)

# Load model
print("Loading model...")
weights = load_file(str(MODEL_PATH))
print(f"Loaded {len(weights)} tensors")

# Identify trainable (float) vs frozen (ternary uint8)
trainable_keys = [k for k, v in weights.items() if v.dtype != torch.uint8]
frozen_keys = [k for k, v in weights.items() if v.dtype == torch.uint8]
print(f"Trainable: {len(trainable_keys)} tensors")
print(f"Frozen: {len(frozen_keys)} tensors")

# Convert trainable to float32 for training stability
for k in trainable_keys:
    weights[k] = weights[k].float().requires_grad_(True)

# Simple AdamW on trainable tensors only
optimizer = torch.optim.AdamW(
    [weights[k] for k in trainable_keys],
    lr=5e-5,
    weight_decay=0.01
)

# Load training data
print("Loading training data...")
samples = []
for fname in ["GPT_5.5_Distilled.jsonl", "instruct_train.jsonl", "code_train.jsonl"]:
    fpath = DATA_DIR / fname
    if fpath.exists():
        with open(fpath, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                try:
                    s = json.loads(line.strip())
                    text = s.get('text', s.get('content', s.get('instruction', '')))
                    if len(text) > 50:
                        samples.append(text)
                except:
                    pass
        print(f"  {fname}: loaded")

print(f"Total samples: {len(samples)}")

# Simple token-level loss on embeddings
# We train by predicting next-token shifts in the embedding space
def compute_loss(weights, text_sample, seq_len=256):
    emb = weights['model.embed_tokens.weight']
    # Hash text to token indices (simplified - maps chars to vocab range)
    tokens = []
    for ch in text_sample[:seq_len]:
        tokens.append(ord(ch) % emb.shape[0])
    if len(tokens) < 2:
        return torch.tensor(0.0, requires_grad=True)
    
    tok_tensor = torch.tensor(tokens, dtype=torch.long)
    embeddings = emb[tok_tensor]  # [seq_len, hidden]
    
    # Simple next-token prediction loss in embedding space
    input_emb = embeddings[:-1]   # [seq_len-1, hidden]
    target_emb = embeddings[1:]   # [seq_len-1, hidden]
    
    loss = torch.nn.functional.mse_loss(input_emb, target_emb)
    return loss

# Training loop
print("\nStarting training...")
EPOCHS = 3
BATCH_SIZE = 1
ACCUM_STEPS = 8
log_interval = 50
save_interval = 200
best_loss = float('inf')

global_step = 0
for epoch in range(EPOCHS):
    epoch_loss = 0
    num_batches = 0
    
    # Shuffle samples
    import random
    random.shuffle(samples)
    
    for i in range(0, len(samples), BATCH_SIZE):
        if emergency_save_flag:
            break
            
        batch = samples[i:i+BATCH_SIZE]
        text = batch[0] if batch else ""
        
        loss = compute_loss(weights, text)
        loss = loss / ACCUM_STEPS
        loss.backward()
        
        epoch_loss += loss.item() * ACCUM_STEPS
        num_batches += 1
        
        if (num_batches % ACCUM_STEPS) == 0:
            torch.nn.utils.clip_grad_norm_(
                [weights[k] for k in trainable_keys], 1.0
            )
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1
            
            if global_step % log_interval == 0:
                avg = epoch_loss / num_batches
                print(f"  Epoch {epoch+1} Step {global_step}: loss={avg:.6f}")
                
                if avg < best_loss:
                    best_loss = avg
        
        if global_step > 0 and global_step % save_interval == 0:
            save_checkpoint(weights, f"step_{global_step}")
    
    # End of epoch save
    avg_loss = epoch_loss / max(num_batches, 1)
    print(f"\nEpoch {epoch+1} complete: avg_loss={avg_loss:.6f}")
    save_checkpoint(weights, f"epoch_{epoch+1}")

# Final save
save_checkpoint(weights, "final")
print("\nTraining complete!")
print(f"Best loss: {best_loss:.6f}")
print(f"Checkpoints saved to: {CKPT_DIR}")
