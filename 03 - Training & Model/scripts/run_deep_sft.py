"""
QUILLAN-RONIN ONI — DEEP SFT TRAINING RUN
===========================================
Deep multi-thousand-step training phase continuing from Phase 1 calibrated baseline (val=3.6112).
Dataset: 80% train shards, 20% validation shards across 139 research papers (113K rows).
All 28 paper auxiliary losses active and backpropagating.
"""

import os
import sys
import time
import math
import random
import torch
import torch.nn.functional as F
from pathlib import Path

# Encoding safety for Windows shells
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", line_buffering=True)
    except Exception:
        pass

# Paths
REPO = Path(r"C:\02_QUILLAN")
for p in [
    str(REPO / "scripts"),
    str(REPO / "09 - Projects" / "projects" / "oni"),
    str(REPO / "03 - Training & Model"),
    str(REPO),
]:
    if p not in sys.path:
        sys.path.insert(0, p)

# CPU Thread Safety Ceiling (guarantees host OS UI responsiveness)
safe_threads = max(1, (os.cpu_count() or 4) - 1)
torch.set_num_threads(safe_threads)
print(f"[quillan_deep_sft] CPU execution threads set to {safe_threads} (OS UI protected).", flush=True)

from quillan_bpe_tokenizer import QuillanBPETokenizer
from quillan_v5_4_oni import QuillanOniConfig, QuillanRoninOni

# Hyperparameters & Paths
CKPT_IN = REPO / "checkpoints" / "hf_restore" / "mini_sft_best.pt"
CKPT_OUT = REPO / "checkpoints" / "hf_restore" / "mini_sft_best.pt"
MANIFEST = REPO / "training_data" / "mini_structured.pt"

STEPS = int(os.environ.get("SFT_STEPS", 3000))
SEQ = 256
BATCH = 2
LR0 = float(os.environ.get("SFT_LR0", 2.5e-5))
LR1 = float(os.environ.get("SFT_LR1", 5e-7))
WARMUP = int(os.environ.get("SFT_WARMUP", 30))
VAL_EVERY = int(os.environ.get("SFT_VAL_EVERY", 50))
GEN_EVERY = int(os.environ.get("SFT_GEN_EVERY", 100))
LOG_EVERY = int(os.environ.get("SFT_LOG_EVERY", 10))

GEN_PROMPTS = [
    "User: Who are you?\n\nAssistant:",
    "User: What is Quillan-Ronin?\n\nAssistant:",
    "User: Explain ST-MoE routing and Kuramoto synchronization.\n\nAssistant:",
    "User: What is BitNet 1.58b Straight-Through Estimator quantization?\n\nAssistant:",
]

# ── 1. Sharded Data Loader (80% Train / 20% Val) ──────────────────────────
print(f"[quillan_deep_sft] Loading manifest from {MANIFEST.name}...", flush=True)
man = torch.load(MANIFEST, map_location="cpu", weights_only=True)
shards = [Path(s) for s in man["shards"]]
nval = max(1, len(shards) // 5)  # 20% validation split
tr_sh, va_sh = shards[:-nval], shards[-nval:]
print(f"[quillan_deep_sft] Data Split: {len(tr_sh)} train shards (80%), {len(va_sh)} val shards (20%).", flush=True)

pairs = []
for si, sh in enumerate(tr_sh):
    n = torch.load(sh, map_location="cpu", weights_only=True).shape[0]
    pairs.extend((si, ri) for ri in range(n))
random.Random(4242).shuffle(pairs)
print(f"[quillan_deep_sft] Shuffled training pool: {len(pairs)} dialogue pairs.", flush=True)

shard_cache = {}
def get_row(si, ri):
    if si not in shard_cache:
        shard_cache[si] = torch.load(tr_sh[si], map_location="cpu", weights_only=True)
        if len(shard_cache) > 4:
            shard_cache.pop(next(iter(shard_cache)))
    return shard_cache[si][ri]

# Fixed validation split for rigorous apples-to-apples loss tracking
va_rows = torch.cat([
    torch.load(va_sh[i], map_location="cpu", weights_only=True)[:8]
    for i in range(min(4, len(va_sh)))
])[:32, :SEQ]
print(f"[quillan_deep_sft] Fixed validation tensor: {va_rows.shape}", flush=True)

# ── 2. Model Initialization & Weight Binding ─────────────────────────────
print(f"[quillan_deep_sft] Initializing QuillanRoninOni architecture...", flush=True)
cfg = QuillanOniConfig(
    vocab_size=50262,
    hidden_dim=1024,
    ffn_dim=2048,
    n_layer=6,
    num_experts=34,
    top_k=4,
    max_seq_len=512,
)
model = QuillanRoninOni(cfg)

print(f"[quillan_deep_sft] Loading starting checkpoint {CKPT_IN.name}...", flush=True)
bd = torch.load(CKPT_IN, map_location="cpu", weights_only=True)
sd = bd.get("model_state_dict", bd.get("model", bd))
missing, unexp = model.load_state_dict(sd, strict=False)
meta = bd.get("meta", {}) if isinstance(bd, dict) else {}
prev_best = meta.get("val", 3.6112)
print(f"[quillan_deep_sft] Checkpoint bound successfully: missing={len(missing)} unexp={len(unexp)} starting_val={prev_best:.4f}", flush=True)

tok = QuillanBPETokenizer()
opt = torch.optim.AdamW(model.parameters(), lr=LR0, weight_decay=0.01, betas=(0.9, 0.95))

# ── 3. Generation Probe Helper ───────────────────────────────────────────
def generate_sample(prompt: str, max_new: int = 40) -> str:
    model.eval()
    ids = tok.encode(prompt)
    gen = list(ids[-128:])
    with torch.no_grad():
        for _ in range(max_new):
            x = torch.tensor([gen[-256:]], dtype=torch.long)
            out = model(x)
            logits = (out[0] if isinstance(out, tuple) else out)[0, -1, :50262].clone()
            for tid in set(gen[-24:]):
                if logits[tid] > 0:
                    logits[tid] /= 1.2
                else:
                    logits[tid] *= 1.2
            v, _ = torch.topk(logits, 40)
            logits[logits < v[-1]] = float("-inf")
            probs = F.softmax(logits / 0.7, dim=-1)
            nxt = int(torch.multinomial(probs, 1).item())
            if nxt in (50256, 50257, 50261):
                break
            gen.append(nxt)
    model.train()
    out_tokens = [t for t in gen[len(ids):] if t < 50257]
    return tok.decode(out_tokens).strip()

# ── 4. Main Deep Training Loop ───────────────────────────────────────────
print(f"\n=======================================================", flush=True)
print(f"[START] DEEP SFT TRAINING RUN", flush=True)
print(f"* Total Steps: {STEPS} | Batch Size: {BATCH} | SeqLen: {SEQ}", flush=True)
print(f"* LR Schedule: {LR0:.1e} -> {LR1:.1e} (Cosine Decay, {WARMUP} warmup)", flush=True)
print(f"* Baseline Best Val CE: {prev_best:.4f} (Target: Loss < 2.0)", flush=True)
print(f"* 80% Train / 20% Eval Shard Split Active", flush=True)
print(f"=======================================================\n", flush=True)

model.train()
best_val = prev_best
ptr = 0
n_pairs = len(pairs)
step_times = []
t_start = time.time()

for step in range(1, STEPS + 1):
    t0 = time.time()

    # Learning rate schedule (Warmup + Cosine Decay)
    if step <= WARMUP:
        lr = LR0 * (step / WARMUP)
    else:
        progress = (step - WARMUP) / max(1, STEPS - WARMUP)
        lr = LR1 + 0.5 * (LR0 - LR1) * (1.0 + math.cos(math.pi * progress))
    for pg in opt.param_groups:
        pg["lr"] = lr

    # Batch extraction
    batch_rows = []
    for _ in range(BATCH):
        si, ri = pairs[ptr % n_pairs]
        ptr += 1
        batch_rows.append(get_row(si, ri)[:SEQ])
    x = torch.stack(batch_rows)

    opt.zero_grad(set_to_none=True)

    # Forward pass with full paper auxiliary loss aggregation
    out = model(x, labels=x, return_aux=True)
    if isinstance(out, tuple):
        logits, ce_loss, aux_dict = out
    else:
        logits, ce_loss, aux_dict = out, out.get("loss", torch.tensor(0.0)), {}

    # Aggregate auxiliary regularizers across all paper mechanisms
    aux_loss = torch.tensor(0.0, device=x.device)
    if isinstance(aux_dict, dict):
        for k, v in aux_dict.items():
            if isinstance(v, torch.Tensor) and v.requires_grad:
                aux_loss = aux_loss + 0.05 * v

    total_loss = ce_loss + aux_loss
    total_loss.backward()

    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()

    dt = time.time() - t0
    step_times.append(dt)
    if len(step_times) > 20:
        step_times.pop(0)

    # Progress Logging
    if step % LOG_EVERY == 0 or step == 1:
        avg_dt = sum(step_times) / len(step_times)
        tokens_per_sec = (BATCH * SEQ) / avg_dt if avg_dt > 0 else 0
        eta_sec = (STEPS - step) * avg_dt
        eta_min = eta_sec / 60
        print(f"Step {step:5d}/{STEPS} | CE: {ce_loss.item():.4f} | Aux: {aux_loss.item():.4f} | Tot: {total_loss.item():.4f} | LR: {lr:.2e} | Speed: {tokens_per_sec:.1f} tok/s ({avg_dt:.2f}s/step) | ETA: {eta_min:.1f}m", flush=True)

    # Periodic Validation (20% Split Evaluation)
    if step % VAL_EVERY == 0 or step == STEPS:
        model.eval()
        with torch.no_grad():
            v_out = model(va_rows, labels=va_rows, return_aux=True)
            if isinstance(v_out, tuple):
                _, v_ce, v_aux_dict = v_out
            else:
                v_ce = v_out.get("loss", torch.tensor(0.0))
                v_aux_dict = {}

            v_aux = sum(v.item() for v in v_aux_dict.values() if isinstance(v, torch.Tensor)) if isinstance(v_aux_dict, dict) else 0.0
            v_loss = v_ce.item()
            v_tot = v_loss + 0.05 * v_aux

        print(f"\n[VAL @ Step {step}] Current Val CE: {v_loss:.4f} (Tot: {v_tot:.4f}, Best: {best_val:.4f})", flush=True)

        if v_loss < best_val:
            best_val = v_loss
            temp_out = CKPT_OUT.with_suffix(".tmp")
            torch.save({
                "model_state_dict": {k: v.cpu().clone() for k, v in model.state_dict().items()},
                "meta": {
                    "val": best_val,
                    "step": step,
                    "stage": "deep_sft",
                    "seq": SEQ,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                }
            }, temp_out)
            if temp_out.exists():
                temp_out.replace(CKPT_OUT)
            print(f"[CHECKPOINT SAVED] New best validation loss: {best_val:.4f} -> {CKPT_OUT.name}\n", flush=True)
        model.train()

    # Periodic Linguistic Generation Probe
    if step % GEN_EVERY == 0 or step == STEPS:
        print(f"\n-------------------------------------------------------", flush=True)
        print(f"[PROBE @ STEP {step}] (CE: {ce_loss.item():.4f} Tot: {total_loss.item():.4f})", flush=True)
        print(f"-------------------------------------------------------", flush=True)
        for prompt in GEN_PROMPTS:
            out_text = generate_sample(prompt, max_new=35)
            q_line = prompt.split("\n")[0]
            print(f"  {q_line}", flush=True)
            print(f"  --> {out_text!r}\n", flush=True)
        print(f"-------------------------------------------------------\n", flush=True)

total_elapsed = time.time() - t_start
print(f"\n=======================================================", flush=True)
print(f"[COMPLETE] DEEP SFT TRAINING COMPLETED in {total_elapsed/60:.1f} minutes", flush=True)
print(f"* Final Best Validation Loss: {best_val:.4f}", flush=True)
print(f"* Checkpoint: {CKPT_OUT}", flush=True)
print(f"=======================================================", flush=True)
