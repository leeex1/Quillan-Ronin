"""
Quillan Mini SFT — Fast Convergence Run
========================================
Target: Loss < 2.5 (coherent output) within 2-3h on CPU.
Strategy:
  - Load best checkpoint (mini_sft_best.pt)
  - 500 steps, seq=256, batch=4, grad_accum=4 (=16 effective seqs/update)
  - Cosine LR 4e-5 -> 1e-6
  - BEST-ONLY save (never overwrites a better checkpoint)
  - Gen test every 50 steps so you can see quality improving live
  - 3-thread CPU ceiling
"""
import sys, time, random, math
from pathlib import Path
import torch
import torch.nn.functional as F

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

REPO = Path(r"C:\02_QUILLAN")
for p in [str(REPO / "scripts" / "hf_code_era"),
          str(REPO / "scripts" / "hf_code"),
          str(REPO / "scripts"),
          str(REPO / "09 - Projects" / "projects" / "oni"),
          str(REPO / "03 - Training & Model"),
          str(REPO)]:
    if p not in sys.path:
        sys.path.insert(0, p)

torch.set_num_threads(3)

from quillan_bpe_tokenizer import QuillanBPETokenizer
from quillan_v5_4_oni import QuillanOniConfig, QuillanRoninOni

# ── Config ─────────────────────────────────────────────────────────────────
CKPT_IN  = REPO / "checkpoints" / "hf_restore" / "mini_sft_best.pt"
CKPT_OUT = REPO / "checkpoints" / "hf_restore" / "mini_sft_best.pt"
MANIFEST = REPO / "training_data" / "mini_structured.pt"

STEPS     = 500
SEQ       = 256
LR0       = 4e-5
LR1       = 1e-6
BATCH     = 4
GRAD_ACC  = 4        # effective batch = 16 seqs per update
VAL_EVERY = 25
GEN_EVERY = 50
WARMUP    = 20       # steps

GEN_PROMPTS = [
    "User: Who are you?\n\nAssistant:",
    "User: What is 2 + 2?\n\nAssistant:",
    "User: Explain what a neural network is in one sentence.\n\nAssistant:",
]

# ── Load data ───────────────────────────────────────────────────────────────
print("Loading manifest...", flush=True)
man = torch.load(MANIFEST, map_location="cpu", weights_only=True)
shards = [Path(s) for s in man["shards"]]
nval = max(1, len(shards) // 5)
tr_sh, va_sh = shards[:-nval], shards[-nval:]
print(f"train_shards={len(tr_sh)} val_shards={len(va_sh)} total_rows={man['total']}", flush=True)

# Build global row index and shuffle
pairs = []
for si, sh in enumerate(tr_sh):
    n = torch.load(sh, map_location="cpu", weights_only=True).shape[0]
    pairs.extend((si, ri) for ri in range(n))
random.Random(7777).shuffle(pairs)
print(f"Shuffled index: {len(pairs)} pairs", flush=True)

shard_cache = {}
def get_row(si, ri):
    if si not in shard_cache:
        shard_cache[si] = torch.load(tr_sh[si], map_location="cpu", weights_only=True)
        if len(shard_cache) > 4:
            shard_cache.pop(next(iter(shard_cache)))
    return shard_cache[si][ri]

# Validation rows (fixed subset)
va_rows = torch.cat([
    torch.load(va_sh[i], map_location="cpu", weights_only=True)[:8]
    for i in range(min(3, len(va_sh)))
])[:24, :SEQ]

# ── Load model ──────────────────────────────────────────────────────────────
print("Loading model...", flush=True)
cfg = QuillanOniConfig(vocab_size=50262, hidden_dim=1024, ffn_dim=2048,
                       n_layer=6, num_experts=34, top_k=4, max_seq_len=512)
model = QuillanRoninOni(cfg)
bd = torch.load(CKPT_IN, map_location="cpu", weights_only=True)
missing, unexp = model.load_state_dict(bd.get("model_state_dict", bd), strict=False)
prev_best = bd.get("meta", {}).get("val", 1e9) if isinstance(bd, dict) else 1e9
print(f"Bound: missing={len(missing)} unexp={len(unexp)} prev_best_val={prev_best:.4f}", flush=True)

tok = QuillanBPETokenizer()
opt = torch.optim.AdamW(model.parameters(), lr=LR0, weight_decay=0.01, betas=(0.9, 0.95))

# ── Generation helper ────────────────────────────────────────────────────────
def generate(prompt, max_new=60, temp=0.7, top_k=50):
    model.eval()
    ids = tok.encode(prompt)
    gen = list(ids[-128:])
    with torch.no_grad():
        for _ in range(max_new):
            x = torch.tensor([gen[-256:]], dtype=torch.long)
            out = model(x)
            logits = (out[0] if isinstance(out, tuple) else out)[0, -1, :].clone()
            # Repetition penalty
            for tid in set(gen[-32:]):
                logits[tid] = logits[tid] / 1.3 if logits[tid] > 0 else logits[tid] * 1.3
            # Top-k filter
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[-1]] = float('-inf')
            probs = F.softmax(logits / temp, dim=-1)
            nxt = int(torch.multinomial(probs, 1).item())
            if nxt in (50256, 50257):  # EOS tokens
                break
            gen.append(nxt)
    model.train()
    return tok.decode(gen[len(ids):]).strip()

# ── Training loop ────────────────────────────────────────────────────────────
print(f"\nSTARTING: {STEPS} steps, seq={SEQ}, batch={BATCH}, grad_acc={GRAD_ACC}", flush=True)
print(f"Effective batch size: {BATCH * GRAD_ACC} seqs per update", flush=True)
print(f"LR: {LR0:.1e} -> {LR1:.1e} cosine with {WARMUP} step warmup\n", flush=True)

model.train()
best_val = prev_best
t0 = time.time()
opt.zero_grad()

for step in range(1, STEPS + 1):
    # LR schedule: linear warmup + cosine decay
    if step <= WARMUP:
        cur_lr = LR0 * step / WARMUP
    else:
        progress = (step - WARMUP) / max(1, STEPS - WARMUP)
        cur_lr = LR1 + 0.5 * (LR0 - LR1) * (1.0 + math.cos(math.pi * progress))
    for pg in opt.param_groups:
        pg["lr"] = cur_lr

    # Accumulate gradients
    step_loss = 0.0
    for acc in range(GRAD_ACC):
        global_idx = ((step - 1) * GRAD_ACC + acc) * BATCH
        batch_rows = [pairs[(global_idx + k) % len(pairs)] for k in range(BATCH)]
        x = torch.stack([get_row(si, ri) for si, ri in batch_rows])[:, :SEQ]
        out = model(x)
        logits = out[0] if isinstance(out, tuple) else out
        loss = F.cross_entropy(
            logits[:, :-1, :].reshape(-1, logits.size(-1)),
            x[:, 1:].reshape(-1)
        ) / GRAD_ACC
        loss.backward()
        step_loss += loss.item()

    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    opt.zero_grad()

    # ── Validation + logging ─────────────────────────────────────────────
    if step % VAL_EVERY == 0 or step == STEPS:
        model.eval()
        with torch.no_grad():
            vo = model(va_rows)
            vl = vo[0] if isinstance(vo, tuple) else vo
            vv = F.cross_entropy(
                vl[:, :-1, :].reshape(-1, vl.size(-1)),
                va_rows[:, 1:].reshape(-1)
            ).item()
        elapsed = time.time() - t0
        tps = (step * BATCH * GRAD_ACC * SEQ) / elapsed
        print(f"step={step:4d}/{STEPS} | train={step_loss:.4f} | val={vv:.4f} | "
              f"lr={cur_lr:.2e} | {tps:.0f} tok/s | {elapsed/60:.1f}min", flush=True)

        if vv < best_val:
            best_val = vv
            torch.save({
                "model_state_dict": {k: v.cpu().clone() for k, v in model.state_dict().items()},
                "meta": {"val": best_val, "step": step, "stage": "fast_sft", "seq": SEQ}
            }, CKPT_OUT)
            print(f"  >> SAVED new best val={best_val:.4f}", flush=True)
        model.train()

    # ── Generation test ──────────────────────────────────────────────────
    if step % GEN_EVERY == 0:
        print(f"\n--- Generation test @ step {step} ---", flush=True)
        for prompt in GEN_PROMPTS:
            out_text = generate(prompt, max_new=50)
            print(f"  Q: {prompt.split(chr(10))[0]}", flush=True)
            print(f"  A: {out_text[:150]!r}", flush=True)
        print("---\n", flush=True)

elapsed = time.time() - t0
print(f"\nDONE. best_val={best_val:.4f} ({elapsed/3600:.1f}h total)", flush=True)
print(f"Checkpoint: {CKPT_OUT}", flush=True)
