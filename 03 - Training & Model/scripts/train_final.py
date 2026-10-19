#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QUILLAN-RONIN v5.3.1 — FINAL TRAINING RUN
==========================================
Loads ALL available datasets, handles every format correctly.
No caps. 100% data usage. Dialogue-first priority.
"""

import os, sys, time, math, json, random
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "6"
os.environ["MKL_NUM_THREADS"] = "6"
os.environ["OPENBLAS_NUM_THREADS"] = "6"

import torch
torch.set_num_threads(6)
torch.set_num_interop_threads(6)


try:
    import psutil
    p = psutil.Process(os.getpid())
    p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
except Exception:
    pass

import torch.nn.functional as F
import tiktoken

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

REPO_ROOT = r"C:\02_QUILLAN"
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from _dev.quillan_v8_saturated import QuillanRoninSovereign, QuillanArchConfig

# ─── CONFIG ──────────────────────────────────────────────────────────────────
TOTAL_STEPS  = 3000
LR           = 1e-5
MIN_LR       = 5e-7
WARMUP_STEPS = 150
GRAD_ACCUM   = 2
SEQ_LEN      = 512
STRIDE       = 512       # Non-overlapping chunks for clean speed
SAVE_EVERY   = 100
SAMPLE_EVERY = 100       # Frequent live speech verification
LOG_EVERY    = 10


DATA_DIR = os.path.join(REPO_ROOT, "training_data")
CKPT_DIR = os.path.join(REPO_ROOT, "checkpoints", "checkpoints_sft")

BASE_CKPT   = os.path.join(CKPT_DIR, "quillan_sft_v3_best.pt")
BEST_CKPT   = os.path.join(CKPT_DIR, "quillan_final_best.pt")
LATEST_CKPT = os.path.join(CKPT_DIR, "quillan_final_latest.pt")

# Priority: dialogue first, then quillan-specific, then instruct, then knowledge/science/code
# .pt files are flat token streams — loaded as overlapping SEQ_LEN chunks
# .jsonl files are parsed line-by-line — all formats handled
DATASET_ORDER = [
    # ── DIALOGUE (highest priority) ──
    "GPT_5.5_Distilled.pt",
    "GPT_5.5_Distilled.jsonl",
    # ── QUILLAN SPECIFIC ──
    "quillan_12mb_training_dataset.pt",
    "quillan_12mb_training_dataset.jsonl",
    "Quillan_Ronin_v5.3.1_Samurai_Training_Seed_Dataset.jsonl",
    # ── INSTRUCT / CONVERSATION ──
    "instruct_train.pt",
    "instruct_train.jsonl",
    "full_train.pt",
    "full_train.jsonl",
    "full_dataset.pt",
    "full_dataset.jsonl",
    "train.pt",
    "train.jsonl",
    # ── CODE ──
    "code_train.pt",
    "code_train.jsonl",
    # ── KNOWLEDGE ──
    "quillan_corpus_CLEAN_V7.pt",
    "quillan_corpus_CLEAN_V7.jsonl",
    "quillan_tokenized.pt",
    # ── SCIENCE ──
    "quillan_science_absolute.pt",
    "quillan_science_absolute.jsonl",
    "quillan_science_additional.pt",
    "quillan_science_additional.jsonl",
    "pdf_papers_corpus.jsonl",
    # ── RAW TOKENIZED (last — already partially covered above) ──
    "unified_tokenized_corpus.jsonl",
]

# ─── LOADERS ─────────────────────────────────────────────────────────────────

def load_pt(fpath):
    """Load a .pt file — handles flat Tensors, dicts, and lists."""
    data = torch.load(fpath, map_location="cpu", weights_only=False)
    samples = []

    # Case 1: flat 1D tensor (entire corpus concatenated)
    if isinstance(data, torch.Tensor) and data.dim() == 1:
        ids = data.tolist()
        for i in range(0, len(ids) - SEQ_LEN, STRIDE):
            chunk = ids[i:i + SEQ_LEN]
            if len(chunk) >= 32:
                samples.append((chunk, chunk))
        return samples

    # Case 2: 2D tensor (batch of sequences)
    if isinstance(data, torch.Tensor) and data.dim() == 2:
        for row in data:
            t = row.tolist()
            if len(t) >= 32:
                samples.append((t, t))
        return samples

    # Case 3: dict with input_ids key
    if isinstance(data, dict):
        ids = data.get("input_ids", data.get("tokens", None))
        labels = data.get("labels", data.get("target_ids", ids))
        if ids is not None:
            if isinstance(ids, torch.Tensor):
                if ids.dim() == 1:
                    ids_l = ids.tolist()
                    lbl_l = labels.tolist() if isinstance(labels, torch.Tensor) and labels.dim() == 1 else ids_l
                    for i in range(0, len(ids_l) - SEQ_LEN, STRIDE):
                        c = ids_l[i:i+SEQ_LEN]
                        l = lbl_l[i:i+SEQ_LEN]
                        if len(c) >= 32: samples.append((c, l))
                elif ids.dim() == 2:
                    for j in range(ids.size(0)):
                        c = ids[j].tolist()
                        l = labels[j].tolist() if isinstance(labels, torch.Tensor) and labels.dim() == 2 else c
                        if len(c) >= 32: samples.append((c, l))
            elif isinstance(ids, list):
                for j, t in enumerate(ids):
                    if isinstance(t, torch.Tensor): t = t.tolist()
                    l = labels[j].tolist() if isinstance(labels, list) and j < len(labels) else t
                    if len(t) >= 32: samples.append((t, l))
        return samples

    # Case 4: list of dicts or tensors
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                t = item.get("input_ids", item.get("tokens", []))
                l = item.get("labels", t)
                if isinstance(t, torch.Tensor): t = t.tolist()
                if isinstance(l, torch.Tensor): l = l.tolist()
                if len(t) >= 32: samples.append((t, l))
            elif isinstance(item, torch.Tensor):
                t = item.tolist()
                if len(t) >= 32: samples.append((t, t))
        return samples

    return samples


def load_jsonl(fpath, enc):
    """Load a .jsonl file — handles all formats found in training_data."""
    samples = []
    with open(fpath, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                obj = json.loads(line)

                # Format 1: pre-tokenized
                toks = obj.get("input_ids", obj.get("tokens", []))
                if toks:
                    labs = obj.get("labels", obj.get("target_ids", toks))
                    if isinstance(toks, torch.Tensor): toks = toks.tolist()
                    if isinstance(labs, torch.Tensor): labs = labs.tolist()
                    if len(toks) >= 32: samples.append((toks, labs))
                    continue

                text = ""

                # Format 2: plain text (GPT_5.5_Distilled)
                if "text" in obj:
                    text = obj["text"]

                # Format 3: messages array (instruct_train, full_train)
                elif "messages" in obj:
                    msgs = obj["messages"]
                    if isinstance(msgs, str):
                        try: msgs = json.loads(msgs.replace("'", '"'))
                        except: msgs = []
                    parts = []
                    if isinstance(msgs, list):
                        for m in msgs:
                            if not isinstance(m, dict): continue
                            role    = m.get("role", "")
                            content = m.get("content", "")
                            if role == "system":    parts.append(f"<|system|>\n{content}")
                            elif role == "user":    parts.append(f"<|user|>\n{content}")
                            elif role == "assistant": parts.append(f"<|assistant|>\n{content}")
                    text = "\n".join(parts)

                # Format 4: question/final_output (quillan_12mb)
                elif "question" in obj and "final_output" in obj:
                    q = obj["question"]
                    a = obj["final_output"]
                    r = obj.get("reasoning_trace", "")
                    if r:
                        text = f"<|user|>\n{q}\n<|assistant|>\n<think>\n{r}\n</think>\n{a}"
                    else:
                        text = f"<|user|>\n{q}\n<|assistant|>\n{a}"

                # Format 5: prompt/response or instruction/output
                elif "prompt" in obj or "instruction" in obj:
                    p = obj.get("prompt", obj.get("instruction", ""))
                    r = obj.get("response", obj.get("output", obj.get("completion", "")))
                    t = obj.get("thought", obj.get("reasoning", ""))
                    if t:
                        text = f"<|user|>\n{p}\n<|assistant|>\n<think>\n{t}\n</think>\n{r}"
                    elif r:
                        text = f"<|user|>\n{p}\n<|assistant|>\n{r}"
                    else:
                        text = p

                # Format 6: content field
                elif "content" in obj:
                    text = obj["content"]

                if text and len(text) > 20:
                    toks = enc.encode(text)[:SEQ_LEN]
                    if len(toks) >= 32:
                        samples.append((toks, toks))

            except Exception:
                pass
    return samples


def load_all_datasets():
    enc = tiktoken.get_encoding("gpt2")
    all_samples = []
    loaded_bases = set()

    for fname in DATASET_ORDER:
        fpath = os.path.join(DATA_DIR, fname)
        if not os.path.exists(fpath):
            continue

        base = fname.replace(".pt", "").replace(".jsonl", "")
        if base in loaded_bases:
            print(f"[DATA] SKIP (already loaded): {fname}", flush=True)
            continue

        try:
            if fname.endswith(".pt"):
                samp = load_pt(fpath)
            else:
                samp = load_jsonl(fpath, enc)

            if samp:
                all_samples.extend(samp)
                loaded_bases.add(base)
                sz = os.path.getsize(fpath) / 1e6
                print(f"[DATA] {fname:<55s} {len(samp):>8d} samples  ({sz:.0f}MB)", flush=True)
            else:
                print(f"[DATA] EMPTY: {fname}", flush=True)

        except Exception as e:
            print(f"[DATA] ERROR {fname}: {e}", flush=True)

    random.shuffle(all_samples)
    print(f"\n[DATA] TOTAL: {len(all_samples):,} samples ready", flush=True)
    return all_samples


# ─── GENERATION ──────────────────────────────────────────────────────────────
def generate(model, enc, prompt, max_tokens=60, temp=0.8, top_p=0.92):
    model.eval()
    tokens = enc.encode(prompt)
    with torch.no_grad():
        for _ in range(max_tokens):
            inp    = torch.tensor([tokens[-512:]], dtype=torch.long)
            logits = model(inp)[:, -1, :] / temp
            probs  = F.softmax(logits, dim=-1)
            s_p, s_i = torch.sort(probs, descending=True)
            cum  = torch.cumsum(s_p, dim=-1)
            mask = cum > top_p
            mask[..., 1:] = mask[..., :-1].clone(); mask[..., 0] = 0
            logits = logits.scatter(1, s_i,
                logits.gather(1, s_i).masked_fill(mask, float('-inf')))
            next_tok = torch.multinomial(F.softmax(logits, dim=-1), 1).item()
            if next_tok == 50256: break
            tokens.append(next_tok)
    return enc.decode(tokens[len(enc.encode(prompt)):]).replace('\ufffd', '?')


# ─── MAIN ────────────────────────────────────────────────────────────────────
def main():
    enc = tiktoken.get_encoding("gpt2")

    cfg = QuillanArchConfig(
        hidden_dim=1024, ffn_dim=2048, num_experts=34,
        text_only=True, eggroll_rank=256,
    )
    model = QuillanRoninSovereign(cfg)
    total_p = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[MODEL] {total_p:.1f}M parameters", flush=True)

    start_step = 1
    best_loss = 999.0

    # Auto-resume from latest checkpoint if available
    ckpt_to_load = LATEST_CKPT if os.path.exists(LATEST_CKPT) else BASE_CKPT
    if os.path.exists(ckpt_to_load):
        ckpt = torch.load(ckpt_to_load, map_location="cpu", weights_only=False)
        sd   = ckpt.get("model_state_dict", ckpt)
        msd  = model.state_dict()
        n    = 0
        for k, v in sd.items():
            if k in msd and v.shape == msd[k].shape:
                msd[k].copy_(v); n += 1
        model.load_state_dict(msd)
        start_step = ckpt.get("step", 0) + 1
        best_loss = ckpt.get("best_loss", 999.0)
        print(f"[RESUME] Loaded {n}/510 layers from {os.path.basename(ckpt_to_load)} | start_step={start_step} | best_loss={best_loss:.4f}", flush=True)

    samples = load_all_datasets()
    if not samples:
        print("[FATAL] No training data found!", flush=True); sys.exit(1)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)

    def get_lr(s):
        if s < WARMUP_STEPS: return LR * (s + 1) / WARMUP_STEPS
        p = (s - WARMUP_STEPS) / max(1, TOTAL_STEPS - WARMUP_STEPS)
        return MIN_LR + 0.5 * (LR - MIN_LR) * (1 + math.cos(math.pi * p))

    test_prompts = [
        ("<|user|>\nHello! Who are you?\n<|assistant|>\n",                      "Identity"),
        ("<|user|>\nWhat can you help me with?\n<|assistant|>\n",               "Capability"),
        ("<|user|>\nExplain quantum entanglement in simple terms.\n<|assistant|>\n", "Science"),
        ("<|user|>\nWrite a short poem about artificial intelligence.\n<|assistant|>\n", "Creative"),
    ]

    model.train()
    best_loss = 999.0
    idx = 0
    t0  = time.time()

    print(f"\n[TRAIN] Final Run  0->{TOTAL_STEPS} | {len(samples):,} samples | LR={LR:.1e} | warmup={WARMUP_STEPS}", flush=True)

    for step in range(start_step, TOTAL_STEPS + 1):

        lr = get_lr(step - 1)
        for pg in optimizer.param_groups: pg['lr'] = lr

        optimizer.zero_grad()
        acc_loss = 0.0

        for _ in range(GRAD_ACCUM):
            toks, labs = samples[idx % len(samples)]; idx += 1
            sl  = min(SEQ_LEN, len(toks))
            x   = torch.tensor([toks[:sl]], dtype=torch.long)
            y   = torch.tensor([labs[:sl]], dtype=torch.long)
            logits, aux_loss = model(x)
            ce_loss = F.cross_entropy(
                logits[..., :-1, :].contiguous().view(-1, cfg.vocab_size),
                y[..., 1:].contiguous().view(-1),
                ignore_index=-100
            )
            total_loss = ce_loss + aux_loss
            (total_loss / GRAD_ACCUM).backward()
            acc_loss += total_loss.item() / GRAD_ACCUM


        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step <= 10 or step % LOG_EVERY == 0:
            elapsed = time.time() - t0
            sps = elapsed / step
            eta = (TOTAL_STEPS - step) * sps / 3600
            print(f"  step {step:5d}/{TOTAL_STEPS}  loss={acc_loss:.4f}  lr={lr:.2e}  {sps:.1f}s/st  ETA:{eta:.1f}h", flush=True)

        if acc_loss < best_loss and step >= 25:
            best_loss = acc_loss
            os.makedirs(CKPT_DIR, exist_ok=True)
            torch.save({'model_state_dict': model.state_dict(), 'step': step,
                        'loss': acc_loss, 'best_loss': best_loss,
                        'version': 'quillan-v5.3.1-final'}, BEST_CKPT)
            print(f"  [BEST] step={step}  loss={acc_loss:.4f}", flush=True)

        if step % SAVE_EVERY == 0:
            os.makedirs(CKPT_DIR, exist_ok=True)
            torch.save({'model_state_dict': model.state_dict(), 'step': step,
                        'loss': acc_loss, 'best_loss': best_loss,
                        'version': 'quillan-v5.3.1-final'}, LATEST_CKPT)
            print(f"  [SAVE] step={step} -> quillan_final_latest.pt", flush=True)
            for prompt, label in test_prompts:
                out = generate(model, enc, prompt, max_tokens=60)
                print(f"  [{label}] {out[:120]}", flush=True)
            model.train()

    os.makedirs(CKPT_DIR, exist_ok=True)
    torch.save({'model_state_dict': model.state_dict(), 'step': TOTAL_STEPS,
                'loss': acc_loss, 'best_loss': best_loss,
                'version': 'quillan-v5.3.1-final'}, LATEST_CKPT)
    print(f"\n[DONE] Training complete! Best loss: {best_loss:.4f}", flush=True)
    print("\n=== FINAL GENERATION TEST ===", flush=True)
    for prompt, label in test_prompts:
        out = generate(model, enc, prompt, max_tokens=100)
        q = prompt.split('\n')[1]
        print(f"\nQ: {q}\nA: {out[:400]}", flush=True)

if __name__ == "__main__":
    main()
