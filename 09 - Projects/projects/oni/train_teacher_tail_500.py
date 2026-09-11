#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
👑 QUILLAN-RONIN v5.4-ONI — TEACHER SFT TAIL TRAINING ENGINE (500 STEPS)
========================================================================
Purpose:
  Targeted 500-step fine-tuning tail initializing strictly from
  `quillan_frontier_v2_best.pt` (step 5251, loss 0.9165).
  Up-samples `<think>...</think>` chain-of-thought closing sequences
  to eliminate repetitive looping and reinforce structured answers.

Hardware Governor:
  - 3 worker threads on Intel Core i5-7500 CPU (zero desktop latency)
  - Below-normal OS process priority
  - In-place working set memory compaction

Optimizer & Schedule:
  - MuonK2 + AdamW + CCRL (lr: 5e-5 -> 1e-5 via Cosine Annealing)
  - 18.6% active trainable parameter scoping (Swarm + LoRA + Bridges)
  - Fluency Gate evaluation on completion
"""

import os
import sys
import time
import json
import math
import random
import logging
import warnings
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR

# ── Path Resolution ────────────────────────────────────────────────────────────
REPO_ROOT = Path(r"C:\02_QUILLAN")
DATA_DIR = REPO_ROOT / "training_data"
CKPT_DIR = REPO_ROOT / "checkpoints" / "checkpoints_sft"
LOGS_DIR = REPO_ROOT / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)
CKPT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(REPO_ROOT / "03 - Training & Model" / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from quillan_v5_4_oni import QuillanRoninOni as QuillanRoninSovereign, QuillanOniConfig as QuillanArchConfig
from sovereign_inference_engine import SovereignTokenizer
from quillan_muonk2_optimizer import create_quillan_muonk2_optimizer

# ── Hardware Thread & Governor Hardening ──────────────────────────────────────
try:
    if not torch.cuda.is_available():
        torch.set_num_threads(min(3, os.cpu_count() or 3))
        torch.set_num_interop_threads(min(2, os.cpu_count() or 2))
except Exception:
    pass

try:
    import psutil
    p = psutil.Process()
    if hasattr(psutil, "BELOW_NORMAL_PRIORITY_CLASS"):
        p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
except Exception:
    pass

# ── Logging Configuration ─────────────────────────────────────────────────────
LOG_FILE = LOGS_DIR / "teacher_tail_training.log"
STATUS_FILE = LOGS_DIR / "teacher_tail_status.json"

log_format = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s")
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(log_format)
file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
file_handler.setFormatter(log_format)

LOGGER = logging.getLogger("quillan.teacher_tail")
LOGGER.setLevel(logging.INFO)
LOGGER.handlers = [console_handler, file_handler]

# ── Checkpoint Persistence ────────────────────────────────────────────────────
def safe_torch_save(obj: Dict[str, Any], target_path: Path) -> bool:
    """Safely persist PyTorch checkpoint using atomic rename with fallback retries."""
    target_path = Path(target_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target_path.with_name(f"{target_path.stem}_tmp_{os.getpid()}_{int(time.time()*1000)}{target_path.suffix}")
    try:
        torch.save(obj, str(tmp_path))
        for attempt in range(5):
            try:
                if target_path.exists():
                    try:
                        target_path.unlink()
                    except Exception:
                        pass
                os.replace(str(tmp_path), str(target_path))
                return True
            except Exception:
                time.sleep(0.2)
                if attempt == 4:
                    import shutil
                    try:
                        shutil.copy2(str(tmp_path), str(target_path))
                        tmp_path.unlink(missing_ok=True)
                        return True
                    except Exception as err:
                        LOGGER.warning("safe_torch_save fallback error: %s, tmp: %s", err, tmp_path)
        return False
    except Exception as e:
        LOGGER.error("Failed to save checkpoint to %s: %s", target_path, e)
        return False

# ── Hyperparameters ────────────────────────────────────────────────────────────
MAX_SEQ_LEN = 256
BATCH_SIZE = 2
ACCUM_STEPS = 1
NUM_TAIL_STEPS = 500
LR_INITIAL = 5e-5
LR_MIN = 1e-5
WEIGHT_DECAY = 0.01
CCRL_LIMIT = 4.0
PROBE_EVERY = 50
SAVE_EVERY = 50
BASE_CKPT = CKPT_DIR / "quillan_frontier_v2_best.pt"
TAIL_BEST_CKPT = CKPT_DIR / "quillan_teacher_tail_best.pt"
TAIL_LATEST_CKPT = CKPT_DIR / "quillan_teacher_tail_latest.pt"

PROBE_PROMPTS = [
    "<|user|>\nIf all humans are mortal and Socrates is human, is Socrates mortal? Explain your logical deduction.\n<|assistant|>\n",
    "<|user|>\nWhat is the difference between synchronous and asynchronous execution?\n<|assistant|>\n",
    "<|user|>\nA triangle has side lengths 5, 12, and 13. Is it a right triangle? Show your work.\n<|assistant|>\n",
    "<|user|>\nExplain the Second Law of Thermodynamics with a real-world example.\n<|assistant|>\n",
]

def normalize_tensor_pair(inp: torch.Tensor, lbl: torch.Tensor, target_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pad or crop both tensors to exactly target_len."""
    if not isinstance(inp, torch.Tensor):
        inp = torch.tensor(inp, dtype=torch.long)
    if not isinstance(lbl, torch.Tensor):
        lbl = torch.tensor(lbl, dtype=torch.long)
    inp = inp.squeeze()
    lbl = lbl.squeeze()
    if inp.dim() == 0:
        inp = inp.unsqueeze(0)
    if lbl.dim() == 0:
        lbl = lbl.unsqueeze(0)
    curr = inp.size(0)
    if curr > target_len:
        inp = inp[:target_len]
        lbl = lbl[:target_len]
    elif curr < target_len:
        pad = target_len - curr
        inp = torch.cat([inp, torch.full((pad,), 50256, dtype=torch.long)])
        lbl = torch.cat([lbl, torch.full((pad,), -100, dtype=torch.long)])
    return inp.contiguous(), lbl.contiguous()

def load_tail_corpus(tokenizer: SovereignTokenizer) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """Loads curated reasoning corpus with heavy up-sampling of <think> closing chains."""
    samples: List[Tuple[torch.Tensor, torch.Tensor]] = []

    def add_pt_dataset(path: Path, label: str, repeat: int = 1, cap: Optional[int] = None) -> int:
        if not path.exists():
            LOGGER.warning("[SKIP] Not found: %s", path.name)
            return 0
        data = torch.load(str(path), map_location="cpu", weights_only=False)
        if not isinstance(data, dict) or "input_ids" not in data:
            return 0
        inp_all = data["input_ids"]
        lbl_all = data["labels"]
        count = inp_all.size(0) if cap is None else min(inp_all.size(0), cap)
        added = 0
        for _ in range(repeat):
            for i in range(count):
                inp, lbl = normalize_tensor_pair(inp_all[i], lbl_all[i], MAX_SEQ_LEN)
                if (lbl != -100).sum().item() >= 10:
                    samples.append((inp, lbl))
                    added += 1
        LOGGER.info("[+] %s: %d samples loaded (repeat=%d).", label, added, repeat)
        return added

    def encode_jsonl_pair(prompt_text: str, response_text: str) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        p_ids = tokenizer.encode(f"<|user|>\n{prompt_text.strip()}\n<|assistant|>\n")
        r_ids = tokenizer.encode(f"{response_text.strip()}<|im_end|>")
        seq = p_ids + r_ids
        labels = [-100] * len(p_ids) + list(r_ids)
        if len(seq) > MAX_SEQ_LEN:
            resp_remaining = MAX_SEQ_LEN - len(p_ids)
            if resp_remaining < 20:
                return None
            seq = seq[:MAX_SEQ_LEN]
            labels = labels[:MAX_SEQ_LEN]
        pad_len = MAX_SEQ_LEN - len(seq)
        inp = seq + [50256] * pad_len
        lbl = labels + [-100] * pad_len
        if sum(1 for l in lbl if l != -100) < 10:
            return None
        return (torch.tensor(inp, dtype=torch.long), torch.tensor(lbl, dtype=torch.long))

    def load_jsonl(path: Path, prompt_key: str, resp_key: str, label: str, repeat: int = 1, cap: int = 10000) -> int:
        if not path.exists():
            LOGGER.warning("[SKIP] Not found: %s", path.name)
            return 0
        batch_pairs = []
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if len(batch_pairs) >= cap:
                    break
                if not line.strip():
                    continue
                try:
                    d = json.loads(line)
                    q = d.get(prompt_key, "").strip()
                    r = d.get(resp_key, "").strip()
                    if not q or len(r) < 30:
                        continue
                    pair = encode_jsonl_pair(q, r)
                    if pair:
                        batch_pairs.append(pair)
                except Exception:
                    pass
        added = 0
        for _ in range(repeat):
            samples.extend(batch_pairs)
            added += len(batch_pairs)
        LOGGER.info("[+] %s: %d samples loaded (repeat=%d).", label, added, repeat)
        return added

    LOGGER.info("Ingesting Teacher Tail Corpus (Focus: Intact <think> Reasoning)...")
    # 1. Intact Thought Reasoning (High quality <think> chains) - repeat x2
    add_pt_dataset(DATA_DIR / "intact_thought_reasoning_gold.pt", "Intact Thought Reasoning Gold (3k)", repeat=2)
    # 2. Sovereign Thinking Gold (18 curated complete <think> traces) - up-sample x25
    load_jsonl(DATA_DIR / "sovereign_thinking_gold.jsonl", "question", "response", "Sovereign Thinking Gold (curated)", repeat=25)
    # 3. Refined Thought Corpus - repeat x2
    load_jsonl(DATA_DIR / "Quillan_Refined_Thought_Corpus.jsonl", "prompt", "refined_reasoning", "Refined Thought Corpus", repeat=2)
    # 4. Direct factual anchors (1200 syllogisms & facts)
    load_jsonl(DATA_DIR / "Quillan_Direct_Answers_Gold.jsonl", "prompt", "response", "Direct Factual Anchors", cap=1200)
    # 5. Frontier intact master (anchor 5,000 general samples)
    add_pt_dataset(DATA_DIR / "frontier_intact_gold_master.pt", "Frontier Intact Master (Anchor 5k)", repeat=1, cap=5000)

    random.shuffle(samples)
    LOGGER.info("[✓] Total Teacher Tail Corpus: %d samples ready.", len(samples))
    return samples

@torch.no_grad()
def run_generation_probe(model: torch.nn.Module, tokenizer: SovereignTokenizer, step: int) -> str:
    """Generate sample continuation and inspect for <think> closure and fluency."""
    model.eval()
    prompt = random.choice(PROBE_PROMPTS)
    tokens = tokenizer.encode(prompt)

    gen_ids = model.generate(
        tokens,
        max_tokens=100,
        temp=0.20,
        top_k=40,
        top_p=0.90,
        repetition_penalty=1.05,
    )

    decoded = tokenizer.decode(gen_ids[len(tokens):])
    decoded = decoded.split("<|im_end|>")[0].split("<|endoftext|>")[0].strip()
    LOGGER.info("--- [PROBE @ STEP %d] ---", step)
    LOGGER.info("Prompt: %s", prompt.strip())
    LOGGER.info("Response: %s", decoded.strip())
    LOGGER.info("--------------------------")
    model.train()
    return decoded.strip()

@torch.no_grad()
def evaluate_fluency_gate(model: torch.nn.Module, tokenizer: SovereignTokenizer) -> Dict[str, Any]:
    """Evaluates the model against 3 standardized test prompts to verify fluency and structure."""
    model.eval()
    results = {}
    pass_count = 0
    test_prompts = [
        ("Syllogism", "<|user|>\nIf all humans are mortal and Socrates is human, is Socrates mortal? Explain step by step.\n<|assistant|>\n"),
        ("Asynchronous", "<|user|>\nWhat is the difference between synchronous and asynchronous execution?\n<|assistant|>\n"),
        ("Geometry", "<|user|>\nA triangle has side lengths 5, 12, and 13. Is it a right triangle? Show your work.\n<|assistant|>\n"),
    ]

    for name, prompt in test_prompts:
        tokens = tokenizer.encode(prompt)
        gen_ids = model.generate(
            tokens,
            max_tokens=150,
            temp=0.20,
            top_k=40,
            top_p=0.90,
            repetition_penalty=1.05,
        )
        resp = tokenizer.decode(gen_ids[len(tokens):])
        resp = resp.split("<|im_end|>")[0].split("<|endoftext|>")[0].strip()
        words = resp.split()
        unique_ratio = len(set(words)) / max(len(words), 1)
        passed = len(resp) >= 40 and unique_ratio > 0.35
        if passed:
            pass_count += 1
        results[name] = {
            "response_preview": resp[:200] + ("..." if len(resp) > 200 else ""),
            "total_tokens": len(gen_ids) - len(tokens),
            "unique_ratio": round(unique_ratio, 3),
            "passed": passed,
        }

    overall_pass = pass_count >= 2
    model.train()
    return {"passed": overall_pass, "pass_count": pass_count, "details": results}

def run_teacher_tail_training():
    LOGGER.info("======================================================================")
    LOGGER.info("   👑 QUILLAN-RONIN v5.4-ONI — TEACHER SFT TAIL TRAINING (500 STEPS)  ")
    LOGGER.info("======================================================================")

    if not BASE_CKPT.exists():
        LOGGER.error("Base checkpoint does not exist: %s", BASE_CKPT)
        sys.exit(1)

    device = torch.device("cpu")
    tokenizer = SovereignTokenizer("gpt2")
    cfg = QuillanArchConfig(
        n_layer=6,
        hidden_dim=1024,
        max_seq_len=MAX_SEQ_LEN,
        num_experts=34,
        router_mode="dense_pull",
    )
    model = QuillanRoninSovereign(cfg).to(device)

    LOGGER.info("Loading Base Checkpoint: %s", BASE_CKPT.name)
    base_data = torch.load(str(BASE_CKPT), map_location=device, weights_only=False)
    sd = base_data.get("model_state_dict", base_data)
    miss, unex = model.load_state_dict(sd, strict=False)
    base_step = int(base_data.get("step", 5251))
    base_loss = float(base_data.get("loss", 0.9165))
    LOGGER.info("Checkpoint loaded successfully. Base Step: %d | Base Loss: %.4f", base_step, base_loss)
    del base_data, sd

    # ── Active Parameter Scoping (18.6% active trainable params) ─────────────
    trainable_numel = 0
    total_numel = 0
    for name, p in model.named_parameters():
        total_numel += p.numel()
        if any(k in name for k in ['lora', 'swarm', 'expert_swarms', 'q1_bridge', 'q2_bridge', 'ingest_gate', 'prism', 'ln_', 'router', 'marta', 'dqso', 'e_ice', 'finalizer', 'evo_moe']):
            p.requires_grad = True
            trainable_numel += p.numel()
        else:
            p.requires_grad = False

    LOGGER.info("Active Trainable Parameters: %s / %s (%.1f%%)", f"{trainable_numel:,}", f"{total_numel:,}", 100.0 * trainable_numel / total_numel)

    # ── Load Tail Corpus ───────────────────────────────────────────────────────
    corpus = load_tail_corpus(tokenizer)
    if not corpus:
        LOGGER.error("No training samples loaded. Aborting.")
        return

    # ── Optimizer & Cosine Annealing Scheduler ────────────────────────────────
    optimizer = create_quillan_muonk2_optimizer(
        model,
        lr_muon=LR_INITIAL,
        lr_adamw=LR_INITIAL,
        weight_decay=WEIGHT_DECAY,
        ccRL_limit=CCRL_LIMIT if hasattr(create_quillan_muonk2_optimizer, 'ccRL_limit') else 4.0,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=NUM_TAIL_STEPS, eta_min=LR_MIN)

    LOGGER.info("Optimizer initialized: MuonK2 + AdamW (LR: %.2e -> %.2e over %d steps)", LR_INITIAL, LR_MIN, NUM_TAIL_STEPS)

    # Baseline Probe prior to training
    LOGGER.info("Executing Pre-Training Baseline Probe...")
    run_generation_probe(model, tokenizer, step=0)

    model.train()
    best_tail_loss = base_loss
    data_idx = 0
    t0 = time.time()

    for tail_step in range(1, NUM_TAIL_STEPS + 1):
        step_t0 = time.time()
        optimizer.zero_grad(set_to_none=True)

        batch_inp = []
        batch_lbl = []
        for _ in range(BATCH_SIZE):
            inp, lbl = corpus[data_idx % len(corpus)]
            batch_inp.append(inp)
            batch_lbl.append(lbl)
            data_idx += 1

        x = torch.stack(batch_inp).to(device)
        y = torch.stack(batch_lbl).to(device)

        out = model(x, labels=y)
        if isinstance(out, tuple) and len(out) == 3:
            logits, ce, aux = out
            loss = ce + (model.total_aux_loss(aux) if hasattr(model, "total_aux_loss") else 0.0)
        elif isinstance(out, tuple) and len(out) == 2:
            logits, loss = out
        else:
            logits = out
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=-100)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad],
            1.0
        )
        optimizer.step()
        scheduler.step()

        current_loss = loss.item()
        step_dt = time.time() - step_t0
        current_lr = scheduler.get_last_lr()[0]

        if tail_step % 10 == 0 or tail_step == 1:
            LOGGER.info("Tail Step [%3d/%d] | Loss: %.4f (Best: %.4f) | LR: %.2e | %.2f s/step", tail_step, NUM_TAIL_STEPS, current_loss, best_tail_loss, current_lr, step_dt)

        # Periodic generation probe
        if tail_step % PROBE_EVERY == 0:
            run_generation_probe(model, tokenizer, step=tail_step)

        # Periodic checkpoint save
        if tail_step % SAVE_EVERY == 0 or tail_step == NUM_TAIL_STEPS:
            is_best = current_loss < best_tail_loss
            if is_best:
                best_tail_loss = current_loss
                LOGGER.info("🏆 New Best Tail Checkpoint: %.4f @ Tail Step %d", best_tail_loss, tail_step)
                safe_torch_save({
                    "model_state_dict": model.state_dict(),
                    "step": base_step + tail_step,
                    "tail_step": tail_step,
                    "loss": best_tail_loss,
                }, TAIL_BEST_CKPT)
                # If beating the historical 0.9165 loss, also preserve to frontier v2 best
                if best_tail_loss < base_loss:
                    LOGGER.info("🌟 Surpassed global best loss (%.4f < %.4f)! Updating frontier v2 best.", best_tail_loss, base_loss)
                    safe_torch_save({
                        "model_state_dict": model.state_dict(),
                        "step": base_step + tail_step,
                        "loss": best_tail_loss,
                    }, BASE_CKPT)

            safe_torch_save({
                "model_state_dict": model.state_dict(),
                "step": base_step + tail_step,
                "tail_step": tail_step,
                "loss": current_loss,
            }, TAIL_LATEST_CKPT)

    total_time = time.time() - t0
    LOGGER.info("======================================================================")
    LOGGER.info("   [✓] 500-STEP TEACHER TAIL TRAINING COMPLETE (Elapsed: %.1f min)     ", total_time / 60.0)
    LOGGER.info("   Final Best Loss: %.4f                                              ", best_tail_loss)
    LOGGER.info("======================================================================")

    # ── Post-Training Fluency Gate ─────────────────────────────────────────────
    LOGGER.info("Executing Final Fluency Verification Gate...")
    gate_results = evaluate_fluency_gate(model, tokenizer)
    LOGGER.info("Fluency Gate Result: %s (Pass Count: %d/3)", "PASSED [✓]" if gate_results["passed"] else "FAILED [X]", gate_results["pass_count"])
    for test_name, d in gate_results["details"].items():
        LOGGER.info("  * %s: %s (unique_ratio: %.2f)", test_name, "PASS" if d["passed"] else "FAIL", d["unique_ratio"])
        LOGGER.info("    Preview: %s", d["response_preview"])

    with open(LOGS_DIR / "teacher_tail_fluency_report.json", "w", encoding="utf-8") as f:
        json.dump(gate_results, f, indent=2)

    return gate_results

if __name__ == "__main__":
    run_teacher_tail_training()
