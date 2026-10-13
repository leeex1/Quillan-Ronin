#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
👑 QUILLAN-RONIN v5.3.1 — FRONTIER CAPABILITY TRAINING ENGINE (v2)
===================================================================
FIXES vs previous run:
  - All tensors normalized to uniform MAX_SEQ_LEN (no size mismatch crash)
  - Uses frontier_intact_gold_master.pt (38,663 samples @ seq_len=384) as primary
  - Adds intact_thought_reasoning_gold.pt (3,196 samples @ seq_len=384)
  - Adds sovereign_thinking_gold.jsonl (17 curated complete <think> samples)
  - Also ingests Quillan_Master_Combined_Gold.jsonl with correct size filtering
  - All JSONL samples padded/clipped to EXACTLY MAX_SEQ_LEN=384
  - 89.3% trainable parameters (only wte frozen)
  - MuonK2 + AdamW + CCRL optimizer
  - 2000 steps with live generation probes every 150 steps

EXPECTED OUTCOME: Loss < 3.5, fluent multi-paragraph reasoning in all domains
"""

import os
import sys
import time
import json
import math
import random
import logging
from pathlib import Path
from typing import List, Tuple, Optional

import warnings
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR

# ── Path Setup ─────────────────────────────────────────────────────────────────
REPO_ROOT = Path(r"C:\02_QUILLAN")
DATA_DIR  = REPO_ROOT / "training_data"
CKPT_DIR  = REPO_ROOT / "checkpoints" / "checkpoints_sft"
PROD_DIR  = REPO_ROOT / "checkpoints" / "production_export"
CKPT_DIR.mkdir(parents=True, exist_ok=True)
PROD_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(REPO_ROOT / "scripts"))

from quillan_v10_unrolled_sovereign import QuillanRoninSovereign, QuillanArchConfig
from sovereign_inference_engine import SovereignTokenizer
from quillan_muonk2_optimizer import create_quillan_muonk2_optimizer

# ── Thread & CPU Hardware Optimization ───────────────────────────────────────
try:
    num_cores = os.cpu_count() or 4
    torch.set_num_threads(num_cores)
except Exception:
    pass

try:
    import psutil
    p = psutil.Process()
    p.nice(psutil.NORMAL_PRIORITY_CLASS)
except Exception:
    pass


def safe_torch_save(obj, target_path):
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
                break
            except Exception as e:
                time.sleep(0.2)
                if attempt == 4:
                    import shutil
                    try:
                        shutil.copy2(str(tmp_path), str(target_path))
                        tmp_path.unlink(missing_ok=True)
                    except Exception as err:
                        LOGGER.warning("safe_torch_save fallback error: %s, tmp: %s", err, tmp_path)
    except Exception as e:
        LOGGER.error("Failed to save checkpoint to %s: %s", target_path, e)

# ── Runtime Config ─────────────────────────────────────────────────────────────
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

LOGS_DIR = REPO_ROOT / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOGS_DIR / "training.log"
STATUS_FILE = LOGS_DIR / "live_status.json"

log_format = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s")
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(log_format)
file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
file_handler.setFormatter(log_format)

LOGGER = logging.getLogger("quillan.frontier_v2")
LOGGER.setLevel(logging.INFO)
LOGGER.handlers = [console_handler, file_handler]

def update_live_status(step, total_steps, loss, best_loss, lr, speed, active_experts=None, probe=None):
    try:
        status = {
            "step": step,
            "total_steps": total_steps,
            "progress_percent": round(100.0 * step / max(total_steps, 1), 2),
            "current_loss": round(float(loss), 4),
            "best_loss": round(float(best_loss), 4),
            "learning_rate": f"{lr:.2e}",
            "speed_steps_per_sec": round(float(speed), 2),
            "active_experts": active_experts or "C0-C33 (MoE Dynamic)",
            "last_updated": time.strftime("%Y-%m-%d %H:%M:%S"),
            "pid": os.getpid()
        }
        if probe:
            status["latest_probe"] = probe
        with open(STATUS_FILE, "w", encoding="utf-8") as f:
            json.dump(status, f, indent=2)
    except Exception:
        pass

# ── Hyperparameters ────────────────────────────────────────────────────────────
MAX_SEQ_LEN   = 256    # 256 tokens: zero-truncation for QA & fits 100% in CPU cache without page faults
BATCH_SIZE    = 2      # 2 sequences per sub-step
ACCUM_STEPS   = 1      # Lean memory per step
NUM_STEPS     = 7100   # Total optimizer steps targeting ~90k+ sample corpus
LR_MUON       = 0.015  # Fast low-rank Muon convergence rate
LR_ADAMW      = 1e-4   # Fast adaptive rate for bridges and routers
WEIGHT_DECAY  = 0.01
CCRL_LIMIT    = 4.0    # CCRL gradient curvature clip
AUX_ROUTER_WT = 0.02   # Auxiliary router diversity regularizer to enforce multi-expert deliberation
GRAD_CLIP     = 1.0
PROBE_EVERY   = 150    # Live generation probe frequency
SAVE_EVERY    = 25     # Save checkpoint candidate every N steps
BEST_CKPT     = "quillan_frontier_v2_best.pt"
# bfloat16 autocast halves activation+gradient memory; safer than float16 on CPU (no NaN risk)
USE_BFLOAT16  = True

# ── Dynamic Alpha Schedule for auxiliary load-balancing loss ──────────────────
# Stage 1 (Steps 1176-1999):  alpha=0.001  — stable warm-in under BitNet STE adaptation
# Stage 2 (Steps 2000-3499):  alpha=0.003  — increased expert diversity pressure at mid-run
# Stage 3 (Steps 3500-7100):  alpha=0.001  — release pressure, allow content-accuracy optimization
ALPHA_SCHEDULE = [(2000, 0.001), (3500, 0.003), (7101, 0.001)]

# ── Live Probe Prompts ─────────────────────────────────────────────────────────
PROBE_PROMPTS = [
    "<|user|>\nWhat is the difference between synchronous and asynchronous execution?\n<|assistant|>\n",
    "<|user|>\nIf all humans are mortal and Socrates is human, is Socrates mortal? Explain step by step.\n<|assistant|>\n",
    "<|user|>\nExplain the Second Law of Thermodynamics with a real-world example.\n<|assistant|>\n",
    "<|user|>\nWrite a Python function to perform binary search on a sorted list.\n<|assistant|>\n",
    "<|user|>\nWhat is the Ship of Theseus paradox and what does it reveal about identity?\n<|assistant|>\n",
]


def normalize_tensor_pair(
    inp: torch.Tensor, lbl: torch.Tensor, target_len: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pad or crop both tensors to exactly target_len. Critical to prevent stack() crashes."""
    curr = inp.size(0)
    if curr > target_len:
        inp = inp[:target_len]
        lbl = lbl[:target_len]
    elif curr < target_len:
        pad = target_len - curr
        inp = torch.cat([inp, torch.full((pad,), 50256, dtype=torch.long)])
        lbl = torch.cat([lbl, torch.full((pad,), -100, dtype=torch.long)])
    return inp, lbl


def load_frontier_corpus(tokenizer: SovereignTokenizer) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Loads all available training corpora, normalizes all tensors to MAX_SEQ_LEN=384.

    Corpus Priority Tiers:
      TIER 1 — Pre-tokenized .pt (highest quality, pre-validated sequences)
        1. frontier_intact_gold_master.pt     — 38,663 pristine intact samples
        2. intact_thought_reasoning_gold.pt   — 3,196  <think> reasoning chains
        3. augmented_frontier_v2.pt           — 8,848 multi-source augmented

      TIER 2 — JSONL Gold (teacher-distilled, curated, or domain-specific)
        4. GPT_5.5_Distilled.jsonl            — 18,198 GPT-5.5 teacher-distilled
           (pre-formatted <|user|>/<|assistant|> — uses text-format loader)
        5. Quillan_Universal_100_Percent_Master_Gold.jsonl — 30,100 sovereign QA
        6. Quillan_Master_Combined_Gold.jsonl — 12,000 cap (expanded from 6k)
        7. sovereign_thinking_gold.jsonl      — 17 curated complete <think> traces
        8. Quillan_Refined_Thought_Corpus.jsonl — 251 refined <thought> chains
        9. Quillan_Clean_Reasoning_Gold_Dataset.jsonl — multi-domain reasoning
       10. Quillan_Hyper_Tune_Gold_Dataset.jsonl — high-density tuning
       11. Quillan_Universal_Sovereign_Gold_1000.jsonl — sovereign QA
       12. Quillan_Direct_Answers_Gold.jsonl  — 1,200 factual anchor syllogisms
       13. Quillan_Explanatory_Prose_Dataset.jsonl — Quillan persona/capability docs
       14. pdf_papers_corpus.jsonl            — scientific paper corpus
       15. Quillan_General_Knowledge_Dataset.jsonl — general knowledge QA
       16. quillan_science_absolute.jsonl     — absolute science facts
       17. quillan_science_additional.jsonl   — additional science coverage

    All samples normalized to MAX_SEQ_LEN=384 with pad_token=50256 and label=-100.
    Samples with fewer than 10 non-masked label tokens are rejected.
    """
    samples: List[Tuple[torch.Tensor, torch.Tensor]] = []

    def add_pt_dataset(path: Path, label: str) -> int:
        """Load a pre-tokenized .pt dataset, normalizing all tensors to MAX_SEQ_LEN."""
        if not path.exists():
            LOGGER.warning("[SKIP] Not found: %s", path.name)
            return 0
        data = torch.load(str(path), map_location="cpu", weights_only=False)
        if not isinstance(data, dict) or "input_ids" not in data:
            LOGGER.warning("[SKIP] Invalid format: %s", path.name)
            return 0
        inp_all = data["input_ids"]
        lbl_all = data["labels"]
        added = 0
        for i in range(inp_all.size(0)):
            inp, lbl = normalize_tensor_pair(inp_all[i], lbl_all[i], MAX_SEQ_LEN)
            # Only include samples with meaningful response content
            if (lbl != -100).sum().item() >= 10:
                samples.append((inp, lbl))
                added += 1
        LOGGER.info("[+] %s: %d samples loaded.", label, added)
        return added

    def encode_jsonl_pair(prompt_text: str, response_text: str) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Encode and pad a prompt/response pair to exactly MAX_SEQ_LEN."""
        p_ids = tokenizer.encode(f"<|user|>\n{prompt_text.strip()}\n<|assistant|>\n")
        r_ids = tokenizer.encode(f"{response_text.strip()}<|im_end|>")
        seq = p_ids + r_ids
        labels = [-100] * len(p_ids) + list(r_ids)
        # Clip if too long — but only if we keep a meaningful response portion
        if len(seq) > MAX_SEQ_LEN:
            # Check that we're not clipping all the response
            resp_remaining = MAX_SEQ_LEN - len(p_ids)
            if resp_remaining < 20:
                return None  # Prompt is too long, skip
            seq = seq[:MAX_SEQ_LEN]
            labels = labels[:MAX_SEQ_LEN]
        # Pad if short
        pad_len = MAX_SEQ_LEN - len(seq)
        inp = seq + [50256] * pad_len
        lbl = labels + [-100] * pad_len
        # Validate response tokens exist
        if sum(1 for l in lbl if l != -100) < 10:
            return None
        return (
            torch.tensor(inp, dtype=torch.long),
            torch.tensor(lbl, dtype=torch.long),
        )

    def load_jsonl(path: Path, prompt_key: str, resp_key: str, label: str, cap: int = 10000) -> int:
        if not path.exists():
            LOGGER.warning("[SKIP] Not found: %s", path.name)
            return 0
        added = 0
        seen: set = set()
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if added >= cap:
                    break
                if not line.strip():
                    continue
                try:
                    d = json.loads(line)
                    q = d.get(prompt_key, "").strip()
                    r = d.get(resp_key, "").strip()
                    if not q or len(r) < 40:
                        continue
                    q_key = q[:60]
                    if q_key in seen:
                        continue
                    seen.add(q_key)
                    pair = encode_jsonl_pair(q, r)
                    if pair:
                        samples.append(pair)
                        added += 1
                except Exception:
                    pass
        LOGGER.info("[+] %s: %d samples loaded.", label, added)
        return added

    LOGGER.info("Loading training corpus (all tensors -> MAX_SEQ_LEN=%d)...", MAX_SEQ_LEN)

    # ── TIER 1: Pre-tokenized .pt datasets (highest quality, pre-validated) ───
    add_pt_dataset(DATA_DIR / "frontier_intact_gold_master.pt",    "Frontier Intact Gold Master (38k)")
    add_pt_dataset(DATA_DIR / "intact_thought_reasoning_gold.pt",  "Intact Thought Reasoning Gold (3k)")
    add_pt_dataset(DATA_DIR / "augmented_frontier_v2.pt",          "Augmented Frontier v2 (8.8k multi-source)")

    # ── TIER 2a: Pre-formatted text sequences (GPT-5.5 teacher-distilled) ─────
    # These have the <|user|>...<|assistant|> template baked in — use text-format loader
    def load_text_format_jsonl(path: Path, label: str, cap: int = 10000) -> int:
        """
        Loads JSONL files with 'text' key containing pre-formatted chat sequences.
        Expected format: {"text": "<|user|>\n...\n<|assistant|>\n...\n<|end|>"}
        Encodes the full text, splits at <|assistant|> boundary for label masking.
        """
        if not path.exists():
            LOGGER.warning("[SKIP] Not found: %s", path.name)
            return 0
        added = 0
        seen: set = set()
        asst_id = tokenizer.encode("<|assistant|>")[-1]  # Last token of the marker
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if added >= cap:
                    break
                if not line.strip():
                    continue
                try:
                    d = json.loads(line)
                    text = d.get("text", "").strip()
                    if not text or len(text) < 80:
                        continue
                    q_key = text[:60]
                    if q_key in seen:
                        continue
                    seen.add(q_key)

                    # Encode full text and find the first <|assistant|> boundary
                    full_ids = tokenizer.encode(text)
                    # Find where assistant response begins (first occurrence of asst_id)
                    split_idx = next(
                        (i + 1 for i, t in enumerate(full_ids) if t == asst_id),
                        min(40, len(full_ids))  # fallback: mask first 40 tokens
                    )

                    # Pad/clip to MAX_SEQ_LEN
                    if len(full_ids) > MAX_SEQ_LEN:
                        full_ids = full_ids[:MAX_SEQ_LEN]
                    pad_len = MAX_SEQ_LEN - len(full_ids)
                    inp = full_ids + [50256] * pad_len

                    # Labels: -100 for prompt, token IDs for response
                    lbl = [-100] * min(split_idx, MAX_SEQ_LEN) + \
                          full_ids[split_idx:MAX_SEQ_LEN]
                    lbl = lbl + [-100] * (MAX_SEQ_LEN - len(lbl))

                    if sum(1 for l in lbl if l != -100) < 10:
                        continue

                    samples.append((
                        torch.tensor(inp[:MAX_SEQ_LEN], dtype=torch.long),
                        torch.tensor(lbl[:MAX_SEQ_LEN], dtype=torch.long),
                    ))
                    added += 1
                except Exception:
                    pass
        LOGGER.info("[+] %s: %d samples loaded.", label, added)
        return added

    # ── TIER 2b: JSONL Gold Datasets (prompt/response or question/response) ────
    # GPT-5.5 teacher distilled — highest quality external distillation data
    load_text_format_jsonl(
        DATA_DIR / "GPT_5.5_Distilled.jsonl",
        "GPT-5.5 Teacher Distilled (18k)",
        cap=8000,  # Cap at 8k to avoid overfit on any single distribution
    )
    # Universal 100% Master Gold — 30,100 sovereign QA pairs
    load_jsonl(DATA_DIR / "Quillan_Universal_100_Percent_Master_Gold.jsonl",
               "prompt", "response", "Universal 100% Master Gold", cap=8000)
    # Master Combined Gold — expanded from 6k to 12k cap
    load_jsonl(DATA_DIR / "Quillan_Master_Combined_Gold.jsonl",
               "prompt", "response", "Master Combined Gold", cap=12000)
    # Sovereign <think> traces — curated multi-step reasoning chains
    load_jsonl(DATA_DIR / "sovereign_thinking_gold.jsonl",
               "question", "response", "Sovereign Thinking Gold (curated <think>)")
    # Refined Thought Corpus — 251 structured <thought> analysis chains
    load_jsonl(DATA_DIR / "Quillan_Refined_Thought_Corpus.jsonl",
               "prompt", "refined_reasoning", "Refined Thought Corpus (251 <thought>)")
    # Clean multi-domain reasoning
    load_jsonl(DATA_DIR / "Quillan_Clean_Reasoning_Gold_Dataset.jsonl",
               "question", "response", "Clean Reasoning Gold")
    # High-density tuning examples
    load_jsonl(DATA_DIR / "Quillan_Hyper_Tune_Gold_Dataset.jsonl",
               "question", "response", "Hyper-Tune Gold")
    # Sovereign QA 1000
    load_jsonl(DATA_DIR / "Quillan_Universal_Sovereign_Gold_1000.jsonl",
               "question", "response", "Universal Sovereign Gold")
    # Direct factual anchors — 1,200 formal syllogisms and science proofs
    load_jsonl(DATA_DIR / "Quillan_Direct_Answers_Gold.jsonl",
               "prompt", "response", "Direct Factual Anchors", cap=1200)
    # Quillan persona and capability explanations
    load_jsonl(DATA_DIR / "Quillan_Explanatory_Prose_Dataset.jsonl",
               "question", "response", "Explanatory Prose (Quillan Persona)")
    # General knowledge QA
    load_jsonl(DATA_DIR / "Quillan_General_Knowledge_Dataset.jsonl",
               "question", "response", "General Knowledge", cap=500)
    # Scientific corpus — absolute facts and additional coverage
    # These use 'text' key containing pre-formatted PHYSICS PROBLEM: ... style content
    load_text_format_jsonl(
        DATA_DIR / "quillan_science_absolute.jsonl",
        "Science Absolute (physics/science text format)",
        cap=800,
    )
    load_text_format_jsonl(
        DATA_DIR / "quillan_science_additional.jsonl",
        "Science Additional (physics/science text format)",
        cap=800,
    )
    # PDF papers corpus — academic/research paper knowledge distillation
    load_text_format_jsonl(
        DATA_DIR / "pdf_papers_corpus.jsonl",
        "PDF Papers Corpus (scientific)",
        cap=100,  # Papers are very long; cap at 100 for sequence diversity
    )

    random.shuffle(samples)
    LOGGER.info("[\u2713] Total frontier corpus: %d complete intact samples.", len(samples))
    return samples


@torch.no_grad()
def run_probe(model: torch.nn.Module, tokenizer: SovereignTokenizer, step: int) -> None:
    """Execute a live generation probe and log the result."""
    model.eval()
    prompt = random.choice(PROBE_PROMPTS)
    toks = tokenizer.encode(prompt)

    gen_ids = model.generate(
        toks,
        max_tokens=250,
        temp=0.20,
        top_k=40,
        top_p=0.90,
        repetition_penalty=1.05,
        frequency_penalty=0.0,
        presence_penalty=0.0,
    )

    gen_text = tokenizer.decode(gen_ids[len(toks):])
    gen_text = gen_text.split("<|im_end|>")[0].split("<|endoftext|>")[0].strip()
    q_display = prompt.replace("<|user|>\n", "").replace("\n<|assistant|>\n", "").strip()

    LOGGER.info("=" * 70)
    LOGGER.info("[PROBE @ Step %d]", step)
    LOGGER.info("Q: %s", q_display)
    LOGGER.info("A:\n%s", gen_text)
    LOGGER.info("=" * 70)
    model.train()
    return {"prompt": q_display, "response": gen_text}


def train():
    LOGGER.info("=" * 70)
    LOGGER.info("   👑 QUILLAN-RONIN v5.3.1 — FRONTIER CAPABILITY TRAINING v2")
    LOGGER.info("=" * 70)

    device = torch.device("cpu")
    tokenizer = SovereignTokenizer("gpt2")
    cfg = QuillanArchConfig()
    model = QuillanRoninSovereign(cfg).to(device)

    # ── Resume from best available checkpoint ─────────────────────────────────
    candidates = [
        CKPT_DIR / "quillan_frontier_v2_latest.pt",
        CKPT_DIR / BEST_CKPT,
        CKPT_DIR / "quillan_frontier_generalization_best.pt",
        CKPT_DIR / "quillan_direct_factual_best.pt",
        PROD_DIR / "quillan_ronin_v531_sovereign_production.pt",
    ]
    best_loss = float("inf")
    start_step = 1

    for ckpt_path in candidates:
        if ckpt_path.exists():
            LOGGER.info("Resuming from: %s", ckpt_path.name)
            import gc
            gc.collect()
            ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False, mmap=True)
            sd = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
            
            # Shape-safe state dictionary loading
            model_sd = model.state_dict()
            filtered_sd = {}
            for k, v in sd.items():
                if k in model_sd:
                    if model_sd[k].shape == v.shape:
                        filtered_sd[k] = v
                    else:
                        LOGGER.warning("Skipping key %s due to shape mismatch: %s vs %s", k, model_sd[k].shape, v.shape)
            
            miss, unex = model.load_state_dict(filtered_sd, strict=False)
            LOGGER.info("Loaded checkpoint (matching=%d, missing=%d, unexpected=%d)", len(filtered_sd), len(miss), len(unex))
            if "loss" in ckpt:
                best_loss = float(ckpt["loss"])
                LOGGER.info("Resuming with best recorded loss: %.4f", best_loss)
            if "step" in ckpt:
                start_step = int(ckpt["step"]) + 1
                LOGGER.info("Resuming from recorded step: %d", start_step)
            break

    # ── Active Parameter Scoping (Edge-Native & Low-Memory Hardening) ─────────
    trainable = 0
    total = 0
    for name, p in model.named_parameters():
        total += p.numel()
        if any(k in name for k in ['lora', 'swarm', 'expert_swarms', 'q1_bridge', 'q2_bridge', 'ingest_gate', 'prism', 'ln_', 'router']):
            p.requires_grad = True
            trainable += p.numel()
        else:
            p.requires_grad = False

    LOGGER.info(
        "Active Trainable: %s / %s (%.1f%%) — Zero-Paging Mode",
        f"{trainable:,}", f"{total:,}", 100.0 * trainable / total,
    )

    # ── Load corpus ───────────────────────────────────────────────────────────
    dataset = load_frontier_corpus(tokenizer)
    N = len(dataset)
    if N == 0:
        LOGGER.error("No training samples loaded! Aborting.")
        return

    # ── Optimizer ─────────────────────────────────────────────────────────────
    optimizer = create_quillan_muonk2_optimizer(
        model,
        lr_muon=LR_MUON,
        lr_adamw=LR_ADAMW,
        weight_decay=WEIGHT_DECAY,
        ccrl_limit=CCRL_LIMIT,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=NUM_STEPS, eta_min=1e-5)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(start_step - 1):
            scheduler.step()

    LOGGER.info("[+] MuonK2 + AdamW + CCRL Optimizer Initialized (Starting @ Step %d).", start_step)
    LOGGER.info(
        "Training: %d steps | EffBatch=%d | MAX_SEQ_LEN=%d | Corpus=%d",
        NUM_STEPS, BATCH_SIZE * ACCUM_STEPS, MAX_SEQ_LEN, N,
    )

    model.train()
    window_loss = 0.0
    t0 = time.time()
    data_idx = (start_step - 1) * BATCH_SIZE * ACCUM_STEPS

    # Determine autocast dtype: bfloat16 on CPU avoids NaN that float16 can produce
    _autocast_dtype = torch.bfloat16 if USE_BFLOAT16 else None

    for step in range(start_step, NUM_STEPS + 1):
        optimizer.zero_grad(set_to_none=True)
        step_loss = 0.0

        for _ in range(ACCUM_STEPS):
            batch_inp, batch_lbl = [], []
            for _ in range(BATCH_SIZE):
                inp, lbl = dataset[data_idx % N]
                inp, lbl = normalize_tensor_pair(inp, lbl, MAX_SEQ_LEN)
                batch_inp.append(inp)
                batch_lbl.append(lbl)
                data_idx += 1

            inp_t = torch.stack(batch_inp).to(device)
            lbl_t = torch.stack(batch_lbl).to(device)

            # bfloat16 autocast: forward pass runs in bf16, cutting activation memory ~50%.
            # Loss scaling not needed for bfloat16 on CPU (unlike float16 which can underflow).
            if _autocast_dtype is not None:
                with torch.amp.autocast(device_type="cpu", dtype=_autocast_dtype):
                    logits, loss = model(inp_t, labels=lbl_t)
            else:
                logits, loss = model(inp_t, labels=lbl_t)

            (loss / ACCUM_STEPS).backward()
            step_loss += loss.item() / ACCUM_STEPS
            del inp_t, lbl_t, logits, loss

        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad],
            GRAD_CLIP,
        )
        optimizer.step()
        scheduler.step()
        window_loss += step_loss

        # Print progress every single step for clear live visibility
        if step % 25 == 0:
            import gc
            gc.collect()

        avg = step_loss
        elapsed = time.time() - t0
        speed = 1.0 / max(elapsed, 1e-4)
        lr_now = scheduler.get_last_lr()[0]
        LOGGER.info(
            "Step [%4d/%4d] | Loss: %.4f | LR: %.2e | %.2f step/s",
            step, NUM_STEPS, avg, lr_now, speed
        )
        update_live_status(step, NUM_STEPS, avg, best_loss, lr_now, speed)

        if step % 25 == 0 and hasattr(model, "_last_router_probs"):
            r_probs = model._last_router_probs.float().cpu().numpy()
            active_count = int((r_probs > 0.005).sum())
            top_3_idx = r_probs.argsort()[-3:][::-1]
            top_3_str = ", ".join([f"C{idx}:{r_probs[idx]:.3f}" for idx in top_3_idx])
            LOGGER.info(
                "[ROUTER @ Step %d] Active: %d/34 | Mean: %.4f | Min: %.4f | Max: %.4f | Top-3: [%s]",
                step, active_count, float(r_probs.mean()), float(r_probs.min()), float(r_probs.max()), top_3_str
            )
            update_live_status(step, NUM_STEPS, avg, best_loss, lr_now, speed, active_experts=f"{active_count}/34 ({top_3_str})")

        # Dynamic alpha schedule: update model.aux_alpha at each threshold boundary
        for threshold, alpha_val in ALPHA_SCHEDULE:
            if step == threshold:
                model.aux_alpha = alpha_val
                stage_name = (
                    "warm-in" if alpha_val == 0.001 and step < 2000 else
                    "mid-run pressure" if alpha_val == 0.003 else
                    "content-accuracy release"
                )
                LOGGER.info(
                    "[ALPHA SCHEDULE] Step %d: model.aux_alpha set to %.4f (%s)",
                    step, alpha_val, stage_name
                )

        t0 = time.time()

        if avg < best_loss and step >= SAVE_EVERY:
            best_loss = avg
            save_dict = {"model_state_dict": model.state_dict(), "step": step, "loss": best_loss}
            safe_torch_save(save_dict, CKPT_DIR / BEST_CKPT)
            safe_torch_save(save_dict, PROD_DIR / "quillan_ronin_v531_sovereign_production.pt")
            LOGGER.info("🏆 New Best Checkpoint: %.4f @ Step %d", best_loss, step)

        if step % SAVE_EVERY == 0:
            latest_dict = {"model_state_dict": model.state_dict(), "step": step, "loss": avg}
            safe_torch_save(latest_dict, CKPT_DIR / "quillan_frontier_v2_latest.pt")

        if step % PROBE_EVERY == 0:
            probe_out = run_probe(model, tokenizer, step)
            if probe_out:
                update_live_status(step, NUM_STEPS, avg, best_loss, lr_now, speed, probe=probe_out)

    LOGGER.info("=" * 70)
    LOGGER.info("   🏆 FRONTIER TRAINING v2 COMPLETE — Best Loss: %.4f", best_loss)
    LOGGER.info("=" * 70)


if __name__ == "__main__":
    train()
