# PLAN.md — Quillan Recovery & Completion (approved 2026-09-17)

## 0. Current state
- Local checkpoints lost (cleanup with no backup — my fault). Raw data, code,
  fixed tokenizer (+backup), logs, and 850MB ternary qbin survived.
- Restored from HF `CrashOverrideX/Quillan-Ronin` (66.4K downloads, lineage
  BitNet-b1_58-3B + Qwen3.5-0.8B) into `checkpoints\hf_restore\`:
  `quillan_frontier_v2_best_loss0.0789_step2500.pt` (1.9GB) +
  `quillan_oni_5.4.0_step660_5.22GB.pt` (5.6GB).
- Triage: frontier needs transpose-bind (HF convention flipped); MoE c_proj
  genuine mismatch ([1024,4096] actual vs [1024,2048]) → frontier = FOREIGN
  donor (probe/embeddings only), NOT graft target.
- Speed doctrine: stop ollama/edge/browser/extra; 3-thread ceiling; cap-style
  checks only until green.
- VRAM (GTX 1050 4GB): CPU lane = custom models (PyTorch/quillan.cpp).
  GPU lane = standard models via Ollama for ChatGPT-tier answers.
  No GPU work on custom models until CUDA port exists.
- Space: 69GB free. Nothing >2GB lands without explicit approval. No 13GB
  merged download (space kills the box, and 1050 can't run fp32 6.7B anyway).

## 1. Phase 1 — Frontier binding probe (NOW)
Script `scripts/probe_hf_bind.py`: transpose-bind reversed 2D tensors, report
bound/skipped per module, frozen 10-batch loss probe (~3 min, 3-thread, detach).
Accept: binding diagram with zero ambiguity.

## 1b. Build order (owner-locked)
MINI FIRST, fully trained and verified on this box — then scale the recipe to
the 1B Main. 1B is the hard size cap (local machine); DENSITY is unlimited:
MoE, swarm, prism, reasoning traces, TRM/HRM-style recurrence all live INSIDE
the small envelope. Small-vs-large is settled science; we build dense, not big.

## 2. Phase 2 — Tri-Run head-only (after Phase 1 green)
Re-stage mapped-6L from qbin (fp32 approx), repack 20k CPU slice from
surviving CLEAN_V7.jsonl, run head-64 Tri-Run (45/val40/stop2.5, BEST-only).
Gate: any config with val<5 AND gen-score>=4 proceeds.

## 3. Phase 3 — Main 12L
Triage 5.22GB file (resume pt, strict arch matching first). If foreign:
distill from best Mini head, not graft. Serve via quillan.cpp loader.

## 4. Phase 4 — Serving
Gateway on unified tokenizer (done). quillan.cpp fast path: general loader
for ternary+FP mixes using manifest schema. Backfill manifest V2 (deps,
calibration, tokenizer contract, harness cmd) for every future artifact.

## 5. Safety rules (never again)
Push every best to HF, even half-finished infants — backups first, pride later.
Backup manifest before any delete. Two-person rule on weight deletes
(list + confirm). Mirror policy: ONE canonical tree, no silent twins.

## 6b. Product requirement (owner-locked)
Mini = 0.6B REASONING model (6L/1024 line). Main = 1B REASONING model
(12L/1536 line). Both trained on thinking-trace data (SFT first, GRPO later).
Main candidate: hf_restore 5.22GB file (BOUND STRICT 3652/3652).
LINE INDEPENDENCE: Mini and Main are separate models, same family. No grafting,
no pairing, no shared serving. Each line trains, evaluates, and ships alone.

## 6. Open questions
Q1: c_proj mismatch — graft (A) or foreign donor (B)? Default B unless probe
proves otherwise. Q2: Main file arch — triage decides. Q3: Ollama standard
seat — frozen until custom lane green.
