# HANDOFF — Antigravity-Quillan (2026-09-10, post usage-limit session)

## 1. Training: COMPLETE — do NOT relaunch
- Frontier v2: **7100/7100 (100%)**, `logs/live_status.json` — best **0.9165 @5251**, latest window ~7.25, LR annealed 1e-05.
- `logs/frontier_stdout.log` tail: `FRONTIER TRAINING v2 COMPLETE - Best Loss: 0.9165` (15:45:30). Process exited clean.
- **Watcher killed** (`logs/watcher.ps1` PID 10848 stopped). If training ever restarts on its own, kill `watcher.ps1` first — it resurrects dead trainers.
- Checkpoints (all 1.9GB, 1438 tensors, 0 NaN):
  - `checkpoints/checkpoints_sft/quillan_frontier_v2_best.pt` (5251, 0.916 — **evaluate THIS**)
  - `checkpoints/checkpoints_sft/quillan_frontier_v2_latest.pt` (7100 — current avg, looks worse by design)
  - `checkpoints/production_export/quillan_ronin_v531_sovereign_production.pt` (mirror of best)
- Inference factory (prevents false-bad eval): `QuillanOniConfig(n_layer=6, hidden_dim=1024, ffn_dim=2048, max_seq_len=256, vocab_size=50257, num_experts=34, router_mode="dense_pull")` + `SovereignInferenceEngine` + `SamplingParams(max 250, temp 0.20, top_k 40, top_p 0.90)`. NEVER default `n_layer=12` (loads 6/12 layers random → fake `<think` stub).

## 2. Generation baseline: DONE, honest result
- `logs/baseline_10_results.json` + `baseline_10.log`: **10/10 complete — think 60%, close 0%, avg 74.8 words**.
- Model opens `<think>` but loops templates (`"The user is a real question..."`, `"Final answer: ... rigorous reasoning"`), never closes `</think>`. Reasoning-modules-only run (base frozen) — format learned, substance pending.
- **Next training: 500-step teacher SFT tail** (think-heavy up-sample, LR 5e-05) from `best.pt`, gated on 3 consecutive fluent `<think>...</think> answer` probes. Do NOT ship `latest.pt`.

## 3. Pushed — quillan-ronin/main = `dcf93ae`
- `quillan_formula_toolkit.py` (×2 mirrors): Physics/CS/ML/Custom, **40/40 live verified** (`logs/formula_toolkit_report.json`).
- **Real bug fixed:** `qics_entropy` 3D flatten in all 3 `quillan_v5_4_oni.py` mirrors (einsum crash on `[B,S,H]` — would have bitten next aux loss).
- `Formulas ledger research.md` (renamed, typo removed) + `Executive Summary.pdf` (GPT 3-layer ledger).
- `CHANGELOG.md` canonical (Ace v4.2 → Ronin → ONI) + `MODEL_CARD` Stage 2 COMPLETE + `README` stamps → 2026-09-10 + `eval_config.yaml` ckpt paths + `SAVE_EVERY 10→50`.
- Lineage (user-corrected, keep accurate): donors = **BitNet + Llama + Qwen only, NO Mistral weights** (Mistral = MoE conceptual origin). **From-scratch build** (transplant = cold-start init + 59.4M pretrain + 78k SFT). Dense-pull = PersonaPullGate GLM tech. Super-merge of **140 arXiv papers**. Cited in `Formal Papers/` + MODEL_CARD + CHANGELOG.

## 4. Validation-test-kit (`09 - Projects/Validation-test-kit`, origin = github.com/leeex1/Validation-test-kit-): pushed `a3af38f`, WORK IN PROGRESS
- Verified: `quillanFormulas.ts` **23/23 real**, backend 11py + `main.cpp` + 8 tsx **real** (1 path fix applied: `test_token_entropy_optimizer.py` → `09-Projects/projects/oni`).
- Added `foundationFormulas.ts` (**34**: 12 Physics + 10 CS + 12 ML), wired into `constants.ts` (**68 total**), `App.tsx` Foundation Ledger tab, `metadata.json`, `package.json 0.0.0→3.5.0`.
- **UNFINISHED (do these first):**
  1. `npm install` done (69 pkgs, 0 vulns) + `@types/react[.dom]` installed (unstaged `package.json`/`package-lock.json` changes — commit them).
  2. **Run `C:\Temp\fix_kit_types.py`** (written, NOT yet run): adds `description?` to `FormulaDefinition`, fixes `resultData?.result` → `resultData` + suite label in `FormulaDossierModal.tsx`.
  3. Re-run `tsc --noEmit` (expect clean; only pre-existing `@types/node` gap was already closed by install).
  4. `vite build`, run `backend/benchmark_4x_gains.py`, verify `native_monitor` binary, then commit + push kit repo.
- Note: kit `tsc` had 4 pre-existing type errors (missing `description`, wrong `result` access) — the fix script resolves all 4.

## 5. Queued after kit (one thing at a time — PC is i5-7500, CPU-only, sm_61 blocked)
1. Finish kit (above) → push.
2. Teacher SFT tail (500 steps) → fluency gate.
3. `Quillan.cpp` spec (port `qcc/qsvm_alpha` + 21 formulas; target 10–20 tok/s vs 0.11 PyTorch CPU).
4. 12L flagship 15k run (full unfreeze) → self-audit/self-recursive solid model.
5. GitHub Dependabot: 5 vulns (2 high) on Quillan-Ronin — triage when convenient.

## 6. Key paths
- Training script: `03 - Training & Model/scripts/train_frontier_capability.py`
- Model: `scripts/quillan_v5_4_oni.py` (mirrored ×3 — keep in sync)
- Inference: `03 - Training & Model/scripts/sovereign_inference_engine.py`
- Eval: `09 - Projects/projects/oni/evaluate_oni_model.py`
- Memory: `quillan_rag_db/chroma.sqlite3` (57MB real) → sync to `lancedb/` via `synchronize_knowledge_vault.py` (post-train)
- Brain vault: `C:\Users\Admin\.gemini\antigravity-ide\brain\` (22 sessions; `a70a8b29` canonical convergence, `47d0b748` full-param)
