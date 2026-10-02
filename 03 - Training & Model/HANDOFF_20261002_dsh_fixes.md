# HANDOFF — 2026-10-02 — CPT + eval repairs (DeepSeek Harness session)

**Scope of this change set:** make Continued Pre-Training runnable again after the
Oct-1 vault reorganisation, widen the trainable set so attention can learn syntax,
add a real held-out validation metric, and fix the benchmark that was reporting
word salad. Two scripts changed. Nothing else was touched.

**Backups of every modified file:**
`<script>.py.bak_20261002_dsh` in `03 - Training & Model\scripts\`

---

## 1. Why the models looked broken (root cause, verified)

`run_mini_clean_forward_eval.py` contained:

```python
out = block(x, layer_past=None, use_cache=False,
            gov_scale=None, token_ids=input_ids)
# ...wrapped in `except Exception: pass`
```

`UnrolledTransformerBlock.forward` is defined as
([quillan_v5_4_oni.py:1891](../../../02_QUILLAN/03%20-%20Training%20&%20Model/scripts/quillan_v5_4_oni.py)):

```python
def forward(self, x, layer_past=None, use_cache=False, gov_scale: float = 1.0):
```

**There is no `token_ids` parameter, and `gov_scale` must be a float, not `None`.**
So every block call raised `TypeError`, and the bare `except: pass` swallowed it.
**No transformer block ever executed.** The "model" collapsed to
`wte -> ln_f -> lm_head` — an embedding table plus an output projection.

That is the entire explanation for the word-salad text in
`clean_eval_quillan_{6l,12l}_clean_sft.md`. Those two reports are **void** and are
preserved as `*.md.BROKEN_evidence`.

**Consequence: the 6L and 12L models were never actually benchmarked.** The real
generation evidence we do have (`checkpoints/heldout_eval_report.json`, 27 Sep)
shows the 6L producing correct answers, e.g.:

- "chemical symbol and atomic number of Gold" -> `Au ... Atomic Number: 79`
- "If all A are B and all B are C..." -> `Yes, all A are necessarily C. Proof by Transitivity of Subsets...`
- photosynthesis -> correct balanced equation `6H2O + light energy -> C6H12O6 + 6O2`

So the models work at a basic level. Treat them as **early and under-trained, not
broken**.

---

## 2. Why the CPT only reached PPL ~14 (the actual lever)

`run_quillan_cpt_training.py` auto-enabled a freeze whenever `--device cuda`:

```python
if args.freeze_base or (args.device.startswith("cuda") and args.freeze_base is None):
    if any(k in n for k in ("expert", "router", "ln", "norm", "gate", "w_gate")):
```

The frozen set therefore included **`c_attn`, `c_proj` (all of self-attention) and
`wte` (the entire embedding table)**. Attention is where syntax lives — which is
exactly the observed failure mode: correct topic content with repetition and drift.

Also, the 500-step run at batch 2 x seq 256 consumed only
**500 x 512 = 256,000 tokens** out of a 10,000,128-token corpus — **~2.6% of a
single epoch**. That is why it plateaued, not the architecture.

---

## 3. Changes applied

### `run_quillan_cpt_training.py`

| Change | Detail |
|---|---|
| **Paths repaired** | All checkpoint/data/script paths were pre-reorg (`C:\02_QUILLAN\checkpoints`, `\training_data`, `\scripts` — all three no longer exist). Now resolved via `MODEL_DIR = 03 - Training & Model` with a legacy fallback. **The script could not run at all before this.** |
| **Baseline auto-resolution** | `quillan_6l_ma_best.pt` no longer exists. Now walks a candidate list (`quillan_6l_cpt_best.pt` -> `quillan_6l_clean_sft.pt` -> `quillan_6l_sft_v2.pt` -> `quillan_6l_ma_best.pt`). Override with `--init-ckpt`. |
| **Widened unfreeze** | New `--unfreeze-set {wide,legacy,full}`, default **`wide`** = experts+routers+norms+gates **+ attn/c_attn/c_proj/wte/prism/bridge/finalizer/lora**. `legacy` reproduces the old behaviour. |
| **fp16 frozen weights** | New `--half-frozen` (default ON). Frozen weights stored in fp16, freeing ~1.6 GB. Same trick already used in `train_expert_cycle.py` (lines 76-78). |
| **Autocast + GradScaler** | Required once frozen weights are fp16. Automatically enabled with `--half-frozen`. |
| **Gradient checkpointing** | New `--grad-checkpoint` (default ON). `cfg.grad_checkpoint` was already implemented in the model but never switched on. |
| **Held-out validation** | New `--val-fraction` (default 0.05), `--val-interval`, `--val-batches`. **Checkpoints are now selected on held-out loss, not training loss.** Previously "best" meant "most memorised". |
| **VRAM estimate logged** | Prints estimated optimizer+frozen footprint and warns if it approaches the 4 GB card limit. |
| **Default steps 500 -> 5000** | 500 was ~2.6% of one epoch. |
| **Grad clip restricted** | Now clips trainable params only (was clipping all). |

### `run_mini_clean_forward_eval.py`

| Change | Detail |
|---|---|
| **`clean_forward` rewritten** | Correct positional call `block(x, None, False, gov_scale)`; real float `gov_scale` from `model.governor.current_scale`; **no silent `except`**; mirrors the model's ingestion gating; **asserts every layer executed**. |
| **Paths repaired** | Tokenizer / checkpoints / output dir were pre-reorg. Now resolved with fallback. |
| **`strict=False` -> loud failure** | A partial load now raises with the missing/unexpected key names instead of silently benchmarking a stub. |
| **Device default = CPU** | A 3.0-4.7 GB fp32 checkpoint cannot fit the 4 GB card (desktop already holds ~1.4 GB). Override with `QUILLAN_EVAL_DEVICE=cuda`. |
| **Dynamic header** | No longer prints "QUILLAN 6L" when evaluating the 12L. |

---

## 4. How to run

```powershell
$PY = "C:\Users\Admin\AppData\Local\Programs\Python\Python314\python.exe"
$S  = "C:\02_QUILLAN\03 - Training & Model\scripts"
Set-Location $S

# 1) Honest benchmark first (CPU, ~5-15 min). Writes to 03 - Training & Model\evaluation_results\
& $PY -u "$S\run_mini_clean_forward_eval.py" "C:\02_QUILLAN\03 - Training & Model\checkpoints\checkpoints_oni\quillan_6l_cpt_best.pt"

# 2) Then the real CPT run with attention unfrozen
& $PY -u "$S\run_quillan_cpt_training.py" --model mini --steps 5000 --batch-size 2 --seq-len 256 --device cuda
```

**Watch the startup log line** — it prints trainable/frozen counts and an
estimated VRAM figure. If it warns about exceeding ~3.7 GB, either keep
`--half-frozen` on, or drop to `--unfreeze-set legacy`, or lower
`--batch-size` / `--seq-len`.

**Escape hatches if something breaks:**
- dtype mismatch on the forward -> `--no-half-frozen` (runs fp32, needs more VRAM)
- OOM -> `--unfreeze-set legacy` or `--batch-size 1`
- want the old behaviour exactly -> `--unfreeze-set legacy --no-half-frozen --no-grad-checkpoint`
- restore a file -> copy `<script>.py.bak_20261002_dsh` over it

---

## 5. NOT fixed (open items, honest list)

1. **Generation has no KV cache.** `clean_forward` recomputes the whole sequence
   per token, hence 1-4 tok/s. Routing through `model.generate()` / `use_cache=True`
   would be a large speedup. Left alone deliberately (risk).
2. **The held-out eval** that produced `heldout_eval_report.json` still has two bugs:
   every prompt is scored a constant `0.85 / COHERENT_SYNTAX` (including pure word
   salad), and state leaks between prompts (the quantum question returned the
   E=mc^2 answer). That script was not located in this pass.
3. **`bitsandbytes` is not installed**, so 8-bit Adam is unavailable. fp16 frozen
   weights is the memory lever used instead. `pip install bitsandbytes` would add
   another ~5 GB of headroom.
4. **73 scripts still reference the dead `C:\02_QUILLAN\scripts` root**, including
   `train_frontier_capability.py`. Not touched.
5. **`quillan.cpp` does not load weights.** `quillan_model_load` reads the
   magic/version/config header then returns without reading a single tensor, so
   `quillan-cli.exe --model X.qbin` reports ~21,000 tok/s of noise. Its
   "CCRL Safety Gate: PASSED" line is unconditional.
6. **The fluency gate cannot fail**: `run_fluency_evaluation.py` passes on
   `len(resp) >= 30 and unique_ratio > 0.35`, which gibberish scores *higher* on
   than prose. `teacher_tail_fluency_report.json` says `overall_passed: true` on
   template loops that never close `</think>`.
7. **`git status` has ~5,644 uncommitted entries.** Reorg never committed. The
   `.bak_20261002_dsh` files are new and untracked.
8. **The 343 MB source corpus is not on disk.** Only the capped 10M-token
   derivative. HF `CrashOverrideX/Quillan_Samurai_sets` is public and holds
   `quillan_corpus_CLEAN_V7.pt` (1.33 GB, pre-tokenized) plus ~3.2 GB total.

---

## 6. Reference figures (measured, not quoted from docs)

| Fact | Value |
|---|---|
| 6L total params | **879,793,392** (config `n_layer=6, hidden_dim=1024, 34 experts`) |
| 12L total params | **1,331,636,046** (`n_layer=12, hidden_dim=1024`) |
| Router mode in both checkpoints | **`topk`** (not `dense_pull`) |
| CPT trainable (old GPU freeze) | 67,695,527 of 879.8M -> **812M frozen** |
| CPT run | 500 steps, batch 2, seq 256, **157.6 min, ~27 tok/s**, PPL 639 -> 14.49 |
| Corpus actually consumed | 256,000 of 10,000,128 tokens (**2.6% of one epoch**) |
| GPU | GTX 1050, sm_61, 4096 MiB, ~2,641 MiB free, driver 582.78 |
| CPU / RAM | i5-7500 4c/4t; 27.87 GB total, ~16 GB free, 2400 MHz mixed DIMMs |
| Disk | C: 476 GB SSD (29.9 free); E: 46.9 GB exFAT (37.8 free); Disk 1 = 3.9 TB **USB** |
| Measured trained-vs-garbage evidence | `triage_all.txt` (15 Sep): 7 checkpoints all `COLD`, mean loss 7.52-12.17 |
