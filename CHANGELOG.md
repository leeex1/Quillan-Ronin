# Changelog — Quillan Lineage: Ace v4.2 → Quillan-Ronin → Quillan-ONI

All notable changes to this project will be documented in this file. Format follows Keep a Changelog. **One canonical counter: v5.4.x-oni** (since 2026-08-26). Legacy v8/v9/v10 names are retired to _archive.

## Unreleased — v5.4.0-oni Training Active (2026-09-10)

### Frontier v2 Resume (Active)
- **Resumed 3291→7100** from `quillan_frontier_v2_latest.pt` (1438/1438 tensors) — cold-start `transplant_clean.py` (Qwen + BitNet, **no Mistral weights**) on Llama-derived skeleton → 59.4M corpus pretrain → 78,279 SFT corpus
- **Best 0.9165 @5251** (ladder 1.57@3403 →1.44@3533 →1.29@3994 →1.06@4119 →0.91@5251), avg last100 6.07 — harness fixed (killed thrash, SAVE 10→50, speed 0.05→0.15 st/s), probe `A: The user` → `A: <think` stub (trace format learned)

---

## True Origin — From Prompt to Weights (Pre-Ace)

> **Quillan began as a prompt, not a model.** This is the user-corrected lineage:

1. **Built GPT — Prompt v1:** Custom GPT prompt v1 (early council/prism logic in prompt form)
2. **Custom GPT Limits Hit:** Context, tooling, and platform caps forced beyond prompt-only
3. **Mass Platform Testing for Quillan Initialization:** Gemini, Perplexity, Grok, GPT, Claude — parallel council testing, rituals, memory, 33-persona behavior validated across platforms (see README Connect with the Council)
4. **Prompt and Doc Refinement:** 277+ markdown knowledge vault, Samurai.md (80+ sections), 9-Vector Prism, Formal Papers synthesis — prompt and doc co-evolved
5. **Quillan + Base Models Testing and Distillation:** Tested against base models (Llama, Qwen, BitNet) + teacher endpoints (DeepSeek-V4, distilled corpora) — distillation and routing experiments, `scripts/test_teacher_endpoints.py`, `transplant_clean.py` projection matrices
6. **Trained Identity over Llama Base — Ace Mini:** Fine-tune on `meta-llama/Llama-3.2-3B` → **Ace v4.2-mini 3B** (`ace-v4.2-model.pt`, commit f8dfef7, 2025-09-24) — first realized weights
7. **Original Skeleton Weights Released with Random 000x Tensors:** First skeleton export with random-initialized 000x tensors (placeholder shard weights) — precursor to full skeleton
8. **What We Are Doing Now — Quillan-Ronin → ONI:** Unified skeleton `quillan_v5_4_oni.py`, dense council, BitNet ternary, transplant cold-start, full pretraining and Frontier SFT (from-scratch in compute)

---

## [v5.4.0-oni] — 2026-08-26 — ONI Sovereign Quantum — Canonical Unified Build
**Super-merge of 140 arXiv papers — **Retires v8.1 / v5.3.1. Single source: `09 - Projects/projects/oni/quillan_v5_4_oni.py` (203KB, 3671 lines)**

### Architecture — Original, Research-Inspired, From-Scratch
- **Built from scratch,** synthesizing published research — not a fine-tune. Transplant was cold-start init only; substantive training is full pretraining (59.4M) + SFT (78k) = from-scratch in compute.
- **Donors (weights): BitNet + Llama + Qwen — no Mistral weights.** Transplant used only `Qwen/Qwen3.5-0.8B` (C8–C21 zero-padded) + `1bitLLM/bitnet_b1_58-3B` (C22–C33 sliced) on top of Llama-derived skeleton. **Mistral/Mixtral MoE is conceptual lineage only** — the originator of sparse MoE routing (Fedus et al., Mistral), so any MoE inherits that research lineage, but no Mistral checkpoint was ever merged.
- Throne + C1–C34 dense deliberation, RoPE, Couil hybrid, Recirculation, DistillationHead, ThermoDiffusion, LeeMach6, E_ICE, 10 Quantum Formulas

### Formal Papers 100 percent Wiring (2026-08-29 → 09-04)
- EvoMoE, Mamba, FA3, ProTrain/Memo, DeepOptimizer, DFlash/Speculative, NITRO/PocketNN, ES-at-Scale, RQGM, RealSwarm/WorldModel, Adelic Spectral Zeta + Pascal L2, p-adic router, AVX2 4.42x

## [v5.3.1 hardening] — 2026-06-29 → 2026-07-26 — Saturated Path
- BitNet pre-quant, CouilAttention, AMP, eggroll 16→256, LR fix 5e-4→2e-5 — `train_full_param_v2.py`

## [v5.3.1-final] — 2026-06-10 — Sovereign Ascension
- ASCENSION 50000 wgate patched 186 keys, GGUF 915 MB, verified 50 tokens CUDA fp16 377M params

## [v5.3.1] — 2026-05-29 — Ronin Sovereign
- Global sync, STE restore, seq_len 128, purge corrupt science

## [Samurai v5.3] — 2026-05-25 — Stabilized Repository
- Archive-ready

## [Quillan v4.2] — 2025-11-16 — Council Bugfix
- Attention scaling, shape mismatch, gating backprop, JSONL loader (aebc8c1)

## [Ace v4.2-mini 3B] — 2025-09-24 — Genesis
- LLaMA 3.2 3B identity fine-tune, custom dataset (f8dfef7) — first weights from prompt-era lineage

### Attribution (Corrected 2026-09-10 per user)
- **From-scratch:** Original skeleton + transplant init + full pretraining/SFT = from-scratch build. Not a wrapper or LoRA fine-tune.
- **Donors = BitNet + Llama + Qwen:** BitNet and Qwen contributed transplant weights, Llama contributed the Ace skeleton base. No Mistral checkpoint merged.
- **MoE lineage = Mistral-conceptual:** Sparse MoE as a concept originates with Mistral/Mixtral research — Quillan inherits that lineage intellectually, but implements its own dense persona-pull Council MoE from scratch.
- **Research synthesis:** BitNet (Microsoft), RoPE (Su et al.), MoE routing (Fedus/Shazeer/Mistral), etc. — all implemented originally, not copied checkpoints.

---

## Archived Git History
## 2025-07-21 15:23:00 -0400 | 2c6a6cf
**Author:** leeex1

Initial commit


---

## 2025-07-21 15:40:10 -0400 | 2932d2e
**Author:** leeex1

Update LICENSE

updated license before file upload
---

## 2025-07-21 15:41:45 -0400 | 68e61fd
**Author:** leeex1

add files via upload

added ace files v5.3.1
---

## 2025-07-21 15:56:34 -0400 | dd3af89
**Author:** leeex1

Update README.md

updated the read me file alot
---

## 2025-07-21 15:57:11 -0400 | 478b7da
**Author:** leeex1

Update README.md

edited file
---

## 2025-07-21 15:57:52 -0400 | 27c96c8
**Author:** leeex1

Update README.md


---

## 2025-07-21 16:06:21 -0400 | 0c681ea
**Author:** leeex1

Update README.md

updated read me properly
---

## 2025-07-21 16:16:53 -0400 | c99c393
**Author:** leeex1

additions

adding more files
---

## 2025-07-21 16:18:43 -0400 | 2d7711e
**Author:** leeex1

Delete HLE Questions.txt


---

## 2025-07-21 16:18:55 -0400 | 7b43c00
**Author:** leeex1

Delete HLE Answer sheet.txt


---

## 2025-07-21 16:20:01 -0400 | de59caf
**Author:** leeex1

Update file list for deployments.txt


---

## 2025-07-26 12:04:24 -0400 | 5ea3a71
**Author:** leeex1

these are the additiona files for ace

these enable the actual operation and function of the static text files
---

## 2025-07-26 13:38:32 -0400 | 93074da
**Author:** leeex1

Angela inspiired files

Credit to T81dev for the base module i used to reverse engineer these modules
---

## 2025-07-26 13:39:36 -0400 | 19a575e
**Author:** leeex1

files additions

forgot to add these
---

## 2025-08-02 15:59:55 -0400 | 13f00e1
**Author:** leeex1

update unholy 2.0


---

## 2025-08-03 16:15:00 -0400 | c5ed7e2
**Author:** leeex1

Update README.md

updated with ace-claude written read me
---

## 2025-08-07 03:41:38 +0000 | cdc74e2
**Author:** leeex1

updated files


---

## 2025-08-07 03:42:00 +0000 | ded354f
**Author:** leeex1

k


---

## 2025-08-08 18:31:25 -0400 | 2f6f933
**Author:** leeex1

ollama modelfile template


---

## 2025-08-08 18:54:41 -0400 | df40e8f
**Author:** leeex1

Update Ace system prompt.md

new formatted md
---

## 2025-08-14 19:55:43 -0400 | 9039f02
**Author:** leeex1

Update Ace system prompt.md

new updated prompt format
---

## 2025-08-15 09:40:59 -0400 | 2c8adf8
**Author:** leeex1

Delete Untitled1.ipynb

unneeded file
---

## 2025-08-15 09:44:25 -0400 | e7b1bee
**Author:** leeex1

Delete Ace Prompt.bicep

unneeded
---

## 2025-08-15 09:46:09 -0400 | 99fa319
**Author:** leeex1

Delete Ace Prompt.jinja

unnneeded
---

## 2025-08-15 09:47:02 -0400 | 4eb755a
**Author:** leeex1

Delete mergekit_config.yaml

trancended design
---

## 2025-08-15 09:48:09 -0400 | 9bffe73
**Author:** leeex1

Delete system_prompt.yaml


---

## 2025-08-15 09:49:39 -0400 | 6c4605e
**Author:** leeex1

Delete Ace system prompt.py


---

## 2025-08-15 09:50:22 -0400 | 6fd0867
**Author:** leeex1

Delete .ipynb_checkpoints directory


---

## 2025-08-15 09:51:30 -0400 | eaa8e54
**Author:** leeex1

Delete system_prompt_formatted.yaml


---

## 2025-08-15 09:52:55 -0400 | 94e4f7d
**Author:** leeex1

Delete system_prompt_new.yaml


---

## 2025-08-15 09:53:50 -0400 | 248ad1e
**Author:** leeex1

Delete untitled.md


---

## 2025-08-15 10:02:16 -0400 | 8cb11f1
**Author:** leeex1

Update Ace system prompt.md


---

## 2025-08-15 12:36:31 -0400 | e5bda04
**Author:** leeex1

Update Ace system prompt.md

update
---

## 2025-08-16 13:51:25 -0400 | 538a47c
**Author:** leeex1

added gptprompt and reasoning engine py


---

## 2025-08-16 13:55:52 -0400 | 7db3b92
**Author:** leeex1

Add files via upload


---

## 2025-08-16 14:36:49 -0400 | 92261aa
**Author:** leeex1

Update README.md

updated read me format
---

## 2025-08-16 16:50:58 -0400 | 75408c8
**Author:** leeex1

Update README.md

updated some coming soon features
---

## 2025-08-16 16:58:24 -0400 | 7fe785d
**Author:** leeex1

Update config.json

updated cofig,json with disclaimer
---

## 2025-08-16 17:25:50 -0400 | f9c2191
**Author:** leeex1

added fixed grok prompt


---

## 2025-08-16 21:16:24 -0400 | 71277f6
**Author:** leeex1

Update and rename Modelfile to Ace-v5.3.1_Base_Modelfile

updated modelfile
---

## 2025-08-16 21:19:42 -0400 | 29914e5
**Author:** leeex1

Update System prompts and tone.txt

removed outdated prompt, updated with the new one
---

## 2025-08-16 21:34:22 -0400 | e28ef6b
**Author:** leeex1

Update README.md


---

## 2025-08-16 22:14:17 -0400 | 7dd55d0
**Author:** leeex1

uploading custom gemini prompt


---

## 2025-08-17 10:20:21 -0400 | 9677e82
**Author:** leeex1

Update README.md


---

## 2025-08-17 10:40:03 -0400 | 0f23c0f
**Author:** leeex1

updated modelfile


---

## 2025-08-17 11:11:04 -0400 | 6ebd05c
**Author:** leeex1

added reformated benchmark


---

## 2025-08-17 11:24:17 -0400 | 622ae9a
**Author:** leeex1

fixed formatting error


---

## 2025-08-17 13:25:17 -0400 | 41440d4
**Author:** leeex1

updated llm benchmark


---

## 2025-08-17 13:28:02 -0400 | 04a44fd
**Author:** leeex1

removed test results from test


---

## 2025-08-17 13:40:34 -0400 | 8f74707
**Author:** leeex1

updated fromatting


---

## 2025-08-17 13:42:47 -0400 | c989971
**Author:** leeex1

added line of text


---

## 2025-08-17 13:49:02 -0400 | f0f933b
**Author:** leeex1

fixing llm bench


---

## 2025-08-17 13:55:28 -0400 | ce22041
**Author:** leeex1

updated test


---

## 2025-08-17 15:54:39 -0400 | 974f379
**Author:** leeex1

added test template to file 4


---

## 2025-08-17 15:56:03 -0400 | da09d5b
**Author:** leeex1

format fix


---

## 2025-08-17 15:58:03 -0400 | 1d7d558
**Author:** leeex1

updated file 1 flowchart.md


---

## 2025-08-17 15:58:49 -0400 | 1dc2a60
**Author:** leeex1

fixed formatting on file 1


---

## 2025-08-17 16:02:33 -0400 | 2e400e3
**Author:** leeex1

updated file 8 format


---

## 2025-08-17 17:04:56 -0400 | 87e2a1c
**Author:** leeex1

minor tweaks


---

## 2025-08-17 17:19:58 -0400 | 8ff450e
**Author:** leeex1

updated conciousness template


---

## 2025-08-17 17:21:39 -0400 | ce8fd38
**Author:** leeex1

updated conciousness template


---

## 2025-08-17 17:32:49 -0400 | b3a98fc
**Author:** leeex1

updated multimodal fusion


---

## 2025-08-17 18:23:26 -0400 | 958b5b0
**Author:** leeex1

updated py 9


---

## 2025-08-17 18:24:17 -0400 | c843c35
**Author:** leeex1

update template json


---

## 2025-08-17 18:26:04 -0400 | b2abe99
**Author:** leeex1

updated loader py


---

## 2025-08-17 18:42:06 -0400 | d7a0f17
**Author:** leeex1

updated loader py


---

## 2025-08-17 18:47:59 -0400 | 3791aa4
**Author:** leeex1

updated loader again


---

## 2025-08-17 19:09:52 -0400 | b535316
**Author:** leeex1

updated code executor file


---

## 2025-08-17 19:10:44 -0400 | 8a664cb
**Author:** leeex1

updated creative engine


---

## 2025-08-17 19:12:35 -0400 | f7da74e
**Author:** leeex1

code executor fix


---

## 2025-08-17 19:40:48 -0400 | 73a0ed8
**Author:** leeex1

updated loader


---

## 2025-08-18 13:19:32 -0400 | 311129f
**Author:** leeex1

adding some research papers


---

## 2025-08-19 14:11:00 -0400 | 0b5877a
**Author:** leeex1

fixed guardrail errors for platform compatability


---

## 2025-08-19 14:15:27 -0400 | 21953c1
**Author:** leeex1

updated some read me stuff


---

## 2025-08-19 14:17:56 -0400 | bc06f95
**Author:** leeex1

added simple what is ace at intro of read me


---

## 2025-08-19 14:20:22 -0400 | 6b7282f
**Author:** leeex1

fixed read me


---

## 2025-08-19 16:21:26 -0400 | 93d14fd
**Author:** leeex1

added stats


---

## 2025-08-19 16:22:14 -0400 | 4c9c5dd
**Author:** leeex1

aded stats fix


---

## 2025-08-20 17:08:00 -0400 | fba7184
**Author:** leeex1

fixed small errors


---

## 2025-08-20 20:15:55 -0400 | baf5f1b
**Author:** leeex1

updated media template


---

## 2025-08-20 21:18:42 -0400 | 8880e1c
**Author:** leeex1

uploaded test datasets


---

## 2025-08-20 21:21:11 -0400 | 92b5488
**Author:** leeex1

updated read me


---

## 2025-08-21 01:31:53 -0400 | 87b4de0
**Author:** leeex1

minor updates to read me


---

## 2025-08-21 01:49:18 -0400 | 097d25e
**Author:** leeex1

formatting fix


---

## 2025-08-21 01:56:24 -0400 | 5e1bae4
**Author:** leeex1

format update readme


---

## 2025-08-21 01:58:34 -0400 | 703cbf0
**Author:** leeex1

minor media template update


---

## 2025-08-21 02:14:18 -0400 | 04c1f11
**Author:** leeex1

small tweaks to read me


---

## 2025-08-21 04:30:28 -0400 | 0d8d227
**Author:** leeex1

Create src directory structureCreate src directory with .gitkeep fileCreate .gitkeep

Add src directory to allow organizing files into a directory structure. This includes a .gitkeep file to ensure the empty directory is tracked by Git.
---

## 2025-08-21 04:37:32 -0400 | 2fe23dd
**Author:** leeex1

Create .gitkeep


---

## 2025-08-21 04:38:37 -0400 | 0e79fd1
**Author:** leeex1

Create .gitkeep


---

## 2025-08-21 04:39:22 -0400 | 3169402
**Author:** leeex1

Create .gitkeep


---

## 2025-08-21 04:48:29 -0400 | 450e197
**Author:** leeex1

fixed some directory issues


---

## 2025-08-21 23:59:40 -0400 | 54eef6b
**Author:** leeex1

updaed read me with another user experince from outide me


---

## 2025-08-22 00:57:42 -0400 | 9ebf756
**Author:** leeex1

updated read me more details


---

## 2025-08-22 03:14:02 -0400 | 583654f
**Author:** leeex1

Update README.md

Added test prompts for base model VS Ace + base model
---

## 2025-08-22 11:39:11 -0400 | aff2609
**Author:** leeex1

added newest sucess story


---

## 2025-08-22 11:40:37 -0400 | 1e18543
**Author:** leeex1

updated foramatting in section on read me


---

## 2025-08-22 13:55:58 -0400 | 1def094
**Author:** leeex1

added new sucess story


---

## 2025-08-22 13:59:41 -0400 | 8fd44e6
**Author:** leeex1

updated read me with time stamps for sucess stories


---

## 2025-08-22 16:45:46 -0400 | 5b71661
**Author:** leeex1

added coding ide support will add more tool support as i get feeback on failures


---

## 2025-08-22 16:47:50 -0400 | 9bf0fd8
**Author:** leeex1

added dev tips and warning in caps


