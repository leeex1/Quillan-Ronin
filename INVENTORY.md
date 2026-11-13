# InVENTORY — Quillan-Ronin Ecosystem
_Last verified: 2026-11-13 (local filesystem + `gh` + HF Hub API + owner-supplied URLs)_

Legend: `[local]` = exists on this machine · `[GH]` = on GitHub (leeex1) · `[HF]` = on Hugging Face (CrashOverrideX) · `[web]` = public URL

---

## 1. Models (Hugging Face — CrashOverrideX) + Ollama

| Model | HF repo | Ollama | Role | Downloads |
|---|---|---|---|---|
| Quillan-Ronin (flagship / "OS" line) | [HF](https://huggingface.co/CrashOverrideX/Quillan-Ronin) | [ollama](https://ollama.com/crashoverridex/Quillan-Ronin) | WSP merge of Llama-3.2-1B + Qwen2.5-0.5B + BitNet b1.58-2B-4T; ships the sovereign MoE runtime | 2,241 |
| Quillan-Dragon | [HF](https://huggingface.co/CrashOverrideX/Quillan-Dragon) | [ollama](https://ollama.com/crashoverridex/Quillan-Dragon) | GGUF, Falcon3-1B-Instruct WSP merge | 273 |
| Quillan-Ronin-Mini | [HF](https://huggingface.co/CrashOverrideX/Quillan-Ronin-Mini) | [ollama](https://ollama.com/crashoverridex/Quillan-Ronin-mini) | GGUF, Qwen2.5-0.5B; edge/mobile | 215 |
| Ace_v4.2_Mini | [HF](https://huggingface.co/CrashOverrideX/Ace_v4.2_Mini) | — | Legacy pre-WSP LoRA; lineage only | 0 |

> Note: your list names 3 HF model repos — no separate "OS" repo exists under `CrashOverrideX` (and no Spaces). **Quillan-Ronin is the OS-line repo.** Ace_v4.2_Mini is a 4th, legacy model from the HF API.

## 2. Training & checkpoints (local only, not public)

- [local] `C:\02_QUILLAN\03 - Training & Model\` — full training pipeline (tokenizer `quillan_bpe_tokenizer`, `modeling_quillan.py`, fused optimizer, benchmarks, logs)
- [local] `checkpoints\checkpoints_oni\quillan_coherent_base_ternary.pt` — canonical ternary base (Step-1000 lineage)
- [local] `05_Training\` · `evaluation_results\` · `training_logs\`
- [local] `kernels\quillan_cuda_sm61.dll` + `kernel_bridge.py` — native Pascal sm_61 `__dp4a` ternary kernel

## 3. GitHub (leeex1)

| Repo | What | Local mirror |
|---|---|---|
| [**Quillan-Ronin**](https://github.com/leeex1/Quillan-Ronin) | Main repo; this workspace; GitHub Pages via `docs/` | `C:\02_QUILLAN` |
| [**Validation-test-kit-**](https://github.com/leeex1/Validation-test-kit-) | Harness for benchmark packs | `09 - Projects\projects\testing\Validation-test-kit` |
| [**Quillan-Ronin-chrome-extension**](https://github.com/leeex1/Quillan-Ronin-chrome-extension) | Browser extension | ? |
| [**quillan.cpp**](https://github.com/leeex1/quillan.cpp) | llama.cpp fork + 2-bit ternary (.qbin) export | `09 - Projects\projects\Quillan.cpp` |
| [**metamon-game**](https://github.com/leeex1/metamon-game) | Metamon / Ronin Survival Rogue | `09 - Projects\projects\Metamon-Survival-Rogue` + `_projects\Ronin-Saga-Neo-Eden` |
| [**MCP-DevKit**](https://github.com/leeex1/MCP-DevKit) | Tooling for the MCP server fleet | `09 - Projects\projects\MCP-DevKit` |
| [**muse-open-kernal-research-**](https://github.com/leeex1/muse-open-kernal-research-) | Muse open-kernel research | ? |
| **FreeLatticebykirk** | Kirk's lattice project (fork) | `09 - Projects\projects\FreeLatticebykirk` |
| **Quillan-ide** | Quillan IDE fork | `09 - Projects\projects\quillan_quantum_ide` (+ `quillan-ide` desktop build) |
| **Nextverse-protype-app-** | Nextverse prototype (DeepWiki'd) | `09 - Projects\projects\Nextverse-protype-app--main` |
| **GitNexus / llama.cpp / BitNet / mempalace / godot-demo-projects / hermes-agent / void / kilocode / claude-code / Kimi-K2 / Qwen-Agent / free-programming-books / x-algorithm / ai-knowledge-graph / MazeBenchEngine / harness2 / t81-foundation / erdos-ryan-petty-recomendation / hans-sovereign-consciousness-initiative / Matrix-Game** | Forks & upstream mirrors | mixed |

## 4. Pages, landing, community

- [web] **Custom landing page** — https://digitalroninx.grok.me/
- [web] **GitHub Pages** — https://leeex1.github.io/Quillan-Ronin/ (served from `docs/`; `.nojekyll`)
- [web] **X / Twitter** — https://x.com/Crashoverride_X
- [web] **Discord** — https://discord.gg/JcFqvWKkRv
- [web] **YouTube** — https://youtube.com/@JDXX
- [web] **Suno** — https://suno.com/@Crashoverride_X
- [web] **Genius (JDXX · Quillan-Ronin)** — https://genius.com/artists/Jdxx · https://genius.com/artists/Quillan-ronin
- [web] **GoFundMe** — https://gofund.me/a0667596d

## 5. Knowledge bases / wikis / notebooks

- [web] **DeepWiki — Quillan-Ronin** — https://deepwiki.com/leeex1/Quillan-Ronin
- [web] **DeepWiki — Nextverse (formula validation suite)** — https://deepwiki.com/leeex1/Nextverse-protype-app-/6.1-formula-validation-suite
- [web] **Grokipedia — Multi-agent frameworks** — https://grokipedia.com/page/Multi-agent_frameworks
- [web] **Grokipedia — Quillan-Ronin** — https://grokipedia.com/page/Quillan-Ronin
- [web] **NotebookLM — Quillan-Ronin research foundation** — https://notebook.google.com/notebook/aeebd738-c44f-483b-bcfc-c1ed51633c25
- [web] **NotebookLM — Quillan arch breakdown** — https://notebook.google.com/notebook/8ac3819a-627b-4ed1-820d-b803bbfd89f7
- [web] **NotebookLM — Books breakdowns** — https://notebook.google.com/notebook/6840cdd3-a827-44d7-8b43-35b4fb387232
- [local] `.deepwiki\wiki.json`, `MemPalace` wing `02_quillan` (22,142 drawers: `quillan_knowledge_base` 12,572 · `frontend` 6,045 · `general` 3,233 · `checkpoints` 32 · `mcp` 16)

## 6. Hugging Face datasets (CrashOverrideX)

| Dataset | Content | Downloads |
|---|---|---|
| [QuillanTrainingdata](https://huggingface.co/datasets/CrashOverrideX/QuillanTrainingdata) | Training corpus | 26 |
| [Quillan_Samurai_sets](https://huggingface.co/datasets/CrashOverrideX/Quillan_Samurai_sets) | Samurai training sets (security/CTF, defensive) | 203 |
| [quillan-lyrics-corpus](https://huggingface.co/datasets/CrashOverrideX/quillan-lyrics-corpus) | Lyrics/prompt corpus for the music label | 186 |
| [quillan-audio-media](https://huggingface.co/datasets/CrashOverrideX/quillan-audio-media) | Canonical audio for the portal stream | 334 |

## 7. Creative production

- **Book series** (Hybrid Throne saga, on Amazon) — in order of release:
  1. [Twisted Destiny: Awakening of the Hybrid Throne](https://amazon.com/Twisted-Destiny-Awakening-Hybrid-Throne/dp/B0HJ7LQW79)
  2. [Rise of Ascension: Power of the Hybrid Throne](https://amazon.com/Rise-Ascension-Power-Hybrid-Throne/dp/B0HJ5SXZ99)
  3. [Battle Grandeur: Multi-versal Hybrid Throne](https://amazon.com/Battle-Grandeur-Multi-versal-Hybrid-Throne/dp/B0HJ4747DZ)
  4. [Fall of Empires: Resolution of the Hybrid Throne](https://amazon.com/Fall-Empires-Resolution-Hybrid-Throne/dp/B0HJ46XVGW)
  5. [Howling Shadow: Tomorrow of the Hybrid Throne](https://amazon.com/Howling-Shadow-tomorrow-Hybrid-Throne/dp/B0HJ3VCB2T)
  - [local] `05 - Creative Works\Book Series` (manuscripts, covers, audiobook pipeline `convert_to_audiobook.py`)
- **Music label / audio** — [local] `05 - Creative Works\Audio Engineer\Audio files` (~300 tracks, RouteNote exports) · [HF] `CrashOverrideX/quillan-audio-media` · Genius (Jdxx / Quillan-Ronin) · YouTube @JDXX · Suno @Crashoverride_X
- **Images / NFT / art** — [local] `05 - Creative Works\images\concept art`, `06_Media\`, `gallery\`, `main-images\`, `assets\concept_art` · [GH] `docs/assets/nft_images/`, `docs/gallery/` (many hash-verified duplicate mirrors — see cleanup section)
- **Godot fork (game engine upgrade)** — [local] `09 - Projects\projects\godot-ronin-engine` (`addons/` + `mcp/` + `skills/`) — the Quillan-enhanced Godot fork
- **Chip design / custom silicon** — [local] `09 - Projects\projects\Chip design` — SM61 CUDA kernels + schematics (`quillan-oni-v-custom-cpu-chip-design.svg`)

## 8. Games & prototypes

- [local] `09 - Projects\thronglets-ronin` — Godot game (thronglets)
- [local] `09 - Projects\projects\wayward-ronin\wayward-ronin` — Wayward Ronin prototype (nested clone)
- [local] `09 - Projects\projects\_projects\Ronin-Saga-Neo-Eden` — Ronin Saga / Neo Eden (Metamon survival rogue)
- [local] `09 - Projects\projects\Metamon-Survival-Rogue`
- [local] `09 - Projects\projects\quillan-swarm-files` — the 35 swarm subagent specs

## 9. Publications

- [local] `10 - Formal Papers\` — WSP paper + figures + Executive Summary
- [ext] **Zenodo** — `?` DOIs still needed (you said "zenodo papers" — the links didn't come through with the other URLs)

## 10. Validation & QA

- [local] `validation_toolkit\` — master suite (25/25 formulas · 7/7 BitNet/checkpoint · 6/6 standalone gates — 100% PASS)
- [local] `09 - Projects\projects\testing\Validation-test-kit` — benchmarks/token validation

## 11. Open items

- Zenodo DOIs for the papers
- Whether `quillan.cpp`, `quillan-ide`, `MCP-DevKit`, `muse-open-kernal-research-`, `Quillan-Ronin-chrome-extension` should be pushed to GitHub (remotes not all configured)
- 4th NotebookLM notebook (your list named 3)

## 12. Known duplicate locations (pre-consolidation, hash-verified)

- Media mirrors: `images\`, `06_Media\`, `gallery\`, `main-images\`, `assets\` vs `docs\…` — originals in `05 - Creative Works`, copies in `docs/` for Pages. Being consolidated now: redundant copies quarantined to `E:\Quillan-dups-quarantine-20261113\` (~6.2 GB, reversible).
- IDE extension installs: `.devin/`, `.windsurf/`, `.antigravity-ide/`, `.gemini/` — per-agent duplicates of identical extension builds
- Platform knowledge working copies: `00-Meta_Platforms*` / `_archive\duplicates_knowledge` — intentional platform copies + an archive of pasted duplicates
