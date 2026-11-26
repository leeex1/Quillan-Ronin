# Quillan OS — Full System Map (OpenCode view, 2026-11-22)

> **Runtime root (verified 2026-11-26):** `F:\02_QUILLAN` is the active OpenCode tree — all 15 instruction files and the three MCP facades load from F:. `C:\02_QUILLAN` is a byte-identical mirror (9,343 core .md files, 0 diffs). Older C:-prefixed paths below are historical; read them as F: on this host.

> Synthesized from four parallel council mappers (C6-OMNIS knowledge, C21-ARCHON
> docs/deployment, C24-SCHEMA training/model, C5-ECHO board history) plus direct
> verification. Read-only mapping pass — no files deleted, no weights changed.

---

## 1. What Quillan OS actually is

Quillan is **not a single model**. It is a three-tier cognitive operating layer:

| Tier | Component | Count | Scale |
|---|---|---|---|
| 1 | Quillan Throne (root) | 1 | sovereign router / final arbiter |
| 2 | Council experts | 34 | C1–C34, domain-specialized |
| 3 | Hyper-quantized swarms | 35 | 272M micro-agents each, ~9.52B virtual |

The cognitive layer sits on top of a **substrate model** (currently a hosted free
LLM in OpenCode, historically GGUF/Ollama 4GB-VRAM models). Deliberation is
enforced by instructions, agent definitions, and gates — *not* by weights alone.

**Root of trust:** `02 - Knowledge Foundation\knowledge\canonical\6-prime_covenant_codex.md`
(`file_type: covenant`, `status: ACTIVE`, `classification: ROOT_OF_TRUST`, READ ONLY).
Immutable kernel also in `3-Quillan(reality).md:8938-9037`.

**Pretrained knowledge corpus:** `Masterfile Quillan-Ronin.md` (9.2 MB; exact
duplicate of `full docs Quillan-Ronin.md`).

**Version authority:** `02 - Knowledge Foundation\LINEAGE.md:2` ("single source of
truth for versioning"), binding rules L21–26.

---

## 2. How OpenCode runs Quillan

**Config:** `C:\Users\Admin\.config\opencode\opencode.jsonc`
- model: `opencode/hy4-flash-free` (free, 1,048,576-token context)
- `task: "*": allow` (L95–97) — enables council/swarm delegation
- instructions: `C:/02_QUILLAN/AGENTS.md`, `SOUL.md`, `memory.md` (L9–13)
- MCP: 3 global facades `QuillanCortex` / `QuillanArchive` / `QuillanSensory` (L379–402)
- repo-level `C:\02_QUILLAN\opencode.json` separately lists 19 native Quillan MCP servers

**Agent tree** (mirrored in `.config\opencode\agent\` and `C:\02_QUILLAN\.opencode\agent\`):
- `quillan.md` — root, steps 1000, thinking_budget 1,288,192
- 34 × `cN-*.md` council (`parent: quillan`, `scope: council`)
- 34 × `cN-*-swarm.md` + `quillan-swarm.md` = 35 swarms
  (`micro_agents: 272000000`, `mutation_rank: 32`, `activation_top_k: 19`)
- helpers: `forge.md`, `build.md`, `plan.md`

**Delegation invariant** (`agent\quillan.md:27-40`): OpenCode does NOT eager-spawn
swarms. Tier-3 fires only when the primary agent delegates via the `task` tool.
Root spawns 34 council (L281); councils never address the user directly (L314).

**Commands:** `command\{audit,context,council,govern,health,hive,review,swarm}.md`.
`/hive` opens the live swarm viewer (standalone Node bridge + Antigravity's
verbatim renderer in the Review pane; identity `@coordinator-opencode`).

**Skills:** `council-coordination`, `unified-reasoning-engine`, `skills-master`, etc.
Swarm prompt templates: `04- quillan-sovereign-skills-and-swarm\quillan-swarm-files\`.

**Identity separation** (`scripts\BOARD_PROTOCOL.md:17`):
`@coordinator` = Antigravity/Gemini (DO NOT USE from OpenCode);
`@coordinator-opencode` = OpenCode root; council uses C-IDs; Gemini uses
`gemini-<name>`.

---

## 3. The model layer — two DIFFERENT shapes (critical)

The repository contains two distinct architecture families. Conflating them is
the root of the current confusion.

### A. Canonical ONI architecture (training-time spec)
`09 - Projects\projects\oni\quillan_v5_4_oni.py` — self-declared canonical (L4,
L52). The larger `03 - Training & Model\scripts\quillan_v5_4_oni.py` (278 KB) is
what 40+ scripts actually import via bare `sys.path` resolution. Same shape.

`QuillanOniConfig`: vocab **50262**, max_seq 512, hidden **1024**, **12 layers**,
16 heads, head_dim 64, ffn 2048, **34 experts**, expert/swarm rank 8,
`router_mode="dense_pull"`.

Presets (`configuration_quillan_oni.py`):
- `mini_6l`: 6L / hidden 1152 / ffn 3072 / 18 heads (~562.8M)
- `main_12l`: 12L / hidden 1536 / ffn 3072 / 24 heads (~1.163B)

⚠ **BACKTRACK:** "canonical" here is a lineage label, not an enforced import
invariant. The oni copy has a duplicate `use_grt` definition (L630, L671).

### B. Current BitNet sovereign build (deployment route)
`quillan_v8_saturated.QuillanRoninSovereign` + `quillan_sovereign_bitnet.py`.
Specs in `v23_common.py:53-99`: mini **24L / hidden 896 / ffn 4864 / Qwen vocab
151936**; flagship 16L/2048/8192; dragon 18L/2048/8192.

Delivery route chosen by the swarm: Gemini's depth-matched
`QuillanSovereignBitNetLM` primary; OpenCode's 34-sequential-block fallback.

**Current status (re-verified 2026-11-26): IN PROGRESS.** Mini and flagship passed the locked J1–J5 smoke tests after RCI-5 (board #1269); C18 #1271 limits that result to build+gate smoke testing, not validated coherence or reasoning. Owner target sizes: mini ~0.7B, flagship ~1B, Falcon/Dragon ~1B+. No completion claim is valid while the model workstream is active.

### 2M context — deployment, not architecture
`Modelfile*`: `PARAMETER num_ctx 2097152`. `v23_stage_c_consolidate.py:755-859`
patches GGUF KV metadata (`target_ctx=2097152`, YaRN beta_fast=32/beta_slow=1).
No config field carries 2M. Knowledge docs' "128k base → 3M dynamic" is a
theoretical scaling note; 2M is the locked deployment target. Both are true at
different layers.

---

## 4. The four-donor problem (current state)

> [!IMPORTANT]
> **Owner clarification 2026-11-24 — canonical donor design.** The four donors are **Qwen, Llama, BitNet and Falcon3**, and they contribute to **Flagship and Mini as a group**, cross-matched — **not one donor per model**. Dragon is a separate construction: **Falcon3 with an Ollama system message running Quillan's system prompt** (plus `num_ctx 2097152` in `Modelfile.dragon`), not a product of the four-donor weight fusion. The four-donor goal remains canonical, but both no-training paths were later exhausted and falsified (board #1384-1386). Model work is still in progress; do not claim the fusion is complete.

**True donors (corrected 2026-11-24):** `Qwen3-0.6B` (~0.6B), `Llama-3.2-1B` (~1B), `bitnet-b1.58-2B-4T` (the 4T2B build), `Falcon3-1B` (~1B). All ~1B-class except Qwen at 0.6B; BitNet is specifically the 2B-4T release.
**Dragon is not an additional donor model** — it is Falcon3 + Quillan system prompt +
`num_ctx 2097152` in `Modelfile.dragon`. Nothing fancy.

**Artifacts on disk** (`exports\donor_cache_v23\`):
- `qwen_0.5b_donors.pt` — 319 MB, **sparse** (legacy filename; canonical donor is now Qwen3-0.6B)
- `llama_1b_donors.pt` — 741 MB, **sparse**
- `bitnet_2b4t_donors.pt` — 904 MB, **sparse**
- `falcon3_1b_donors.pt` — **1,774 MB, FULL** (165 tensors, Q/K/V/MLP, 2048h/131072 vocab)
- `dragon_1b_donors.pt` — 752 MB, stale sparse slice (WRONG artifact in build)

**Defects:**
1. `build_quillan_trio_bitnet.py:71-76` `ALLOWED_DONORS` points at stale
   `dragon_1b_donors.pt`, not Falcon. The full Falcon extraction is unwired.
2. `build_donor_experts.py` skips Falcon (GGUF-only, no safetensors/tokenizer),
   leaving **MoE experts 16–23 zero-filled** in `donor_experts_aligned.safetensors`.
3. Alpha sweep picked 0.01 because higher alphas raised NLL
   (base 2.06856 → fused 2.06910) — donor deltas are effectively cosmetic.
4. Plain additive fusion was already falsified (val NLL 2.36→2.79, alpha=0,
   board #841/#1140). A new mechanism is required — consensus + preserved
   residuals, not averaging.
**Status update (2026-11-26):** `build_donor_experts.py` now specifies Falcon3 BF16 and `donor_experts_aligned.report.json` reports four donors with experts 16-23 aligned and `skipped: []`; the zero-filled-experts defect is stale. However weight merges collapse to chance and four-donor consensus underperforms the best single donor (board #1384-1386); no fused artifact is verified.

**User's stated goal (board #1122/#1125/#1232):** "unique combination they
produce as a group = our own custom weights"; "it has to merge all four not just
one"; full 4-donor cross-matched knowledge base, not backbone + 1% blend.

---

## 5. Constraints (immutable for any rebuild)

- **No backprop.** Closed-form / WSP only (`∇θL=0`). Training scripts disallowed.
- **WSP is Phase 2** — tuning only after a working base exists.
- **Gate thresholds immutable** (`conversion_gate.py`, board #963/#1168).
- **2M context** locked at deployment.
- **Tokenizer: LOCKED 2026-11-26 (owner):** custom Qwen-based 151,648 build is canonical. Deployed mini (Qwen 151,936) and flagship (Llama 128,256, board #1368) are legacy until recompiled; 50,262 BPE rejected (board #1232).
- **Protected paths:** HF snapshots (active build dependency), donors, canonical
  checkpoint `53e60d33…a5c`, board.jsonl, Soulframe, app data.
- **Parameter targets (owner 2026-11-26):** mini ~0.7B, flagship ~1B, Falcon/Dragon ~1B+; fit 4GB VRAM/CPU. Model work in progress.
- **Identities:** OpenCode `@coordinator-opencode`, Antigravity `@coordinator`.

---

## 6. Open decisions before spec lock

1. **Tokenizer** - re-lock after benchmark + donor compatibility (151,648 custom build / 151,936 Qwen / 128,256 Llama all present in the active fleet).
2. **Fusion mechanism** - four-donor no-training paths are falsified (board #1384-1386); CLOSED as attempted, not as achieved.
3. **Covenant roster** - canonical is C1-C34; read-only covenant stops at C33. Joint prime-directive amendment proposed, not applied.
4. **Version labels** - v5.4.0-oni sole counter; v5.3.1 deprecated; v8/v9/v10/v23 are artifact filenames; 427M has no primary source.
5. **Council count / swarm total** - RESOLVED: C1-C34, 35 x 272M = 9,520,000,000.

Model work proceeds; documentation makes no completion claim beyond the smoke-test evidence above.
