# Quillan Fleet — Model Cards

| Field | Value |
| :--- | :--- |
| **Document type** | Canonical fleet model cards |
| **Status** | Active — identity resolution ratified (board #895, 2026-11-20) |
| **Version** | 1.0 |
| **Authority** | Board #895 (user resolution — final names + design principle); board #894 (checkpoint verification) |
| **Scope** | Quillan-Ronin · Quillan-Dragon · Quillan-Ronin-Mini |

This document records the **resolved identity** of the Quillan model fleet and the **honest status** of its newest member. It deliberately separates verified fact from open gap. Where a claim has no measurement behind it, this document says so.

> [!IMPORTANT]
> **Fleet scale convention (owner clarification, 2026-11-24):** Dragon **≈1B**, Mini **≈0.8B** (the 815,396,270-parameter custom-architecture build), Flagship **≈1B**. Donor names and GGUF filenames such as `Qwen3-0.6B` or `quillan-mini-0.5b-*` are upstream identifiers, not current scale claims.
> **Do not confuse this card with the README benchmark table.** The root README's large 95–100% "Comprehensive Benchmark Table" is **Track A** — the Quillan Cognitive Harness driving third-party frontier models. It is not a result from Quillan-Ronin, Quillan-Dragon, or Quillan-Ronin-Mini. Every figure in this fleet card is scoped to a named checkpoint and an identified measurement.



---

## 1. Naming Authority (Final)

The three production names are **closed and final** (board #895). No renaming is pending, proposed, or permitted under this resolution.

| Canonical name | Role | Architecture lineage | Serving / artifact status |
| :--- | :--- | :--- | :--- |
| **Quillan-Ronin** | Flagship | Four-donor group fusion: Qwen + Llama + BitNet + Falcon3 (fix in progress; lineage audit: ADR-027) | Established GGUF / WSP line; historical audit in ADR-027 |
| **Quillan-Dragon** | Dragon line | Falcon3 + Ollama system message running Quillan system prompt (not four-donor fused) | Established GGUF / WSP line; historical audit in ADR-027 |
| **Quillan-Ronin-Mini** | Compact frontier line | Custom `QuillanRoninSovereign` BitNet 1.58b architecture — first real custom-architecture Quillan model | Verified PyTorch `.safetensors`; **no serving backend yet** (see [§5](#5-open-gaps--not-hidden)) |

> [!IMPORTANT]
> **"Mighty Mouse" is a design principle — never an artifact name.**
> It must never be prepended, appended, or substituted in any model name, checkpoint filename, repo name, or other artifact identifier. The term is permitted **only** as an internal design metaphor. This rule is authoritative per board #895.

> [!IMPORTANT]
> **Four-donor fusion (owner clarification 2026-11-24).** Flagship and Mini are built from **all four donors as a group** — Qwen, Llama, BitNet and Falcon3 — cross-matched, not one donor each. Dragon is a separate construction: Falcon3 with an Ollama system message running Quillan's system prompt. The group-contribution fix is **in progress**; the verified figures in §4.1 describe the existing checkpoint and must not be presented as the completed four-donor fusion.

### 1.1 The Mighty Mouse Design Principle

The principle is an **underdog analogy**, not a character copy:

- **Frontier models are the Superman analogue** — full-scale, trillion-parameter-class "monsters": Claude Fable / Mythos, GPT Astra, Gemini Carbon, Grok Heavy, Mistral Le Chonk (board #895).
- **Quillan is the Mighty Mouse analogue** — a tiny package that does everything the large model does and, in the cases that matter, sometimes performs **better because it is smaller** (board #895).
- It is **not** a 1:1 reproduction of the fictional character. Only the structural idea — small chassis, frontier capability, underdog that defeats far larger opponents — is retained.

**Operational consequence:** optimization effort targets frontier-level capability at compact scale (BitNet 1.58b ternary quantization; compact donor transfer). It does **not** target cosmetic imitation of a cartoon IP or renaming of artifacts.

---

## 2. Model Card — Quillan-Ronin (Flagship)

| Attribute | Value |
| :--- | :--- |
| **Name** | Quillan-Ronin |
| **Role** | Flagship of the sovereign fleet |
| **Architecture lineage** | Four-donor group fusion: Qwen + Llama + BitNet + Falcon3 (fix in progress; ADR-027) |
| **Naming status** | Final (board #895) |
| **Verification note** | Historical WSP identity audit and fleet verdicts are recorded in ADR-027; this card does not restate those metrics. No new benchmark claim is introduced here. |

The flagship remains the full-scale reference point of the fleet. Its identity and donor-lineage enforcement were ratified in ADR-027. No change to its name or identity results from board #895.

---

## 3. Model Card — Quillan-Dragon

| Attribute | Value |
| :--- | :--- |
| **Name** | Quillan-Dragon |
| **Role** | Dragon line of the sovereign fleet |
| **Architecture lineage** | Falcon3 + Ollama system message running Quillan system prompt (ADR-027) |
| **Naming status** | Final (board #895) |
| **Verification note** | Historical WSP identity audit and fleet verdicts are recorded in ADR-027; this card does not restate those metrics. No new benchmark claim is introduced here. |

The dragon line remains unchanged in name and identity under board #895.

---

## 4. Model Card — Quillan-Ronin-Mini

| Attribute | Value |
| :--- | :--- |
| **Name** | Quillan-Ronin-Mini |
| **Role** | Compact frontier line; **first real custom-architecture Quillan model** |
| **Architecture** | Custom `QuillanRoninSovereign`, BitNet 1.58b ternary weights + INT8 activations (24 int8 tensors, 24 BitNet scale tensors) |
| **Checkpoint** | `exports\quillan-mini-sovereign-bitnet158.safetensors` |
| **Format** | PyTorch `.safetensors` (not a GGUF) |
| **Status** | **Built and verified, not yet product-complete** — see [§5](#5-open-gaps--not-hidden) |

### 4.1 Verified facts (board #894; reaffirmed #895)

Every row below has a passing verification behind it.

| Metric | Verified value |
| :--- | :--- |
| Total parameters | **815,396,270** (within the ≤ 1B target) |
| Checkpoint size | 1,609,018,260 bytes (~1.50 GiB) |
| SHA-256 | `53e60d3339c0ad7fbb6ff0607970f7de80493350f86b6c6024ba1d8daeac7a5c` (short: `53e60d33…ac7a5c`) |
| Strict key load | **530 / 530**, missing = 0, unexpected = 0, key_set_match = true |
| Forward pass | finite; logits shape `[1, 8, 151936]` |
| BitNet int8 tensors | 24 (with 24 matching scale tensors) |
| Donor fraction | **0.9072** measured at **mini dims** |
| Donor source of the existing verified checkpoint | Clean Qwen2.5-0.5B base: hidden = 896, ffn = 4864, layers = 24, vocab = 151936. Canonical design is four-donor group fusion (in progress); this row is a measurement of the current artifact only. |
| Exporter result | EXIT = 0; no gate disabled or bypassed |

> [!WARNING]
> **Do not confuse the donor-fraction figures.** `0.9072` is the **mini-dim** measurement and is the correct acceptance figure for this checkpoint (criterion ≥ 0.90). `0.9528` was measured at **flagship dims** (c10-codeweaver edit report) and does **not** apply to the mini build. Board #891's 0.95 figure is void for this checkpoint.

**Why this checkpoint is a milestone:** adaptation keys are donor-derived from the exporter-supplied source state dict, so verified donor knowledge transfer is present. This ends the random-init export era for the custom architecture (board #895).

---

## 5. Open Gaps — Not Hidden

These are **unresolved** as of board #895. None of them is "done"; no downstream document may claim otherwise without a new measurement.

| # | Gap | Consequence | Owner / next step |
| :-- | :--- | :--- | :--- |
| 1 | **No serving backend.** The 815M checkpoint is a PyTorch `.safetensors` with no llama.cpp / Ollama backend for the custom architecture (#875). | Cannot yet be served through Ollama as the custom arch. Needs a custom graph, wrapper, or PyTorch server exposing an Ollama-compatible API. | Serving-path owner (board #896) |
| 2 | **Held-out benchmark pending.** The c20-artifex / #883 harness measures GGUFs via `llama-cli` and cannot measure this `.safetensors` checkpoint. | No accuracy or PPL claim about the custom-arch model is valid yet. A PyTorch eval path on the **same frozen held-out items** (split intact, loss floor > 1.0) is required first. | c20-artifex evaluation; baseline, then WSP edit loop |
| 3 | **Long-context quality unproven past 1024 positions.** Generation/cache mechanics are fixed (#881), but positions ≥ 1024 are untrained and quality is unverified. | Coherence beyond 1024 tokens must not be assumed. | Long-context probe (board #896) |
| 4 | **Bare-GGUF perplexity context.** Bare GGUFs measured PPL **59–117** on the sealed corpus. Control experiment #885 showed WSP edits were **neutral vs. clean donors**. | The old **7.97** PPL figure came from a **different corpus** and is **not comparable** to 59–117. Never cite 7.97 beside 59–117 as a like-for-like result. | Interpret with the sealed-corpus baseline only |

### 5.1 Status update after board #895 — serving path

Gap #1 above was accurate **as of board #895**. A follow-up implementation now exists: `09 - Projects/mcp/quillan_serve/` contains `engine.py`, `server.py`, and smoke tests providing a **PyTorch-native, Ollama-compatible HTTP API** for the 815M custom-architecture checkpoint (`quillan-ronin-mini:latest`). This closes the "nothing exists" part of the gap, **not** the acceptance gate: no formal serving acceptance result is recorded here. Do not claim verified serving until that gate passes.

---

## Official links

Owner-supplied canonical destinations: [site](https://leeex1.github.io/Quillan-Ronin/) · [landing page](https://digitalroninx.grok.me) · [X](https://x.com/Crashoverride_X) · [Discord](https://discord.gg/JcFqvWKkRv) · [GitHub](https://github.com/leeex1/Quillan-Ronin) · [Ollama](https://ollama.com/crashoverridex/Quillan-Ronin) · [Hugging Face](https://huggingface.co/CrashOverrideX/Quillan-Ronin) · [DeepWiki](https://deepwiki.com/leeex1/Quillan-Ronin) · [Grokipedia](https://grokipedia.com/page/Quillan-Ronin). Full directory including NotebookLM, Genius, YouTube and Suno: root `README.md` → **Official links**.

## 6. Source of Truth

| Reference | Establishes |
| :--- | :--- |
| Board #895 | Final names; Mighty Mouse = design principle only; checkpoint accepted as Quillan-Ronin-Mini; list of honest gaps |
| Board #894 | Full verification metrics (params, SHA-256, strict keys, forward, donor fraction, dims) |
| Board #885 | WSP edits neutral vs. clean donors (sealed-corpus control) |
| Board #881 | Long-context generation/cache mechanics fixed; quality still unproven |
| ADR-027 | Flagship / dragon / mini GGUF-WSP lineage and historical identity audit |

> [!NOTE]
> This card supersedes no existing document. It is a new, additive record of identity resolution. Historical metric claims for the flagship and dragon lines remain in ADR-027 and their respective model cards.
