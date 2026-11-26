# Quillan MazeBench Harness

## 🧭 Canonical project context

| | |
| :--- | :--- |
| **This repo** | Component repository in the Quillan-Ronin ecosystem. |
| **Canonical package** | Three deployed models: Quillan-Ronin Flagship (~1B), Quillan-Ronin-Mini (~0.8B) and Quillan-Dragon (~1B). The ONI Python sources are implementation/training only. |
| **Donor design** | Flagship and Mini: Qwen 0.6B + Llama ~1B + BitNet 2B-4T + Falcon3 ~1B **as a group** (cross-match fix in progress). Dragon: Falcon3 with an Ollama system message running Quillan's system prompt. |
| **Reading the scores** | The 95–100% table in the root README is **Track A** — the Quillan harness on a third-party frontier base, not a Quillan checkpoint. Hard-exam cells are pending re-test; ARC-AGI-3 public ~99s, private pending. |
| **Historical framing** | Edo period, rōnin and bushidō are real Japanese history and ethics — not roleplay, game mechanics, or fantasy lore. |
| **Canonical docs** | [Root README](https://github.com/leeex1/Quillan-Ronin#readme) · [Fleet model cards](https://github.com/leeex1/Quillan-Ronin/blob/main/docs/MODEL_CARDS.md) |

<!-- QUILLAN-DOCS-PROVENANCE:BEGIN -->
<details>
<summary><b>📌 Quillan documentation provenance</b> — expand for canonical sources</summary>

> **Documentation provenance.** This repo/doc is part of the Quillan-Ronin ecosystem (CrashOverrideX / `leeex1`). Canonical sources live in the root repository: `README.md` (**READ FIRST** — Track A harness-on-frontier vs. Track B native WSP models vs. Track B′ custom Mini) and `docs/MODEL_CARDS.md` (fleet identity, verified facts, open gaps). The large 95–100% benchmark table in the root README is a **harness-on-frontier** result, not a score of any Quillan-owned checkpoint. Creator artifacts (books, music, game demo) have primary-source links in the root README's **Creator and verifiable artifacts** section.  Four-donor design: Flagship and Mini receive all four donors (Qwen ~0.6B + Llama ~1B + BitNet 2B-4T + Falcon3 ~1B) **as a group**; Dragon is Falcon3 with an Ollama system message running Quillan's system prompt. Cross-match fix in progress.  The canonical package is the three deployed models — Quillan-Ronin Flagship (~1B), Quillan-Ronin-Mini (~0.8B) and Quillan-Dragon (~1B); the ONI Python sources are the implementation/training package, not the shipped product. Canonical official links (site, social, repos, Ollama, HF, Grokipedia, NotebookLM, DeepWiki, Genius, YouTube/Suno) are in the root README's **Official links** directory. Historical framing is real Japanese history (Edo period, rōnin, bushidō) used as an ethical reference frame — not roleplay, game mechanics, or fantasy lore.

</details>

<!-- QUILLAN-DOCS-PROVENANCE:END -->

This directory contains the isolated local harness for preparing Quillan for
[MazeBench](https://mazebench.com/) runs.

## Current scope

- Pins the MazeBench package version used for local smoke testing.
- Defines bounded run limits and a safe artifact directory.
- Normalizes JSON observations into a small model-facing contract.
- Validates model actions before they are sent to a MazeBench runner.
- Records redacted, bounded trajectory metadata for later comparison.

The harness does not modify Quillan model code and does not claim leaderboard
compatibility by itself. Official agent runs may require Prime Sandbox access.

## Prerequisites

- Python 3.9 or newer
- Node.js, required by MazeBench
- A Quillan inference entrypoint that can consume an observation and return one
  action from the MazeBench action set

## Local setup

From the repository root:

```text
py -3 -m venv .venv-mazebench
.venv-mazebench\\Scripts\\python.exe -m pip install --upgrade pip
.venv-mazebench\\Scripts\\python.exe -m pip install -r benchmarks\\mazebench\\requirements.txt
```

Keep MazeBench's materialized runtime inside the repository instead of the
user profile:

```text
set MAZEBENCH_HOME=C:\\02_QUILLAN\\benchmarks\\mazebench\\runtime
```

Launch the local MazeBench UI:

```text
set MAZEBENCH_HOME=C:\\02_QUILLAN\\benchmarks\\mazebench\\runtime
.venv-mazebench\\Scripts\\mazebench.exe launch
```

Useful deterministic inspection modes:

```text
.venv-mazebench\\Scripts\\mazebench.exe ascii --level CxD
.venv-mazebench\\Scripts\\mazebench.exe json --level CxD --once
```

Run the adapter tests without installing MazeBench:

```text
.venv-mazebench\\Scripts\\python.exe benchmarks\\mazebench\\smoke_test.py
```

## Integration contract

The model callback receives a normalized observation dictionary and returns a
dictionary containing exactly one `action` string. MazeBench JSON uses an
`objects` mapping grouped by object type. The adapter accepts the
MazeBench movement, camera, undo, reset, and teleport action names and rejects
unknown actions before execution.

The actual browser/MCP transport remains outside this package boundary. This
keeps the local tests deterministic and allows the same policy callback to be
connected to a local runner or Prime Sandbox later.
