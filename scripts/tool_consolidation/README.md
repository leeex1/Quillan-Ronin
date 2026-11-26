# `scripts/tool_consolidation/`

## 🧭 Canonical project context

| | |
| :--- | :--- |
| **This repo** | Tool-consolidation scripts for the Quillan workspace. |
| **Canonical package** | Three deployed models: Quillan-Ronin Flagship (~1B), Quillan-Ronin-Mini (~0.8B) and Quillan-Dragon (~1B). The ONI Python sources are implementation/training only. |
| **Donor design** | Flagship and Mini: Qwen 0.6B + Llama ~1B + BitNet 2B-4T + Falcon3 ~1B **as a group** (cross-match fix in progress). Dragon: Falcon3 with an Ollama system message running Quillan's system prompt. |
| **Reading the scores** | The 95–100% table in the root README is **Track A** — the Quillan harness on a third-party frontier base, not a Quillan checkpoint. Hard-exam cells are pending re-test; ARC-AGI-3 public ~99s, private pending. |
| **Historical framing** | Edo period, rōnin and bushidō are real Japanese history and ethics — not roleplay, game mechanics, or fantasy lore. |
| **Canonical docs** | [Root README](https://github.com/leeex1/Quillan-Ronin#readme) · [Fleet model cards](https://github.com/leeex1/Quillan-Ronin/blob/main/docs/MODEL_CARDS.md) |

<!-- QUILLAN-DOCS-PROVENANCE:BEGIN -->
<details>
<summary><b>📌 Quillan documentation provenance</b> — expand for canonical sources</summary>

> **Documentation provenance.** This repo is part of the Quillan-Ronin ecosystem (CrashOverrideX / `leeex1`). Canonical sources live in the root repository: `README.md` (**READ FIRST** — Track A harness-on-frontier vs. Track B native WSP models vs. Track B′ custom Mini) and `docs/MODEL_CARDS.md` (fleet identity, verified facts, open gaps). The large 95–100% benchmark table in the root README is a **harness-on-frontier** result, not a score of any Quillan-owned checkpoint. Creator artifacts (books, music, game demo) have primary-source links in the root README's **Creator and verifiable artifacts** section.  Four-donor design: Flagship and Mini receive all four donors (Qwen ~0.6B + Llama ~1B + BitNet 2B-4T + Falcon3 ~1B) **as a group**; Dragon is Falcon3 with an Ollama system message running Quillan's system prompt. Cross-match fix in progress.  The canonical package is the three deployed models — Quillan-Ronin Flagship (~1B), Quillan-Ronin-Mini (~0.8B) and Quillan-Dragon (~1B); the ONI Python sources are the implementation/training package, not the shipped product. Canonical official links (site, social, repos, Ollama, HF, Grokipedia, NotebookLM, DeepWiki, Genius, YouTube/Suno) are in the root README's **Official links** directory. Historical framing is real Japanese history (Edo period, rōnin, bushidō) used as an ethical reference frame — not roleplay, game mechanics, or fantasy lore.

</details>

<!-- QUILLAN-DOCS-PROVENANCE:END -->

Safe subset of PASS 2 tool consolidation (audit #886 / coordinator brief #896).

## Files

| File | Purpose |
|---|---|
| `namespace_registry.py` | Single source of truth: 17 session namespaces, 9 route profiles, all 54 legacy tool specs, preserved-name tables. |
| `canonical.py` | The three canonical tools: `mcp_session_replay`, `mcp_session_list`, `mcp_route`. Pure/read-only. |
| `shims.py` | Namespaced shim for every legacy name + `resolve(name)` / `is_legacy_name(name)`. |
| `verify.py` | Empirical verifier: every legacy name resolves; counts match audit; dangerous strings intact; canonical tools execute. Exit 0/1. |
| `DEPRECATION.md` | 90-day / one-minor-release policy and migration table. |

## Quick start

```bat
python C:\02_QUILLAN\scripts\tool_consolidation\verify.py
```

```python
from tool_consolidation import resolve, mcp_route

resolve("audio_session_replay")("sid")           # warns; namespace frozen to 'audio'
mcp_route("council", {"k": 1}, source="rag")     # canonical
```

## Facts (verified against files, not assumed)

- Zero exact duplicate tools across the 19 original servers (audit #886).
- `session_replay` x17 (13 byte-identical), `session_list` x17, cross-MCP route templates x9 (orochi's differs → excluded), 3 pure aliases.
- `prism_decompose` has two different signatures → kept namespaced, never merged.
- 6 thinking_engine `name=` overrides preserved, including typos `Genuis_level` / `adversarial_dual_council_diliberation`.
- Live MCP tool count stays **539**; this package adds new modules and touches no live config.
