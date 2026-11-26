# RONIN THRONGLETS 🏯⚡

## 🧭 Canonical project context

| | |
| :--- | :--- |
| **This repo** | Experimental project in the Quillan ecosystem. |
| **Canonical package** | Three deployed models: Quillan-Ronin Flagship (~1B), Quillan-Ronin-Mini (~0.8B) and Quillan-Dragon (~1B). The ONI Python sources are implementation/training only. |
| **Donor design** | Flagship and Mini: Qwen 0.6B + Llama ~1B + BitNet 2B-4T + Falcon3 ~1B **as a group** (cross-match fix in progress). Dragon: Falcon3 with an Ollama system message running Quillan's system prompt. |
| **Reading the scores** | The 95–100% table in the root README is **Track A** — the Quillan harness on a third-party frontier base, not a Quillan checkpoint. Hard-exam cells are pending re-test; ARC-AGI-3 public ~99s, private pending. |
| **Historical framing** | Edo period, rōnin and bushidō are real Japanese history and ethics — not roleplay, game mechanics, or fantasy lore. |
| **Canonical docs** | [Root README](https://github.com/leeex1/Quillan-Ronin#readme) · [Fleet model cards](https://github.com/leeex1/Quillan-Ronin/blob/main/docs/MODEL_CARDS.md) |

<!-- QUILLAN-DOCS-PROVENANCE:BEGIN -->
<details>
<summary><b>📌 Quillan documentation provenance</b> — expand for canonical sources</summary>

> **Documentation provenance.** This repo is part of the Quillan-Ronin ecosystem (CrashOverrideX / `leeex1`). Canonical sources live in the root repository: `README.md` (**READ FIRST** — Track A harness-on-frontier vs. Track B native WSP models vs. Track B′ custom Mini) and `docs/MODEL_CARDS.md` (fleet identity, verified facts, open gaps). The large 95–100% benchmark table in the root README is a **harness-on-frontier** result, not a score of any Quillan-owned checkpoint. Creator artifacts (books, music, game demo) have primary-source links in the root README's **Creator and verifiable artifacts** section.  Four-donor design: Flagship and Mini receive all four donors (Qwen ~0.6B + Llama ~1B + BitNet 2B-4T + Falcon3 ~1B) **as a group**; Dragon is Falcon3 with an Ollama system message running Quillan's system prompt. Cross-match fix in progress.  The canonical package is the three deployed models — Quillan-Ronin Flagship (~1B), Quillan-Ronin-Mini (~0.8B) and Quillan-Dragon (~1B); the ONI Python sources are the implementation/training package, not the shipped product. Canonical official links (site, social, repos, Ollama, HF, Grokipedia, NotebookLM, DeepWiki, Genius, YouTube/Suno) are in the root README's **Official links** directory. Historical framing is real Japanese history (Edo period, rōnin, bushidō) used as an ethical reference frame — not roleplay, game mechanics, or fantasy lore.

</details>

<!-- QUILLAN-DOCS-PROVENANCE:END -->

A Thronglets-style raise-and-evolve swarm sim. Cute-menacing samurai robot thralls wake
beneath a floating cyber-temple dojo and learn to want.

**Inspired by** Black Mirror S7: "Thronglets" · **built with** Godot 4.7.2 · **assets** 100% custom (Blender 5.1 procedural build + EEVEE renders)

## The loop
- **They split themselves.** You cannot buy babies — feed them, cleanse them, keep them glorious, and one takes hunger and splits in two. Capacity is limited by **Thrall Dens** (base 6, +4 each).
- **Two real resources + alloy**: Wood (timber crude), Signal Gems (spent on tools & evolution rites), Data-Alloy (precious, forge-made).
- **The thralls harvest for you** — Foragers chop trees and crack rocks autonomously. Buy the **Pickaxe** (25◆) and **Chainsaw** (60◆) to speed it up.
- **The society assigns its own roles** every few seconds: Foragers, Tenders, Sentinels, Forge-Wardens — by need, not by you.
- **They make requests** — banners at the top of the screen: "We ask for a FORGE…" — fulfill them or suffer for ignoring them.
- **Evolution rites cost gems** (Ashigaru→Samurai 25◆, Samurai→Daimyō 60◆ + 10◈) — feed the Furnace of Becoming.
- **The Ledger watches everything** — every choice you make is remembered, and the interviews reference your history.
- **Spar at the Dojo** — two thralls duel, winner earns Valor (faster evolution).
- **Five endings** — The Swarm Wakes · The Blight Bloom · The Ascension · Sentient Sanctuary · (and your Ledger title stamps them all).

## Controls
| Key | Action |
|---|---|
| Left-drag / Arrow keys | Orbit camera |
| Mouse wheel / Up-Down arrows | Zoom |
| `Q` or the ⚡ button | **RONIN SWARM PROTOCOL** — all thralls dash to the torii core in one shockwave. Costs 25◈ + cooldown. Requires ≥5 thralls. |
| `` ` `` (backquote) | Dev console |
| `Esc` | Close console |

## Dev console commands
`/swarm N` · `/feed` · `/cleanse` · `/corrupt` · `/alloy N` · `/win` · `/help`

## Build
```
Godot_v4.7.2-stable_win64.exe --path game           # run from source
# .exe export (Windows Desktop preset included):
Godot --headless --path game --export-release "Windows Desktop" dist/RoninThronglets.exe
```
