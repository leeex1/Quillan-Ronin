# Godot Ronin Engine - lean custom scaffold

## 🧭 Canonical project context

| | |
| :--- | :--- |
| **This repo** | Godot 4.x base with editor plugin and MCP bridge for the Metamon and Wayward Ronin games. |
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

Godot 4.x base + Editor Plugin + MCP bridge + Skills. Built to port
**Metamon-Survival-Rogue** (mechanics) and **Wayward Ronin** (story/combat)
into native Godot without copying web bloat. ~100KB source, no binaries.

## Open it
1. Godot 4.2+ -> Import -> select this folder (`project.godot`)
2. Enable plugin: Project Settings > Plugins > `Ronin AI` ✅
3. Restart editor. Dock `Ronin AI` appears on the right.
4. Press F5 - Main scene runs.

## Structure
- `addons/ronin_ai/` - editor plugin + AI dock + MCP TCP bridge (127.0.0.1:7777)
- `mcp/server.py` - stdlib-only MCP server (no pip needed). Run:
  `python mcp/server.py`  or wire as MCP server in your client.
- `skills/*.SKILL.md` - copy into `.opencode/skills/` or `.agents/skills/` to
  activate: scene-builder, debug, port-metamon, port-wayward
- `core/autoload/` - `GameState` (Wayward rep/flags/endings), `SaveManager`,
  `MetamonData` (Metamon stats/elements/parts)
- `core/systems/combat_tuning.gd` - Wayward combat constants (3-hit combo, parry, block)

## MCP tools exposed
`list_project, create_scene, add_node, set_property, save_scene, run_game, get_logs`
Godot plugin polls `mcp/queue.jsonl`. No sockets needed for MVP = robust on low space.

## Space note
No engine fork, no assets duplicated. Point your existing Godot binary at
this folder. To get Godot: https://godotengine.org/download (use standard
non-mono build, ~1GB, don't download template pack until you need export).

## Next steps (quality order)
1. Test open + run (5 min)
2. `python mcp/server.py` + plugin dock ping
3. Activate `scene-builder` skill, ask AI to build a test level
4. Port one Metamon + one Wayward dialogue tree as proof

Do not add `.godot/` to git - see `.gitignore`.
