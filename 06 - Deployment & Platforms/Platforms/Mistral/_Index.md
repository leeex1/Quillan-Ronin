# Knowledge Filebase Index — Canonical

Generated: 2026-09-07 | 49 files | Tier: 1 Quillan / 2 Council / 3 Swarm / 4 Engine(LLM)

This is the **FILEBASE** — single source of truth for all Quillan Knowledge.
Originals remain in `Quillan Knowledge files/` for safety; this folder is canonical for rendering & execution.

## Tier Map (4-tier)
- **1 Quillan** — `1-Quillan_architecture_flowchart.FIXED.md` (8 mermaid, at limit, now validated)
- **2 Council** — `10- Quillan Persona Manifest.md` (34 personas), `9-Quillan Brain mapping.md`
- **3 Swarm** — 30 WoT branches inside `1-*.FIXED.md`
- **4 Engine** — injected `TIER4_ENGINE` subgraph (LLM) — `src/quillan/engine/`

## Mermaid Status (100% must render)
- ✅ `1-Quillan_architecture_flowchart.FIXED.md` — FIXED (preserved all layers, fixed 48 & chains, 4 bracket bugs, added unique subgraph IDs, injected Engine tier without deleting any node)
- ✅ `1-Quillan_architecture_flowchart.ORIGINAL.md` — preserved untouched for diff
- ✅ `8-Formulas.md` — valid
- ✅ `9-Quillan Brain mapping.md` — valid
- ✅ `10- Quillan Persona Manifest.md` — valid
- ⏳ 43 newly copied files — queued for same detailed mermaid audit (not header-only)

## Content Preservation Guarantee
- No file deleted. All copied, not moved.
- `FIXED.md` keeps **every** vector (12), every branch (30), every wave (5), every persona, emojis, labels.
- Only syntax fixed: `["Syntax]` → `["Syntax"]`, `&` chains expanded to separate arrows, duplicate subgraph IDs uniquified, ST lines quoted, Engine tier added.

## JS Pseudo Rule
All ```js blocks in these 49 md are `pseudo` — checked for internal consistency only, never executed. Real executable JS/Python lives in `src/`.

## Next
- Run detailed per-file mermaid validation for remaining 43 md (body-level)
- Mark pseudo `.py` specs vs real code in `src/quillan/`
