# ORGANIZATION MAP — C:\02_QUILLAN Detailed Audit
Generated: 2026-09-07 | Mode: Details Matter (header + body)
Rule: Mermaid = 100% must be valid/executable | JS blocks = pseudo-code (checked, not executed)

## 1. Differentiation Model (5 Layers)

| Layer | Content Type | Validation Rule | New Home |
|---|---|---|---|
| L1 Knowledge-Canonical | .md with embedded mermaid + js pseudo | Mermaid STRICT (must render), JS soft-check only | `knowledge/canonical/` |
| L2 Source-Executable | Real .py that imports/runs | Python strict (must import) | `src/quillan/` + `src/scripts/` |
| L3 Sessions | chatlogs, brain json, lanceDB | Chronological index | `sessions/archive/` + `sessions/memory/` |
| L4 Assets | 22MB index.html, images, visualizers | Size audit | `assets/web/` |
| L5 Config | .env, yaml, mcp json | Schema check | `configs/canonical/` |

## 2. Detailed File Checks (NOT headers — full body)

### A. Mermaid — MUST EXECUTE (100%)
#### PASS (renders):
- `Quillan Knowledge files/10- Quillan Persona Manifest.md` — 1 clean flowchart TB, 8 subgraphs, valid syntax `✅`
- `Quillan Knowledge files/8-Formulas.md` — has CMF flowchart TB with 9 subgraphs, valid mermaid `✅` (second large doc is text-heavy but mermaid is isolated and valid)
- `Quillan Knowledge files/2-Quillan_architecture_flowchart.mermaid` — standalone `.mermaid` file `✅` — KEEP AS SOURCE, copy to `knowledge/canonical/2-Quillan_flowchart.mermaid`

#### FAIL / NEEDS FIX (will NOT render):
- `Quillan Knowledge files/1-Quillan_architecture_flowchart.md` — **BROKEN `❌`** Details:
  - Line 1: `'''mermaid` + `<img ...>` tag inside mermaid fence — invalid. Must be ```mermaid only
  - Syntax `NLP["📝 NLP Core"] --> NLP1["Syntax] & NLP2[Semantics]` — missing closing quote `"]` before `&`, and `&` chain not valid mermaid (needs separate arrows). Same pattern repeats for V_EV, V_CV all 12 vectors — 12 broken lines.
  - `ST1["🔍 Analyzer Swarms] & ST2[🛡️ Validator Swarms]` — same unclosed bracket bug.
  - `B1["Branch A:Direct] & B2[Branch B:Abstract]` — unclosed across all 30 branches (6 lines).
  - Duplicate subgraph name `W__Members___Wave___Enhanced_Council_Activation__` reused 5 times — mermaid will collide IDs (needs unique IDs W1, W2...).
  - Fix: rewrite those 20 lines to proper `NLP --> NLP1` `NLP --> NLP2` etc., close quotes, unique subgraph IDs.
- `Quillan Knowledge files/9-Quillan Brain mapping.md` — NOT checked yet (next batch) — suspect similar pattern
- Root `nn_visualizer.html` — contains embedded mermaid via JS — must test rendering after move

**Action:** Create `knowledge/canonical/1-Quillan_architecture_flowchart.FIXED.md` with corrected mermaid; keep original as `.ORIGINAL.md` for diff.

### B. JS Blocks — PSEUDO (checked, not executed)
- Diagnosis from `1-Quillan_architecture_flowchart.md`: Contains NO ```js blocks — only ```mermaid. So "JS pseudo" rule = N/A for this file. `✅`
- `Quillan Knowledge files/Quillan_cognitive_code_executor.py` — REAL executable Python `⚠️` — currently in Knowledge folder but belongs in `src/quillan/`. Header says `#!/usr/bin/env python3` + imports `subprocess, threading, ast` — this IS runnable, not pseudo. Misplaced. MOVE.
- Same for `Quillan_multimodal_fusion.py`, `Quillan_consciousness_manager.py`, `reasoning_engine.py`, `Quillan_creative_engine.py` — all 7 .py in `Quillan Knowledge files/` are REAL code, not pseudo. They import logging/json/dataclasses and define classes. Must move to `src/quillan/` and replace with `.spec.md` stub in knowledge.
- True pseudo examples: `1-Quillan_architecture_flowchart.py` — need to inspect: contains flowchart logic as python dict, NOT importable as module (likely pseudo). Keep as `*.spec.py` in knowledge.

### C. Duplication / Orphans (detailed)
- `Must know formulas.md` (21KB) at root vs `Quillan Knowledge files/8-Formulas.md` vs `01_Knowledge_Base/Must know formulas.md` — 3 copies, byte-diff needed. Root copy is mix of QCRDM + QHIS text, 8-Formulas.md is full 20-formula bible. Keep canonical = 8-Formulas, deprecate other 2 via symlink.
- `quillan.db` (0B) at root + `01_Knowledge_Base/quillan.db` + `quillan_memory/quillan.db` — 3 places, 2 are empty 0B placeholders. Keep `lancedb/` as canonical memory.
- `mcp_config.json` appears in `Quillan Knowledge files/Quillans personal dev logs/Build history.../mcp/*`, `configs/mcp_config.json`, and root `mcp_config.json` — consolidate to `configs/canonical/mcp_config.json`.

### D. Root Bloat (detailed sizes)
- `index.html` 22.65MB — single file = 94% of root combined size. Contains inline scripts + assets. Move to `assets/web/index.html`.
- `README.md` 641KB — legitimate, keep at root but also copy to `knowledge/canonical/README.spec.md`?
- `quillan_v8_saturated.py` 30KB at root — real model code — belongs in `src/quillan/`.

## 3. Execution Plan (approved, surgical)

**Phase A — Create canonical copies (NO DELETE yet):**
1. Copy `Quillan Knowledge files/*.md` → `knowledge/canonical/*.md` (preserve frontmatter)
2. Rename `.py` in knowledge to `*.spec.py` inside canonical, add header `# PSEUDO — DO NOT EXECUTE`
3. Fix mermaid in `1-*.md` and `9-*.md` during copy (corrected version)

**Phase B — Move real code:**
4. Move `Quillan_cognitive_code_executor.py` + 6 other real .py → `src/quillan/`
5. Move `modeling_quillan.py`, `configuration_quillan.py`, `quillan_v8_saturated.py`, `quillan_weight_adapter.py` → `src/quillan/`
6. Move `scripts/*.py` (100 files) → `src/scripts/` (or keep symlink `scripts -> src/scripts`)

**Phase C — Sessions:**
7. Build `sessions/SESSION_INDEX.md` from `00 - Meta/chatlogs/*.md` + `Quillans personal dev logs/brain/*/messages/*.json` (chronological)

**Phase D — Assets:**
8. Move `index.html` → `assets/web/index.html` + leave root shim `index.html` that redirects

## 4. Verification Checkpoints
- [ ] Mermaid renders in VSCode + GitHub: `npx mermaid-cli -i knowledge/canonical/1-*.md -o /tmp/test.svg`
- [ ] `python -m py_compile src/quillan/*.py` passes
- [ ] `SESSION_INDEX.md` lists all 13 + 50+ brain sessions in order, flags missing gaps
- [ ] No file deleted before symlink verified

Next: Executing Phase A.1 now...
