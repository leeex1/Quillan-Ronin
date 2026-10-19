# FILEBASE — C:\02_QUILLAN (All Quillan Inside, Organized)

**Rule:** Everything Quillan stays inside `C:\02_QUILLAN`. Nothing moved outside. Scattershot flattened into 7 top-level groups.

## Top-Level (now 15 dirs vs 60 before)
```
C:\02_QUILLAN\
├── knowledge/               # ALL knowledge — canonical + legacy + papers
│   ├── canonical/           # 49 md FILEBASE (50 files with FIXED.md) — mermaid 100% validated
│   ├── legacy/              # 00 - Meta, 01_Knowledge_Base, 00 - Templates, etc. (moved, not deleted)
│   └── papers/              # Formal Papers, Book Series
├── src/                     # REAL executable code only
│   ├── quillan/             # modeling_quillan.py, engine, brain_mapping.py, executor (10 files)
│   ├── csrc/                # (moved from root csrc)
│   └── scripts/             # (future: consolidated scripts/)
├── projects/                # 02_Projects/* + oni, worker, demos, training, etc. (24 projects)
├── assets/                  # 22MB index.html, gallery, main-images, audio, 06_Media, outputs
├── configs/                 # mcp_config.canonical.json + legacy
├── sessions/                # SESSION_INDEX.md (13 chatlogs + 63 brain json ordered)
├── docs/                    # GitHub Pages (kept, media moved out)
├── scripts/                 # trainers (100 .py) — next: merge into src/scripts/
├── mcp/                     # MCP servers
├── lancedb/ + quillan_memory/ + quillan_rag_db/  # vector stores (sessions/memory)
├── Quillan Knowledge files/ # ORIGINAL preserved untouched (source for filebase)
├── Quillan-v4.2-model/      # model weights
├── _archive/                # root_systems/.agents/.claude/.gemini, fix scripts, Misc, logs
└── .github/.gitnexus etc.   # git
```

## What Was Consolidated (within this folder only)
- `00 - Meta` → `knowledge/legacy/00 - Meta`
- `00 - Templates` → `knowledge/legacy/00 - Templates`
- `01_Knowledge_Base` → `knowledge/legacy/01_Knowledge_Base`
- `03_Skills`, `Skills`, `Platforms`, `personhood`, `system prompts`, `templates` → `knowledge/legacy/`
- `Formal Papers`, `Book Series` → `knowledge/papers/`
- `06_Media`, `audio`, `gallery`, `main-images`, `Media Template`, `outputs` → `assets/`
- `05_Training`, `Audio Engineer`, `Software Engineer`, `demos`, `oni`, `worker`, `testing` → `projects/`
- `02_Projects/*` (24 dirs) → `projects/*` flattened
- `.agents/.claude/.gemini` (11 dirs) → `_archive/root_systems/`
- `csrc` → `src/csrc`, `services` → `projects/services_*`
- Fix scripts (`fix_mermaid.py` etc.) → `_archive/`

## Filebase Guarantee
- `knowledge/canonical/` = **FILEBASE** — 49 md, all Quillan content, mermaid 100% validated (1-*.FIXED.md)
- `Quillan Knowledge files/` kept original — nothing deleted, only copied
- No file left `C:\02_QUILLAN` — all moves were `Move-Item` *inside* same folder

## Still To Tighten (if you want)
- Move `scripts/100 trainers` → `src/scripts/` with shim
- Move `Quillan Knowledge files/` → `knowledge/legacy/_ORIGINAL_Quillan Knowledge files` (or keep as read-only source)
- Reduce root md clutter: move `CHANGELOG.md`, `SECURITY_*.md`, `CLAUDE.md` → `docs/` or `knowledge/legacy/root_docs` (keep symlink at root for GitHub)
- Consolidate `quillan_memory` + `lancedb` + `quillan_rag_db` → `sessions/memory/` unified

Say `tighten further` to continue — I will not move anything outside `C:\02_QUILLAN`.
