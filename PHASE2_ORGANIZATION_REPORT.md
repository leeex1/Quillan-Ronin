# Phase 2 Organization Report — 2026-11-23

## Scope

Hash-audit of first-party folders: `docs`, `_config`, `_archive`, `scripts`, `validation_toolkit`,
`05 - Creative Works`, `06_Media`, `gallery`, `main-images`, `assets`, `Audio Engineer`, `audio`,
`demos`, `exports`, `devkit`, `Misc`, `kernels`, `daemons`, `checkpoints`, `lancedb`.

Excluded (intentionally): vendor/runtime trees — `09 - Projects` (238,713 files),
`02 - Knowledge Foundation` (24,673), `.venv*`, `node_modules`, `.git`.

## Evidence

| Metric | Value |
|---|---|
| Files scanned (full SHA256 audit) | **7,056** |
| Duplicate groups (byte-identical) | **937** |
| Redundant copies | **3,193** |
| Wasted space | **12,319.81 MB (~12.0 GB)** |
| Cross-zone duplicate groups | **450** |
| Largest single duplicate | `exports\quillan-dragon-math40-headrow2.safetensors` — 1,118,846 KB, byte-identical to `headrow.safetensors`, zero references |

Duplicate extension breakdown: 199 `.md`, 174 `.webp`, 162 `.png`, 143 `.mp3`, 90 `.jpg`,
58 `.pdf`, 31 `.py`, 16 `.flac`, 14 `.json`, 13 `.html`.

## Immutable zones (verified, not moved)

| Zone | Reason | Evidence |
|---|---|---|
| `docs/**` media (1,454 images) | GitHub Pages serves from `docs/`; HTML references `./assets/...`, `./gallery/...` | `docs/index.html`, `_quillan_backup/index.html:572-579` |
| `exports/**` (except headrow2) | 60+ absolute-path references in scripts | `scripts/analyze_100_root_causes.py:7`, `scripts/compile_flagship_*.py`, etc. |
| `06_Media/**` | Audio/NFT pipeline writes and reads | `run_ears.py:41-45`, `hf_sync_all.py:82`, `generate_nft_metadata.py:18`, `sanitize_audio.py:3` |
| `_config/**` | Canonical skill library (507 skill files vs 6 at root) | `Get-ChildItem _config\.claude\skills` = 507 files |
| Root canonical files (`AGENTS.md`, `SOUL.md`, `memory.md`, `opencode.json`, `.mcp.json`, `Modelfile*`, `quillan.db`, etc.) | Live MCP ingestion whitelist + absolute-path reads | `.../quillan_rag/ingest_second_brain.py:23-24`, `.../quillan_spider/quillan_brain_indexer.py:250-252`, Warden addendum |
| Active-work dirty entries (88 at Phase 1 start) | User's uncommitted changes | `git status --porcelain` |

## Actions taken (reversible, ZERO deletions)

A conservative quarantine rule was applied: a redundant copy was moved **only if** it (a) had a
byte-identical retained canonical copy, (b) contained a nested-repeat directory
(`audio\audio`, `attached_assets\attached_assets`) or a duplicate-name suffix (`(1)`, `_2`, `copy`),
and (c) was inside `05 - Creative Works` (plus the one unreferenced 1.1 GB model duplicate).

| Metric | Value |
|---|---|
| Files quarantined | **250** |
| Space recovered | **2,961.28 MB (~2.89 GB)** |
| Failures | **0** |
| Pages copies touched | **0** |
| Deletions | **0** |
| Manifest | `QUARANTINE_MOVES.json` (250 entries, full rollback paths) |
| Destination | `_archive\duplicate_media_20261123\` |

Breakdown: 177 nested-repeat-dir copies · 72 duplicate-name variants · 1 unreferenced model duplicate.

Verification: 250/250 destinations exist, 250/250 sources absent, quarantine contains exactly 250 files,
`git status` shows no new deletions (the single `D trace_signal.py` is the Phase 1 rename-detection
artifact, SHA256 confirmed preserved).

## Remaining (presented for user decision, NOT acted on)

~2,943 redundant copies remain in the 12.0 GB duplicate set. These require keeping one canonical
copy per hash group while preserving Pages copies. **No deletion or further move has been performed.**
User approval is required before any deletion.

## Prior phases

- Phase 1: 84 root files categorized into `00`–`13` taxonomy; 2 path breakages patched and
  `py_compile` verified; ledger `ORGANIZATION_MOVES.json` (84 entries).
- Divergent pairs: newest versions retained, older moved to `12 - Backups\superseded\`;
  analysis in `MERGE_ANALYSIS.md`.
