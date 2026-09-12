# LanceDB + MD + Obsidian E2E — Quillan Memory System

## Architecture (inside 02_QUILLAN only)
- **MD source:** knowledge/canonical/*.md (49 files, ONE canonical)
- **Obsidian vault:** .obsidian/ at root (vault = C:\02_QUILLAN)
- **LanceDB:** lancedb/thoughts.lance + 07 - Memory & LanceDB/lancedb (synced)
- **Chroma fallback:** quillan_memory/chroma.sqlite3
- **Sessions:** sessions/SESSION_INDEX.md + 00 - Meta/chatlogs

## E2E Flow
1. Canonical MD written -> Obsidian indexes instantly (.obsidian watches)
2. Obsidian -> LanceDB ingest via quillan_memory bridge
3. Query -> LanceDB hybrid search + MD fallback

## Status
- Canonical: 49 md ✅
- Platforms mirrors: 7 folders (Claude/GPT/Gemini/Grok/Mistral) — synced from canonical ✅
- LanceDB: POPULATED (901 chunks, 2048-dim vectors, nvidia/nemotron-3-embed-1b) ✅
  - Primary: `C:\02_QUILLAN\lancedb\thoughts.lance`
  - Synced: `C:\02_QUILLAN\07 - Memory & LanceDB\lancedb\thoughts.lance`
  - Synced: `C:\02_QUILLAN\07 - Memory & LanceDB\quillan_memory\thoughts.lance`
- Obsidian: restored to root (.obsidian/app.json present) ✅
- quillan_memory: chroma.sqlite3 fallback + thoughts.lance primary ✅

## Maintenance / Re-Ingest Command
`python scripts/ingest_lancedb.py`
