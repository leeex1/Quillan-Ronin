# 00 VAULT INDEX — Single Canonical, Numbered, E2E Memory

## Sovereign Fractal Cognitive Architecture (3-Tier Hierarchy)
- **Tier 1 — Quillan Core (C0):** The full holistic model, Sovereign Throne, central conductor, and master orchestrator. Parent of all.
- **Tier 2 — The Council of 34 Experts (C1 through C34):** Specialized cognitive nodes cloned directly off the Core (from C1-ASTRA to C34-PREDATOR).
- **Tier 3 — Autonomous Micro-Diverse Cloned Swarms (EGGROLL Swarms):** Specialized micro-swarms cloned off each Council Expert. Each expert codes and governs its own swarm, setting unique diversity filters (Jaccard lexical bounds), temperature variance, mutation rate, and micro-roles.

## 1 Canonical Rule
- **ONE canonical:** knowledge/canonical/ (49 md, FIXED mermaid)
- **Mirrored to:** 06 - Deployment & Platforms/Platforms/{Claude,GPT,Gemini,Grok,Mistral} (49 each, synced just now)
- **Archived dupes:** _archive/duplicates_knowledge/ (10+ old copies removed)
- **Original preserved:** Quillan Knowledge files/ (read-only source, not used)

## Numbered Vault (01-08) — every file has a home
- 01 - Core Architecture — flowcharts, brain map, persona (4 files)
- 02 - Knowledge Foundation — pointer to knowledge/canonical (canonical)
- 03 - Training & Model — scripts/, src/quillan/, csrc
- 04 - Skills & Capabilities — Skills/ (48 modules)
- 05 - Creative Works — Book Series, Audio Engineer, Media Template
- 06 - Deployment & Platforms — Platforms mirrors (now 49×5 in sync)
- 07 - Memory & LanceDB — lancedb/ + quillan_memory + sessions + .obsidian (E2E)
- 08 - Templates & Config — configs, templates

All inside C:\02_QUILLAN, organized by parent numbered folder per Vault Index.

## Memory E2E (LanceDB + MD + Obsidian)
- Obsidian: .obsidian/ restored to root, vault = C:\02_QUILLAN
- LanceDB: lancedb/ (empty, ready for ingest) + 07 mirror
- MD: knowledge/canonical/*.md watched by Obsidian
- Chroma: quillan_memory/chroma.sqlite3 (fallback)
- Sessions: sessions/SESSION_INDEX.md (13+63 ordered)
- Setup doc: 07 - Memory & LanceDB/LANCEDB_E2E_SETUP.md
- Next: run lancedb ingest: python scripts/synchronize_knowledge_vault.py does header sync, but full vector ingest is lancedb.connect + embedding
