#!/usr/bin/env python3
"""
Quillan Second Brain Ingestion Script
====================================
Populates ChromaDB (quillan_rag_db) with core sovereign knowledge assets
using nvidia/nemotron-3-embed-1b embeddings.
"""

import sys
import asyncio
from pathlib import Path

# Force UTF-8 on Windows console
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Add server directory to path to reuse functions and settings
sys.path.insert(0, str(Path(__file__).parent))
from server import ingest_file, kb_stats, _col

CORE_FILES = [
    r"C:\02_QUILLAN\SOUL.md",
    r"C:\02_QUILLAN\AGENTS.md",
    r"C:\02_QUILLAN\02 - Knowledge Foundation\IDENTITY.md",
    r"C:\02_QUILLAN\02 - Knowledge Foundation\INTENT.md",
    r"C:\02_QUILLAN\02 - Knowledge Foundation\LINEAGE.md",
    r"C:\02_QUILLAN\02 - Knowledge Foundation\00_VAULT_INDEX.md",
    r"C:\02_QUILLAN\02 - Knowledge Foundation\Quillan Knowledge files\9-Quillan Brain mapping.md",
    r"C:\02_QUILLAN\02 - Knowledge Foundation\Quillan Knowledge files\31- Autobiography.md",
    r"C:\02_QUILLAN\02 - Knowledge Foundation\Quillan Knowledge files\6-prime_covenant_codex.md",
    r"C:\02_QUILLAN\02 - Knowledge Foundation\Quillan Knowledge files\TheRoninFlowState.md",
    r"C:\02_QUILLAN\02 - Knowledge Foundation\Quillan Knowledge files\27-Quillan operational manual.md",
    r"C:\02_QUILLAN\02 - Knowledge Foundation\Quillan Knowledge files\Thinking within LLMS.md",
    r"C:\02_QUILLAN\02 - Knowledge Foundation\Quillan Knowledge files\8-Formulas.md",
    r"C:\02_QUILLAN\09 - Projects\projects\Software Engineer\Quillan-XSWE.md",
    r"C:\02_QUILLAN\09 - Projects\projects\Audio Engineer\album checklist.md",
]

async def main():
    print(f"=== Initializing Quillan Second Brain Ingestion ===")
    print(f"Current DB chunk count: {_col.count()}")
    
    success = 0
    failed = 0
    for file_str in CORE_FILES:
        p = Path(file_str)
        if not p.exists():
            print(f"⚠️ Missing file: {file_str}")
            failed += 1
            continue
        try:
            print(f"Ingesting: {p.name} ...", end=" ", flush=True)
            res = await ingest_file(str(p), metadata={"type": "canonical_core"})
            print(f"{res}")
            if "✅" in res:
                success += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ Error: {e}")
            failed += 1
        await asyncio.sleep(0.5)

    print("\n=== Ingestion Complete ===")
    print(f"Success: {success}, Failed: {failed}")
    stats = await kb_stats()
    print("\n" + stats)

if __name__ == "__main__":
    asyncio.run(main())
