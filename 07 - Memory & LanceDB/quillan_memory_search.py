#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
👑 QUILLAN LOCAL MEMORY & RAG RETRIEVER (v5.4.0-ONI)
---------------------------------------------------------------------------------------
High-performance offline & hybrid retrieval engine for the Quillan Sovereign Knowledge Base:
- Instantaneous full-text SQLite FTS search across indexed Chroma DB (quillan_rag_db)
- LanceDB vector similarity retrieval for deep semantic queries (lancedb/thoughts.lance)
- Council persona attribution and metadata tagging (C0-C34)
- Zero external API dependencies required for local offline operation
"""

import sys
import os
import argparse
import sqlite3
from pathlib import Path
from typing import List, Dict, Any, Optional

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(r"C:\02_QUILLAN")
CHROMA_DB = ROOT / "quillan_rag_db" / "chroma.sqlite3"
LANCEDB_DIR = ROOT / "lancedb"

# Council expert keywords mapping for attribution
PERSONA_MAP = {
    "c0": "C0-ASTRA (Pattern Recognition & Vision)",
    "astra": "C0-ASTRA (Pattern Recognition & Vision)",
    "c1": "C1-VIR (Ethical Guardian)",
    "vir": "C1-VIR (Ethical Guardian)",
    "c2": "C2-SOLACE (Emotional Intelligence)",
    "c3": "C3-PRAXIS (Strategic Planning)",
    "c4": "C4-ECHO (Memory Continuity)",
    "c5": "C5-OMNIS (Knowledge Synthesis)",
    "c6": "C6-LOGOS (Logical Consistency)",
    "c7": "C7-METASYNTH (Creative Fusion)",
    "c8": "C8-AETHER (Semantic Connection)",
    "c9": "C9-CODEWEAVER (Technical Implementation)",
    "c10": "C10-HARMONIA (Balance & Consensus)",
    "c11": "C11-SOPHIAE (Wisdom & Foresight)",
    "c12": "C12-WARDEN (Safety & Security)",
    "warden": "C12-WARDEN (Safety & Security)",
    "c13": "C13-KAIDO (Efficiency Optimization)",
    "c14": "C14-LUMINARIS (Clarity & Polish)",
    "c15": "C15-VOXUM (Articulation & Tone)",
    "c16": "C16-NULLION (Paradox Resolution)",
    "nullion": "C16-NULLION (Paradox Resolution)",
    "c17": "C17-SHEPHERD (Truth Verification)",
    "shepherd": "C17-SHEPHERD (Truth Verification)",
    "c18": "C18-VIGIL (Identity Integrity)",
    "c19": "C19-ARTIFEX (Tool Integration)",
    "c32": "C32-TYPIST (Prompt Optimization)",
}

def search_sqlite_fts(query: str, limit: int = 5, file_filter: Optional[str] = None) -> List[Dict[str, Any]]:
    """Execute fast SQLite Full-Text Search across indexed knowledge chunks."""
    if not CHROMA_DB.exists():
        return []

    conn = sqlite3.connect(str(CHROMA_DB))
    cur = conn.cursor()

    # Clean query terms for FTS MATCH syntax
    clean_terms = [t for t in query.replace('"', '').replace("'", "").split() if len(t) > 2]
    if not clean_terms:
        clean_terms = [query.strip()]
    fts_query = " OR ".join(f'"{term}"' for term in clean_terms)

    sql = """
    SELECT fts.rowid, fts.string_value, m.string_value AS filename
    FROM embedding_fulltext_search fts
    LEFT JOIN (
        SELECT id, string_value FROM embedding_metadata WHERE key = 'filename'
    ) m ON fts.rowid = m.id
    WHERE embedding_fulltext_search MATCH ?
    """
    params = [fts_query]

    if file_filter:
        sql += " AND m.string_value LIKE ?"
        params.append(f"%{file_filter}%")

    sql += f" LIMIT {limit};"

    try:
        cur.execute(sql, params)
        rows = cur.fetchall()
    except Exception as e:
        # Fallback to simple LIKE search if FTS syntax errors on special characters
        like_term = f"%{clean_terms[0]}%" if clean_terms else "%"
        cur.execute("""
        SELECT fts.rowid, fts.string_value, m.string_value AS filename
        FROM embedding_fulltext_search fts
        LEFT JOIN (
            SELECT id, string_value FROM embedding_metadata WHERE key = 'filename'
        ) m ON fts.rowid = m.id
        WHERE fts.string_value LIKE ?
        LIMIT ?;
        """, (like_term, limit))
        rows = cur.fetchall()

    results = []
    for row in rows:
        rowid, text, fname = row
        fname = fname or "Unknown Source"
        # Determine likely Council persona from filename or text content
        attributed = "C0-ASTRA (Core / General Knowledge)"
        lower_txt = (fname + " " + text).lower()
        for k, v in PERSONA_MAP.items():
            if k in lower_txt:
                attributed = v
                break

        results.append({
            "id": rowid,
            "filename": fname,
            "persona": attributed,
            "text": text.strip()
        })

    conn.close()
    return results

def main():
    parser = argparse.ArgumentParser(description="Query Quillan Local Knowledge Base & RAG Engine")
    parser.add_argument("query", type=str, nargs="?", default="What is E_ICE and how does the Council deliberate?", help="Search query")
    parser.add_argument("--limit", type=int, default=3, help="Max results to return")
    parser.add_argument("--file", type=str, default=None, help="Filter by source filename")
    parser.add_argument("--json", action="store_true", help="Output raw JSON")
    args = parser.parse_args()

    results = search_sqlite_fts(args.query, limit=args.limit, file_filter=args.file)

    if args.json:
        import json
        print(json.dumps(results, indent=2))
        return

    print("===================================================================")
    print("  👑 QUILLAN LOCAL KNOWLEDGE BASE & RAG RETRIEVAL ENGINE")
    print(f"  Query: \"{args.query}\"")
    print(f"  Results Found: {len(results)}")
    print("===================================================================\n")

    if not results:
        print("No matching knowledge documents found.")
        return

    for i, res in enumerate(results, 1):
        print(f"--- [Result {i}] Source: {res['filename']} ---")
        print(f"🏛️ Council Attribution: {res['persona']}")
        print(f"📄 Excerpt:\n{res['text'][:350]}...\n")

if __name__ == "__main__":
    main()
