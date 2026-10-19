#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
👑 QUILLAN CANONICAL KNOWLEDGE LANCEDB INGESTOR
---------------------------------------------------------------------------------------
E2E Persistent Vector Indexing for Quillan Memory System:
- Reads all canonical knowledge files from `02 - Knowledge Foundation/knowledge/canonical`
- Chunks text into semantic passages
- Computes 2048-dim embeddings via NVIDIA NIM `nvidia/nemotron-3-embed-1b`
- Stores in LanceDB table `thoughts` at `C:\02_QUILLAN\lancedb`
- Synchronizes with `07 - Memory & LanceDB/lancedb` and `07 - Memory & LanceDB/quillan_memory`
"""

import os
import sys
import time
import json
import shutil
import hashlib
from pathlib import Path
from typing import List, Dict, Any

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import httpx
import pyarrow as pa
import lancedb

ROOT = Path(r"C:\02_QUILLAN")
CANONICAL_DIR = ROOT / "02 - Knowledge Foundation" / "knowledge" / "canonical"
LANCEDB_DIR = ROOT / "lancedb"
LANCEDB_MIRROR_DIR = ROOT / "07 - Memory & LanceDB" / "lancedb"
QUILLAN_MEM_DIR = ROOT / "07 - Memory & LanceDB" / "quillan_memory"

NIM_BASE = "https://integrate.api.nvidia.com/v1"
EMBED_MODEL = "nvidia/nemotron-3-embed-1b"
CHUNK_SIZE = 900
CHUNK_OVERLAP = 90
BATCH_SIZE = 32

def get_api_key() -> str:
    key = os.environ.get("NVIDIA_API_KEY", "")
    if key:
        return key.strip()
    if os.name == "nt":
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Environment") as regkey:
                val, _ = winreg.QueryValueEx(regkey, "NVIDIA_API_KEY")
                if val:
                    return str(val).strip()
        except Exception:
            pass
    env_file = ROOT / ".env"
    if env_file.exists():
        try:
            for line in env_file.read_text(encoding="utf-8").splitlines():
                if line.startswith("NVIDIA_API_KEY="):
                    return line.split("=", 1)[1].strip().strip("\"'")
        except Exception:
            pass
    return ""

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if len(chunk) > 40:
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

def embed_batch(http_client: httpx.Client, api_key: str, texts: List[str], max_retries: int = 4) -> List[List[float]]:
    delay = 1.0
    for attempt in range(max_retries):
        try:
            r = http_client.post(
                f"{NIM_BASE}/embeddings",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={
                    "input": texts,
                    "model": EMBED_MODEL,
                    "input_type": "passage",
                    "encoding_format": "float",
                    "truncate": "END"
                },
                timeout=60.0
            )
            r.raise_for_status()
            data = r.json()["data"]
            return [d["embedding"] for d in sorted(data, key=lambda x: x["index"])]
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"Embedding batch error: {e}", file=sys.stderr)
                raise
            time.sleep(delay * (2 ** attempt))

def main():
    api_key = get_api_key()
    if not api_key:
        print("ERROR: NVIDIA_API_KEY could not be found.")
        sys.exit(1)

    print(f"👑 Starting LanceDB Canonical Knowledge Ingestion")
    print(f"  Source Directory: {CANONICAL_DIR}")
    print(f"  Target Database : {LANCEDB_DIR}")

    LANCEDB_DIR.mkdir(parents=True, exist_ok=True)
    LANCEDB_MIRROR_DIR.mkdir(parents=True, exist_ok=True)
    QUILLAN_MEM_DIR.mkdir(parents=True, exist_ok=True)

    md_files = sorted(list(CANONICAL_DIR.glob("*.md")))
    print(f"  Found {len(md_files)} canonical markdown files.")

    all_chunks: List[Dict[str, Any]] = []
    MAX_CHUNKS_PER_FILE = 20

    for f in md_files:
        try:
            content = f.read_text(encoding="utf-8", errors="replace")
            # If huge masterfile/arxiv dump, take top 40k chars for representative indexing
            if len(content) > 60000:
                content = content[:60000]
            chunks = chunk_text(content)
            if len(chunks) > MAX_CHUNKS_PER_FILE:
                chunks = chunks[:MAX_CHUNKS_PER_FILE]

            for idx, c in enumerate(chunks):
                chunk_id = hashlib.md5(f"{f.name}:{idx}:{c[:30]}".encode()).hexdigest()
                first_line = c.splitlines()[0][:100] if c.splitlines() else f.name
                all_chunks.append({
                    "id": chunk_id,
                    "source": f.name,
                    "blueprint": f.stem,
                    "evolution_event": first_line,
                    "text": c,
                    "timestamp": time.time(),
                })
        except Exception as e:
            print(f"  Warning: failed to read {f.name}: {e}")

    print(f"  Total semantic chunks to embed: {len(all_chunks)}")

    # Embed in batches
    vectors: List[List[float]] = []
    with httpx.Client() as client:
        for i in range(0, len(all_chunks), BATCH_SIZE):
            batch = all_chunks[i:i+BATCH_SIZE]
            batch_texts = [item["text"] for item in batch]
            embs = embed_batch(client, api_key, batch_texts)
            vectors.extend(embs)
            print(f"  Progress: embedded {len(vectors)}/{len(all_chunks)} chunks...", flush=True)

    # Attach vectors
    for item, vec in zip(all_chunks, vectors):
        item["vector"] = vec

    schema = pa.schema([
        pa.field("id", pa.string()),
        pa.field("vector", pa.list_(pa.float32(), 2048)),
        pa.field("timestamp", pa.float64()),
        pa.field("blueprint", pa.string()),
        pa.field("evolution_event", pa.string()),
        pa.field("source", pa.string()),
        pa.field("text", pa.string()),
    ])

    # Connect to primary LanceDB
    db = lancedb.connect(str(LANCEDB_DIR))
    tbl = db.create_table("thoughts", data=all_chunks, schema=schema, mode="overwrite")
    print(f"✅ Created LanceDB table 'thoughts' at {LANCEDB_DIR} with {tbl.count_rows()} rows.")

    # Synchronize to mirrors
    src_lance = LANCEDB_DIR / "thoughts.lance"
    for dest_dir in [LANCEDB_MIRROR_DIR, QUILLAN_MEM_DIR]:
        dest_lance = dest_dir / "thoughts.lance"
        if dest_lance.exists():
            shutil.rmtree(dest_lance)
        shutil.copytree(src_lance, dest_lance)
        print(f"🔄 Synced thoughts.lance -> {dest_lance}")

    # Validation test
    print("\n🔍 Running Validation Vector Query...")
    with httpx.Client() as client:
        test_vec = embed_batch(client, api_key, ["Bushido of Computation and the 34 Chambers"])[0]
    hits = tbl.search(test_vec).limit(3).to_list()
    print(f"Top 3 Hits:")
    for h in hits:
        print(f"  - [{h['source']}] (sim/dist metric: {h.get('_distance', 'N/A')}): {h['text'][:140]}...")

    print("\n🎉 LanceDB Ingestion & Memory Synchronization 100% Complete!")

if __name__ == "__main__":
    main()
