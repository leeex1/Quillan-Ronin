#!/usr/bin/env python3
"""
Quillan RAG MCP Server - NIM-Powered Enterprise RAG
====================================================
Embedding: nvidia/nemotron-3-embed-1b (2048 dims, confirmed working)
Generator: nvidia/nemotron-3-super-120b-a12b
Storage:   ChromaDB (local persistent)
"""

import os, json, logging, hashlib, asyncio
from pathlib import Path
from typing import Optional
import httpx, chromadb
from chromadb.config import Settings
from fastmcp import FastMCP

def _get_api_key() -> str:
    key = os.environ.get("NVIDIA_API_KEY", "")
    if key:
        return key.strip()
    # Check Windows User Environment if on Windows
    if os.name == "nt":
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Environment") as regkey:
                val, _ = winreg.QueryValueEx(regkey, "NVIDIA_API_KEY")
                if val:
                    return str(val).strip()
        except Exception:
            pass
    # Check local .env if present (gitignored)
    env_file = Path(r"C:\02_QUILLAN\.env")
    if env_file.exists():
        try:
            for line in env_file.read_text(encoding="utf-8").splitlines():
                if line.startswith("NVIDIA_API_KEY="):
                    return line.split("=", 1)[1].strip().strip("\"'")
        except Exception:
            pass
    return ""

NVIDIA_API_KEY = _get_api_key()
NIM_BASE       = "https://integrate.api.nvidia.com/v1"
EMBED_MODEL    = "nvidia/nemotron-3-embed-1b"
GEN_MODEL      = "nvidia/nemotron-3-super-120b-a12b"
CHROMA_PATH    = Path(os.environ.get("QUILLAN_RAG_DB", r"C:\02_QUILLAN\quillan_rag_db"))
COLLECTION     = "quillan_knowledge"
CHUNK_SIZE     = 800
CHUNK_OVERLAP  = 80

logging.basicConfig(level=logging.INFO, format="%(asctime)s [RAG] %(message)s")
log = logging.getLogger("quillan_rag")

_chroma = chromadb.PersistentClient(path=str(CHROMA_PATH), settings=Settings(anonymized_telemetry=False))
_col    = _chroma.get_or_create_collection(name=COLLECTION, metadata={"hnsw:space": "cosine"})

_client_holder = {"client": None, "loop": None}

def _get_http() -> httpx.AsyncClient:
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if _client_holder["client"] is None or _client_holder["client"].is_closed or _client_holder["loop"] != loop:
        _client_holder["client"] = httpx.AsyncClient(timeout=45.0)
        _client_holder["loop"] = loop
    return _client_holder["client"]

mcp     = FastMCP("Quillan RAG", instructions="NIM-powered knowledge retrieval over Quillan's corpus")

# Allowed root directory to prevent path traversal
ALLOWED_ROOT = Path(r"C:\02_QUILLAN").resolve()

# ── Helpers ───────────────────────────────────────────────────────────────────

async def _embed(texts: list[str], input_type: str = "passage", max_retries: int = 4) -> list[list[float]]:
    delay = 1.0
    client = _get_http()
    for attempt in range(max_retries):
        try:
            r = await client.post(f"{NIM_BASE}/embeddings",
                headers={"Authorization": f"Bearer {NVIDIA_API_KEY}", "Content-Type": "application/json"},
                json={"input": texts, "model": EMBED_MODEL, "input_type": input_type,
                      "encoding_format": "float", "truncate": "END"})
            r.raise_for_status()
            return [d["embedding"] for d in sorted(r.json()["data"], key=lambda x: x["index"])]
        except (httpx.HTTPStatusError, httpx.RequestError) as e:
            if attempt == max_retries - 1:
                log.error(f"Embedding failed after {max_retries} attempts: {e}")
                raise
            import random
            sleep_time = delay * (2 ** attempt) + random.uniform(0.1, 0.5)
            log.warning(f"Embedding attempt {attempt+1} failed: {e}. Retrying in {sleep_time:.2f}s...")
            await asyncio.sleep(sleep_time)

async def _generate(system: str, user: str, max_tokens: int = 1024, max_retries: int = 3) -> str:
    delay = 1.5
    client = _get_http()
    for attempt in range(max_retries):
        try:
            r = await client.post(f"{NIM_BASE}/chat/completions",
                headers={"Authorization": f"Bearer {NVIDIA_API_KEY}", "Content-Type": "application/json"},
                json={"model": GEN_MODEL, "messages": [{"role": "system", "content": system},
                      {"role": "user", "content": user}], "max_tokens": max_tokens, "temperature": 0.2},
                timeout=60.0)
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]
        except (httpx.HTTPStatusError, httpx.RequestError) as e:
            if attempt == max_retries - 1:
                log.error(f"Generation failed after {max_retries} attempts: {e}")
                raise
            import random
            sleep_time = delay * (2 ** attempt) + random.uniform(0.2, 0.8)
            log.warning(f"Generation attempt {attempt+1} failed: {e}. Retrying in {sleep_time:.2f}s...")
            await asyncio.sleep(sleep_time)

def _chunk(text: str) -> list[str]:
    words = text.split()
    chunks, i = [], 0
    while i < len(words):
        chunks.append(" ".join(words[i:i+CHUNK_SIZE]))
        i += CHUNK_SIZE - CHUNK_OVERLAP
    return [c for c in chunks if len(c.strip()) > 50]

def _id(src: str, idx: int) -> str:
    return hashlib.md5(f"{src}::{idx}".encode()).hexdigest()

def _is_safe_path(p: Path) -> bool:
    try:
        resolved = p.resolve()
        return resolved == ALLOWED_ROOT or ALLOWED_ROOT in resolved.parents
    except Exception:
        return False

# ── Tools ─────────────────────────────────────────────────────────────────────

@mcp.tool()
async def ingest_file(file_path: str, metadata: Optional[dict] = None) -> str:
    """Ingest a file into Quillan's RAG knowledge base. Supports .txt .md .pdf .py .json .lrc"""
    path = Path(file_path)
    if not _is_safe_path(path):
        return f"Error: Path traversal blocked. Path must reside within {ALLOWED_ROOT}"
    if not path.exists():
        return f"Error: not found: {file_path}"
    try:
        if path.suffix.lower() == ".pdf":
            import pypdf
            text = "\n".join(p.extract_text() or "" for p in pypdf.PdfReader(str(path)).pages)
        else:
            text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        return f"Error reading {path.name}: {e}"

    chunks = _chunk(text)
    if not chunks:
        return f"Skipped {path.name}: too short"

    all_embs = []
    for i in range(0, len(chunks), 10):
        all_embs.extend(await _embed(chunks[i:i+10]))

    ids  = [_id(str(path), i) for i in range(len(chunks))]
    meta = [{"source": str(path), "filename": path.name, "suffix": path.suffix,
             "chunk": i, **(metadata or {})} for i in range(len(chunks))]
    _col.upsert(ids=ids, embeddings=all_embs, documents=chunks, metadatas=meta)
    log.info(f"Ingested {path.name} -> {len(chunks)} chunks")
    return f"✅ {path.name}: {len(chunks)} chunks"

@mcp.tool()
async def ingest_folder(folder_path: str, extensions: Optional[list[str]] = None, metadata: Optional[dict] = None) -> str:
    """Recursively ingest all matching files from a folder. Default exts: .md .txt .py .lrc .json .pdf"""
    exts  = set(extensions or [".md", ".txt", ".py", ".lrc", ".json", ".pdf"])
    root  = Path(folder_path)
    if not _is_safe_path(root):
        return f"Error: Path traversal blocked. Path must reside within {ALLOWED_ROOT}"
    if not root.exists():
        return f"Error: not found: {folder_path}"
    files = [f for f in root.rglob("*") if f.is_file() and f.suffix.lower() in exts]
    if not files:
        return f"No matching files in {folder_path}"
    ok, errs = 0, []
    for f in files:
        try:
            res = await ingest_file(str(f), metadata)
            if "✅" in res: ok += 1
        except Exception as e:
            errs.append(f"{f.name}: {e}")
    out = f"📚 {ok}/{len(files)} files ingested"
    if errs: out += f"\n⚠️ Errors: {'; '.join(errs[:3])}"
    return out

@mcp.tool()
async def search(query: str, n_results: int = 5, filter_source_type: Optional[str] = None) -> str:
    """Semantic search over Quillan's knowledge base. Returns ranked snippets with source info."""
    if _col.count() == 0:
        return "Knowledge base empty. Run ingest_file or ingest_folder first."
    q_emb   = (await _embed([query], "query"))[0]
    where   = {"type": filter_source_type} if filter_source_type else None
    results = _col.query(query_embeddings=[q_emb], n_results=min(n_results, _col.count()),
                         where=where, include=["documents", "metadatas", "distances"])
    lines = [f'🔍 Results for: "{query}"\n']
    for i, (doc, meta, dist) in enumerate(zip(results["documents"][0], results["metadatas"][0], results["distances"][0]), 1):
        lines.append(f"**[{i}] {meta.get('filename','?')}** (sim: {1-dist:.3f})\n{doc[:280]}...\n")
    return "\n".join(lines)

@mcp.tool()
async def ask(question: str, n_context: int = 5, max_tokens: int = 1024) -> str:
    """Ask Quillan's knowledge base a question. RAG-grounded answer with citations."""
    if _col.count() == 0:
        return "Knowledge base empty. Run ingest_file or ingest_folder first."
    q_emb   = (await _embed([question], "query"))[0]
    results = _col.query(query_embeddings=[q_emb], n_results=min(n_context, _col.count()),
                         include=["documents", "metadatas", "distances"])
    ctx_parts, sources = [], []
    for doc, meta, dist in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
        fname = meta.get("filename", "?")
        ctx_parts.append(f"[{fname} | sim:{1-dist:.2f}]\n{doc}")
        if fname not in sources: sources.append(fname)
    context = "\n\n---\n\n".join(ctx_parts)
    system  = ("You are Quillan-Ronin. Answer using ONLY the provided context. "
               "Be precise, cite sources. If context is insufficient, say so.")
    answer  = await _generate(system, f"Context:\n{context}\n\nQuestion: {question}", max_tokens)
    return f"{answer}\n\n**Sources:** {', '.join(sources)}"

@mcp.tool()
async def kb_stats() -> str:
    """Get statistics about Quillan's knowledge base: chunk count, file types, storage info."""
    count = _col.count()
    if count == 0:
        return "Knowledge base is empty."
    sample = _col.get(limit=min(count, 1000), include=["metadatas"])
    types, sources = {}, set()
    for m in sample["metadatas"]:
        ext = m.get("suffix", "?")
        types[ext] = types.get(ext, 0) + 1
        sources.add(m.get("filename", "?"))
    type_str = "\n".join(f"  {k}: {v} chunks" for k, v in sorted(types.items(), key=lambda x: -x[1]))
    return (f"📚 Quillan RAG KB\n  Chunks: {count}\n  Files: {len(sources)}\n"
            f"  Model: {EMBED_MODEL}\n  DB: {CHROMA_PATH}\n\nBy type:\n{type_str}")

@mcp.tool()
async def delete_source(filename: str) -> str:
    """Remove all chunks for a specific file from the knowledge base."""
    results = _col.get(where={"filename": filename}, include=["metadatas"])
    ids = results.get("ids", [])
    if not ids: return f"No chunks found for '{filename}'"
    _col.delete(ids=ids)
    return f"🗑️ Removed {len(ids)} chunks for '{filename}'"

if __name__ == "__main__":
    log.info(f"Quillan RAG | embed={EMBED_MODEL} | gen={GEN_MODEL} | db={CHROMA_PATH}")
    mcp.run(transport="stdio")
