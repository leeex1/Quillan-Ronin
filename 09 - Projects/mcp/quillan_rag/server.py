#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quillan RAG MCP Server - NIM-Powered Enterprise RAG
====================================================
Embedding: nvidia/nemotron-3-embed-1b (2048 dims, confirmed working)
Generator: nvidia/nemotron-3-super-120b-a12b
Storage:   ChromaDB (local persistent)

Added:
  5. rag_full_pipeline — ingest folder -> search -> ask -> kb_stats (persisted)
  6. rag_route — cross-MCP routing payload (council/code/buddy)
  7. rag_health — knowledge base dashboard summary
  8. rag_session_list / rag_session_replay — persistence
"""

import sys
import os
import logging
import builtins

# --- Quillan-Ronin v5.4.0-ONI Stdio Guard ---
# Ensure all logs and prints go strictly to stderr to prevent stdio JSON-RPC corruption
logging.basicConfig(stream=sys.stderr, level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
_orig_print = builtins.print
def _quillan_safe_print(*args, **kwargs):
    if "file" not in kwargs or kwargs["file"] is None or kwargs["file"] is sys.stdout:
        kwargs["file"] = sys.stderr
    return _orig_print(*args, **kwargs)
builtins.print = _quillan_safe_print

# --- Unified Cross-Platform PYTHONPATH Injection ---
_BASE_PATHS = [
    r"C:\02_QUILLAN\09 - Projects\mcp",
    r"C:\02_QUILLAN\09 - Projects\projects\oni",
    r"C:\02_QUILLAN\03 - Training & Model\scripts",
    r"C:\02_QUILLAN",
]
for _bp in _BASE_PATHS:
    if os.name != "nt":
        if _bp.startswith("C:") or _bp.startswith("c:"):
            _bp = "/mnt/c/" + _bp[3:].replace("\\", "/")
    if os.path.exists(_bp) and _bp not in sys.path:
        sys.path.insert(0, _bp)



import os, json, logging, hashlib, asyncio
from pathlib import Path
from typing import Optional, Dict, Any, List
import httpx, chromadb
from chromadb.config import Settings
from fastmcp import FastMCP

def _get_api_key() -> str:
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
# [QUILLAN-v5.4.0-ONI-SHARED-WIRED]
try:
    from _shared import setup as _quillan_setup, tool_guard, ok_response, error_response, validate_string, validate_path, validate_int, read_file_safe
    from _shared.base_engines import PRISM_ENGINE, COUNCIL_ENGINE, GOVERNOR_ENGINE, SANDBOX_ENGINE
    _SHARED_CTX = _quillan_setup(server_name='quillan_rag', mcp_app=mcp)
except Exception as _shared_err:
    import sys as _sys
    print(f'[quillan_rag] _shared warning: {_shared_err}', file=_sys.stderr)

async def _nim_embed(texts, input_type="passage", max_retries=3):
    delay = 1.5
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

# ── 5. FULL PIPELINE (persisted) ──────────────────────────────────────────────

@mcp.tool()
async def rag_full_pipeline(folder_path: str = "", question: str = "", n_context: int = 5,
                           extensions: Optional[list[str]] = None,
                           persist: bool = True) -> Dict[str, Any]:
    """Chain ingest folder -> search -> ask -> kb_stats into one RAG pipeline."""
    ingest_result = await ingest_folder(folder_path, extensions=extensions) if folder_path else {"skipped": True}
    search_result = await search(query=question, n_results=n_context) if question else {"skipped": True}
    ask_result = await ask(question=question, n_context=n_context) if question else {"skipped": True}
    stats_result = kb_stats()

    session_id = f"rag-{int(time.time())}"
    result = {
        "status": "RAG_PIPELINE_COMPLETE",
        "session_id": session_id,
        "ingest": ingest_result,
        "search": search_result,
        "ask": ask_result,
        "stats": stats_result,
        "overall": "PASS" if (not ingest_result.get("skipped") or not search_result.get("skipped")) else "REVIEW",
        "timestamp": time.time(),
    }

    if persist:
        try:
            p = SESSION_DIR / f"{session_id}.json"
            p.write_text(json.dumps({"type": "rag_pipeline", "result": result}, indent=2), encoding="utf-8")
            result["persisted_to"] = str(p)
        except Exception as e:
            result["persist_error"] = str(e)

    return result

# ── 6. ROUTE (cross-MCP) ───────────────────────────────────────────────────

@mcp.tool()
def rag_route(target_mcp: str = "council", payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Cross-MCP routing payload for council/code/buddy integration."""
    if target_mcp.lower() == "council":
        return {"target": "QuillanCouncil", "suggested_tool": "council_full_deliberation",
                "payload": payload or {}, "note": "Route RAG findings to council deliberation."}
    if target_mcp.lower() == "code":
        return {"target": "QuillanCode", "suggested_tool": "codeweaver_full_pipeline",
                "payload": payload or {}, "note": "Route RAG code docs to code scan pipeline."}
    if target_mcp.lower() == "buddy":
        return {"target": "RoninBuddy", "suggested_tool": "buddy_mirror_collaborate",
                "payload": payload or {}, "note": "Route RAG answer to mirror-collaboration."}
    return {"error": f"Unknown target '{target_mcp}'. Use council, code, or buddy."}


# ── 6. VECTOR MEMORY ENGINE (QuillanMemory v5.4.0-ONI) ───────────────────────────────────────

import numpy as np
_MEMORY_STORE: list = []
_MEMORY_KEYS = None

@mcp.tool()
def memory_write(text: str, metadata: Optional[Dict[str, Any]] = None, importance: float = 1.0) -> Dict[str, Any]:
    """Store a text memory in the Quillan vector memory bank for later recall."""
    global _MEMORY_STORE, _MEMORY_KEYS
    if not text or not text.strip():
        return {"error": "text must not be empty"}
    import hashlib, time
    vec = [float(ord(c) % 256) / 255.0 for c in text[:256]]
    while len(vec) < 256:
        vec.append(0.0)
    entry = {
        "id": hashlib.sha256(text.encode("utf-8")).hexdigest()[:12],
        "text": text,
        "metadata": metadata or {},
        "importance": max(0.0, min(1.0, importance)),
        "timestamp": time.time(),
        "vector": vec,
    }
    _MEMORY_STORE.append(entry)
    if len(_MEMORY_STORE) > 4096:
        _MEMORY_STORE = _MEMORY_STORE[-4096:]
    return {"status": "stored", "id": entry["id"], "total_memories": len(_MEMORY_STORE)}

@mcp.tool()
def memory_recall(query: str, top_k: int = 5, recency_bias: float = 0.3) -> Dict[str, Any]:
    """Recall the most relevant memories for a query using cosine similarity + recency bias."""
    if not _MEMORY_STORE:
        return {"status": "empty", "results": [], "total_memories": 0}
    qvec = [float(ord(c) % 256) / 255.0 for c in query[:256]]
    while len(qvec) < 256:
        qvec.append(0.0)
    now = time.time()
    scored = []
    for entry in _MEMORY_STORE:
        # Cosine similarity
        dot = sum(a * b for a, b in zip(qvec, entry["vector"]))
        norm_q = (sum(a**2 for a in qvec) ** 0.5) or 1e-8
        norm_e = (sum(a**2 for a in entry["vector"]) ** 0.5) or 1e-8
        sim = dot / (norm_q * norm_e)
        # Recency bias
        age_hours = (now - entry["timestamp"]) / 3600.0
        recency = 1.0 / (1.0 + age_hours)
        score = (1.0 - recency_bias) * sim + recency_bias * recency
        scored.append({"id": entry["id"], "text": entry["text"], "metadata": entry["metadata"],
                      "importance": entry["importance"], "score": round(score, 4),
                      "age_hours": round(age_hours, 2)})
    scored.sort(key=lambda x: x["score"], reverse=True)
    return {"status": "OK", "results": scored[:top_k], "total_memories": len(_MEMORY_STORE)}

@mcp.tool()
def memory_stats() -> Dict[str, Any]:
    """Return statistics about the vector memory bank."""
    if not _MEMORY_STORE:
        return {"status": "empty", "total_memories": 0}
    import time
    now = time.time()
    ages = [(now - e["timestamp"]) / 3600.0 for e in _MEMORY_STORE]
    return {
        "status": "OK",
        "total_memories": len(_MEMORY_STORE),
        "avg_age_hours": round(sum(ages) / len(ages), 2),
        "max_age_hours": round(max(ages), 2),
        "min_age_hours": round(min(ages), 2),
        "avg_importance": round(sum(e["importance"] for e in _MEMORY_STORE) / len(_MEMORY_STORE), 3),
    }

@mcp.tool()
def memory_clear() -> Dict[str, Any]:
    """Clear all stored memories."""
    global _MEMORY_STORE
    count = len(_MEMORY_STORE)
    _MEMORY_STORE = []
    return {"status": "cleared", "memories_removed": count}

# ── 7. HEALTH DASHBOARD ──────────────────────────────────────────────────────

@mcp.tool()
def rag_health() -> Dict[str, Any]:
    """Knowledge base dashboard summary."""
    count = _col.count()
    sessions = []
    try:
        from pathlib import Path as _P
        session_dir = Path(r"C:\02_QUILLAN\09 - Projects\mcp\quillan_rag\sessions")
        session_dir.mkdir(parents=True, exist_ok=True)
        for p in sorted(session_dir.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True):
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                sessions.append({
                    "session_id": data.get("session_id"),
                    "overall": data.get("overall"),
                    "timestamp": data.get("timestamp"),
                })
            except Exception:
                continue
    except Exception:
        pass
    return {
        "status": "RAG_HEALTH_READY",
        "kb_chunks": count,
        "embedding_model": EMBED_MODEL,
        "generator_model": GEN_MODEL,
        "db_path": str(CHROMA_PATH),
        "sessions": sessions[:5],
        "council_note": "C6-OMNIS integrates knowledge; C18-SHEPHERD grounds truth.",
    }

# ── 8. SESSION PERSISTENCE ──────────────────────────────────────────────────

SESSION_DIR = Path(r"C:\02_QUILLAN\09 - Projects\mcp\quillan_rag\sessions")
SESSION_DIR.mkdir(parents=True, exist_ok=True)

@mcp.tool()
async def rag_session_list() -> Dict[str, Any]:
    """List persisted RAG sessions."""
    sessions = []
    for p in sorted(SESSION_DIR.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            sessions.append({
                "session_id": data.get("session_id"),
                "overall": data.get("overall"),
                "timestamp": data.get("timestamp"),
                "file": str(p),
            })
        except Exception:
            continue
    return {"count": len(sessions), "sessions": sessions}

@mcp.tool()
async def rag_session_replay(session_id: str) -> Dict[str, Any]:
    """Replay a persisted RAG session by ID."""
    path = SESSION_DIR / f"{session_id}.json"
    if not path.exists():
        matches = list(SESSION_DIR.glob(f"{session_id}*.json"))
        if not matches:
            return {"error": f"Session not found: {session_id}"}
        path = matches[0]
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        return {"error": f"Read failed: {e}"}


# --- Quillan-Ronin Global Error-Handling Middleware ---
from fastmcp.server.middleware import Middleware, MiddlewareContext
import logging as _log

class QuillanErrorMiddleware(Middleware):
    async def on_call_tool(self, context, call_next):
        try:
            return await call_next(context)
        except Exception as e:
            _log.error("Tool error: %s", e, exc_info=True)
            return {"status": "ERROR", "error": str(e), "error_type": type(e).__name__}


mcp.add_middleware(QuillanErrorMiddleware())
if __name__ == "__main__":
    log.info(f"Quillan RAG | embed={EMBED_MODEL} | gen={GEN_MODEL} | db={CHROMA_PATH}")
    mcp.run(transport="stdio")
