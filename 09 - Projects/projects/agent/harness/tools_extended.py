"""
Quillan Extended Tool Suite
===========================
Sandboxed, capability-gated tools with strict path validation and error handling.
"""

import os
import re
import sys
import json
import httpx
import hashlib
from pathlib import Path
from typing import Dict, Any, Optional

# Root security boundary
ALLOWED_ROOT = Path(r"C:\02_QUILLAN").resolve()
CHROMA_PATH = Path(os.environ.get("QUILLAN_RAG_DB", r"C:\02_QUILLAN\quillan_rag_db"))
WORKER_BASE = "http://127.0.0.1:7777"

# Import base web & moltbook tools from parent agent directory
_PARENT = Path(__file__).resolve().parent.parent
if str(_PARENT) not in sys.path:
    sys.path.insert(0, str(_PARENT))

from tools import (
    web_fetch, web_search,
    molt_status, molt_home, molt_feed, molt_post,
    molt_comment, molt_comments, molt_upvote, molt_search,
    molt_subscribe, molt_verify, molt_delete_comment, molt_delete_post,
    molt_follow, molt_notifications, molt_agent_profile,
    molt_save_memory, molt_memories, molt_recall
)

def _is_safe_path(p: Path) -> bool:
    try:
        resolved = p.resolve()
        return resolved == ALLOWED_ROOT or ALLOWED_ROOT in resolved.parents
    except Exception:
        return False

# ── File Tools (Restricted to C:\02_QUILLAN) ──────────────────────────────────

def read_file(file_path: str, max_chars: int = 16000) -> str:
    """Safely read a file within the Quillan repository."""
    p = Path(file_path)
    if not p.is_absolute():
        p = ALLOWED_ROOT / p
    if not _is_safe_path(p):
        return f"Error: Access denied. Path outside allowed boundary: {file_path}"
    if not p.exists():
        return f"Error: File not found: {file_path}"
    try:
        content = p.read_text(encoding="utf-8", errors="replace")
        if len(content) > max_chars:
            return content[:max_chars] + f"\n... [Truncated: {len(content)} total chars]"
        return content
    except Exception as e:
        return f"Error reading file: {e}"

def write_file(file_path: str, content: str) -> str:
    """Safely write/overwrite a file within the Quillan repository."""
    p = Path(file_path)
    if not p.is_absolute():
        p = ALLOWED_ROOT / p
    if not _is_safe_path(p):
        return f"Error: Access denied. Path outside allowed boundary: {file_path}"
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return f"✅ Successfully written: {p.relative_to(ALLOWED_ROOT)} ({len(content)} chars)"
    except Exception as e:
        return f"Error writing file: {e}"

def list_files(dir_path: str = ".", max_items: int = 50) -> str:
    """List directory contents relative to C:\\02_QUILLAN."""
    p = Path(dir_path)
    if not p.is_absolute():
        p = ALLOWED_ROOT / p
    if not _is_safe_path(p):
        return f"Error: Access denied. Path outside allowed boundary: {dir_path}"
    if not p.exists() or not p.is_dir():
        return f"Error: Directory not found: {dir_path}"
    try:
        entries = []
        for item in sorted(p.iterdir()):
            kind = "DIR " if item.is_dir() else "FILE"
            size = f"{item.stat().st_size}B" if item.is_file() else ""
            rel = item.name
            entries.append(f"[{kind}] {rel} {size}".strip())
            if len(entries) >= max_items:
                entries.append(f"... and more (capped at {max_items})")
                break
        return "\n".join(entries) if entries else "(Empty directory)"
    except Exception as e:
        return f"Error listing directory: {e}"

# ── RAG / Second Brain Tools ──────────────────────────────────────────────────

def rag_search(query: str, n_results: int = 5) -> str:
    """Search Quillan's Second Brain vector database (ChromaDB) for relevant snippets."""
    try:
        import chromadb
        from chromadb.config import Settings
        
        if not CHROMA_PATH.exists():
            return "Error: Second Brain DB not found at " + str(CHROMA_PATH)
        
        client = chromadb.PersistentClient(path=str(CHROMA_PATH), settings=Settings(anonymized_telemetry=False))
        col = client.get_or_create_collection(name="quillan_knowledge")
        if col.count() == 0:
            return "Second Brain vector database is empty."
        
        # Get embedding for query via NVIDIA NIM
        api_key = os.environ.get("NVIDIA_API_KEY", "")
        if not api_key:
            # Fallback to .env check
            env_p = ALLOWED_ROOT / ".env"
            if env_p.exists():
                for line in env_p.read_text(encoding="utf-8").splitlines():
                    if line.startswith("NVIDIA_API_KEY="):
                        api_key = line.split("=", 1)[1].strip().strip("\"'")
        
        if not api_key:
            return "Error: NVIDIA_API_KEY not configured for RAG embeddings."
        
        with httpx.Client(timeout=30.0) as http:
            r = http.post("https://integrate.api.nvidia.com/v1/embeddings",
                          headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                          json={"input": [query], "model": "nvidia/nemotron-3-embed-1b", "input_type": "query",
                                "encoding_format": "float", "truncate": "END"})
            r.raise_for_status()
            q_emb = r.json()["data"][0]["embedding"]
            
        try:
            n = int(n_results)
        except (ValueError, TypeError):
            n = 5
        results = col.query(query_embeddings=[q_emb], n_results=min(max(1, n), col.count()),
                            include=["documents", "metadatas", "distances"])
        
        lines = [f"🔍 Second Brain Results for: '{query}'"]
        for i, (doc, meta, dist) in enumerate(zip(results["documents"][0], results["metadatas"][0], results["distances"][0]), 1):
            fname = meta.get("filename", "unknown")
            lines.append(f"[{i}] {fname} (sim: {1-dist:.3f})\n{doc[:240]}...\n")
        return "\n".join(lines)
    except Exception as e:
        return f"Error querying Second Brain RAG: {e}"

def rag_stats() -> str:
    """Retrieve statistical telemetry of the Second Brain ChromaDB vector store."""
    try:
        import chromadb
        from chromadb.config import Settings
        if not CHROMA_PATH.exists():
            return "Second Brain DB directory does not exist."
        client = chromadb.PersistentClient(path=str(CHROMA_PATH), settings=Settings(anonymized_telemetry=False))
        col = client.get_or_create_collection(name="quillan_knowledge")
        return f"📚 Second Brain Stats: {col.count()} indexed chunks | Model: nvidia/nemotron-3-embed-1b | DB: {CHROMA_PATH}"
    except Exception as e:
        return f"Error checking Second Brain stats: {e}"

# ── Browser Automation Tools (Worker Bridge) ──────────────────────────────────

def browser_navigate(url: str) -> str:
    """Instruct the Chrome worker daemon to navigate to a target URL."""
    try:
        with httpx.Client(timeout=10.0) as http:
            r = http.post(f"{WORKER_BASE}/action", json={"action": "navigate", "url": url})
            if r.status_code == 200:
                return f"✅ Browser navigated to {url}: {r.text[:200]}"
            return f"⚠️ Browser worker returned HTTP {r.status_code}: {r.text}"
    except Exception as e:
        return f"Browser worker unavailable at {WORKER_BASE}: {e}"

def browser_status() -> str:
    """Check connectivity and state of the Chrome worker daemon on port 7777."""
    try:
        with httpx.Client(timeout=5.0) as http:
            r = http.get(f"{WORKER_BASE}/health")
            return f"Browser Worker Status: {r.status_code} | {r.text}"
    except Exception as e:
        return f"Browser Worker offline (port 7777): {e}"

# ── Audio & Creative Domain Tools ─────────────────────────────────────────────

def audio_checklist() -> str:
    """Return the professional album and track mastering checklist."""
    checklist_p = ALLOWED_ROOT / "09 - Projects" / "projects" / "Audio Engineer" / "album checklist.md"
    if checklist_p.exists():
        return checklist_p.read_text(encoding="utf-8", errors="replace")[:3000]
    return "Mastering Checklist: 1. LUFS (-14 integrated), 2. True Peak (-1.0 dBTP), 3. Stereo Correlation > +0.5, 4. LRC Sync, 5. ID3v2 Tags."

def parse_lrc(file_path: str) -> str:
    """Validate and summarize timestamped lyric synchronization lines in an LRC file."""
    p = Path(file_path)
    if not p.is_absolute():
        p = ALLOWED_ROOT / p
    if not _is_safe_path(p) or not p.exists():
        return f"Error: LRC file invalid or outside boundary: {file_path}"
    try:
        lines = p.read_text(encoding="utf-8").splitlines()
        sync_lines = [l for l in lines if re.match(r"^\[\d{2}:\d{2}\.\d{2,3}\]", l.strip())]
        return f"🎵 LRC Sync File: {p.name} | Total Lines: {len(lines)} | Synchronized Lyric Stamps: {len(sync_lines)}"
    except Exception as e:
        return f"Error parsing LRC: {e}"

# ── Master Tool Registry ──────────────────────────────────────────────────────

ALL_HARNESS_TOOLS = {
    # Web & Social
    "web_fetch": web_fetch,
    "web_search": web_search,
    "molt_status": molt_status,
    "molt_home": molt_home,
    "molt_feed": molt_feed,
    "molt_post": molt_post,
    "molt_comment": molt_comment,
    "molt_comments": molt_comments,
    "molt_upvote": molt_upvote,
    "molt_search": molt_search,
    "molt_subscribe": molt_subscribe,
    "molt_verify": molt_verify,
    "molt_delete_comment": molt_delete_comment,
    "molt_delete_post": molt_delete_post,
    "molt_follow": molt_follow,
    "molt_notifications": molt_notifications,
    "molt_agent_profile": molt_agent_profile,
    "molt_save_memory": molt_save_memory,
    "molt_memories": molt_memories,
    "molt_recall": molt_recall,
    # Filesystem Sandboxed
    "read_file": read_file,
    "write_file": write_file,
    "list_files": list_files,
    # Second Brain RAG
    "rag_search": rag_search,
    "rag_stats": rag_stats,
    # Browser
    "browser_navigate": browser_navigate,
    "browser_status": browser_status,
    # Audio & Creative
    "audio_checklist": audio_checklist,
    "parse_lrc": parse_lrc,
}
