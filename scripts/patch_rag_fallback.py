import json
from pathlib import Path

path = Path(r"C:\02_QUILLAN\09 - Projects\mcp\quillan_rag\server.py")
content = path.read_text(encoding="utf-8")

old_import = "import httpx, chromadb\nfrom chromadb.config import Settings"
old_import_crlf = "import httpx, chromadb\r\nfrom chromadb.config import Settings"

new_import = """import httpx
try:
    import chromadb
    from chromadb.config import Settings
    _HAS_CHROMADB = True
except ImportError:
    _HAS_CHROMADB = False"""

if old_import_crlf in content:
    content = content.replace(old_import_crlf, new_import.replace("\n", "\r\n"), 1)
elif old_import in content:
    content = content.replace(old_import, new_import, 1)
else:
    print("Warning: Import target not matched")

old_init = """_chroma = chromadb.PersistentClient(path=str(CHROMA_PATH), settings=Settings(anonymized_telemetry=False))\n_col    = _chroma.get_or_create_collection(name=COLLECTION, metadata={"hnsw:space": "cosine"})"""
old_init_crlf = """_chroma = chromadb.PersistentClient(path=str(CHROMA_PATH), settings=Settings(anonymized_telemetry=False))\r\n_col    = _chroma.get_or_create_collection(name=COLLECTION, metadata={"hnsw:space": "cosine"})"""

fallback_code = """if _HAS_CHROMADB:
    _chroma = chromadb.PersistentClient(path=str(CHROMA_PATH), settings=Settings(anonymized_telemetry=False))
    _col    = _chroma.get_or_create_collection(name=COLLECTION, metadata={"hnsw:space": "cosine"})
else:
    import math

    class _NativeCollectionFallback:
        def __init__(self, name: str, base_path: Path):
            self.name = name
            self.base_path = Path(base_path)
            self.file_path = self.base_path / f"{name}_store.json"
            self._data = {}
            self._load()

        def _load(self):
            if self.file_path.exists():
                try:
                    self._data = json.loads(self.file_path.read_text(encoding="utf-8"))
                except Exception:
                    self._data = {}

        def _save(self):
            try:
                self.base_path.mkdir(parents=True, exist_ok=True)
                self.file_path.write_text(json.dumps(self._data), encoding="utf-8")
            except Exception:
                pass

        def count(self) -> int:
            return len(self._data)

        def upsert(self, ids: List[str], embeddings: List[List[float]], documents: List[str], metadatas: List[Dict[str, Any]]):
            for i, doc_id in enumerate(ids):
                self._data[doc_id] = {
                    "document": documents[i] if i < len(documents) else "",
                    "embedding": embeddings[i] if i < len(embeddings) else [],
                    "metadata": metadatas[i] if i < len(metadatas) else {},
                }
            self._save()

        def get(self, limit: Optional[int] = None, where: Optional[Dict[str, Any]] = None, include: Optional[List[str]] = None):
            res_ids, res_docs, res_metas, res_embs = [], [], [], []
            for doc_id, item in self._data.items():
                if where:
                    match = True
                    for k, v in where.items():
                        if item.get("metadata", {}).get(k) != v:
                            match = False
                            break
                    if not match:
                        continue
                res_ids.append(doc_id)
                res_docs.append(item.get("document", ""))
                res_metas.append(item.get("metadata", {}))
                res_embs.append(item.get("embedding", []))
                if limit and len(res_ids) >= limit:
                    break
            out = {"ids": res_ids}
            if include is None or "documents" in include:
                out["documents"] = res_docs
            if include is None or "metadatas" in include:
                out["metadatas"] = res_metas
            if include and "embeddings" in include:
                out["embeddings"] = res_embs
            return out

        def delete(self, ids: List[str]):
            for doc_id in ids:
                self._data.pop(doc_id, None)
            self._save()

        def query(self, query_embeddings: List[List[float]], n_results: int = 10, include: Optional[List[str]] = None):
            if not query_embeddings or not self._data:
                return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}
            q_emb = query_embeddings[0]
            scores = []
            q_norm = math.sqrt(sum(x * x for x in q_emb)) or 1e-9
            for doc_id, item in self._data.items():
                d_emb = item.get("embedding", [])
                if not d_emb:
                    sim = 0.0
                else:
                    d_norm = math.sqrt(sum(y * y for y in d_emb)) or 1e-9
                    dot = sum(x * y for x, y in zip(q_emb, d_emb))
                    sim = dot / (q_norm * d_norm)
                dist = max(0.0, 1.0 - sim)
                scores.append((dist, doc_id, item))
            scores.sort(key=lambda x: x[0])
            top = scores[:n_results]
            return {
                "ids": [[x[1] for x in top]],
                "documents": [[x[2].get("document", "") for x in top]],
                "metadatas": [[x[2].get("metadata", {}) for x in top]],
                "distances": [[x[0] for x in top]],
            }

    class _NativeChromaFallback:
        def __init__(self, path: Path):
            self.path = Path(path)
            self._cols = {}

        def get_or_create_collection(self, name: str, metadata: Optional[Dict[str, Any]] = None):
            if name not in self._cols:
                self._cols[name] = _NativeCollectionFallback(name, self.path)
            return self._cols[name]

        def get_collection(self, name: str):
            if name in self._cols:
                return self._cols[name]
            return _NativeCollectionFallback(name, self.path)

        def list_collections(self):
            class _ColStub:
                def __init__(self, name):
                    self.name = name
            return [_ColStub(c) for c in self._cols.keys()]

        def delete_collection(self, name: str):
            self._cols.pop(name, None)
            store_file = self.path / f"{name}_store.json"
            if store_file.exists():
                try:
                    store_file.unlink()
                except Exception:
                    pass

    _chroma = _NativeChromaFallback(path=CHROMA_PATH)
    _col    = _chroma.get_or_create_collection(name=COLLECTION)"""

if old_init_crlf in content:
    content = content.replace(old_init_crlf, fallback_code.replace("\n", "\r\n"), 1)
elif old_init in content:
    content = content.replace(old_init, fallback_code, 1)
else:
    print("Warning: Init target not matched")

path.write_text(content, encoding="utf-8")
print("SUCCESS")

