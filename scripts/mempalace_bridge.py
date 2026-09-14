#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
👑 QUILLAN-RONIN MEMPALACE COGNITIVE BRIDGE (v5.4.0)
===============================================================================
Provides a production-grade, typed interface connecting Quillan-Ronin's
cognitive architecture (C4-ECHO Memory Continuity, C9-CODEWEAVER, C25-TECHNE)
to the local offline MemPalace knowledge store.

Key Capabilities:
  - Multi-Wing Querying: Query across 'technical', 'consciousness', 'memory', etc.
  - Granular Room Filtering: Restrict queries to specific language/subject rooms.
  - AAAK Compatibility: Supports compact shorthand memory filing.
  - Safe Fallbacks: Graceful degradation when the database is empty or offline.
===============================================================================
"""

from __future__ import annotations

import hashlib
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

LOGGER = logging.getLogger("quillan.mempalace_bridge")


@dataclass(frozen=True)
class PalaceDrawer:
    """Immutable representation of a filed memory drawer."""
    drawer_id: str
    wing: str
    room: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    distance: float = 0.0


class MempalaceBridge:
    """Unified bridge for Quillan to query and file memories in MemPalace."""

    def __init__(self, palace_path: Optional[Path] = None, collection_name: str = "mempalace_drawers"):
        repo_root = Path(r"C:\02_QUILLAN")
        self.repo_root = repo_root if repo_root.exists() else Path(__file__).resolve().parents[3]
        self.palace_path = palace_path or (self.repo_root / "01_Knowledge_Base" / "palace_db")
        self.collection_name = collection_name
        self._client = None
        self._collection = None

    def _get_collection(self, create: bool = False):
        if self._collection is not None:
            return self._collection
        try:
            import chromadb
            self.palace_path.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=str(self.palace_path))
            if create:
                self._collection = self._client.get_or_create_collection(self.collection_name)
            else:
                self._collection = self._client.get_collection(self.collection_name)
            return self._collection
        except Exception as e:
            LOGGER.debug("ChromaDB collection access note: %s", e)
            return None

    def status(self) -> Dict[str, Any]:
        """Return the current health and inventory of the memory palace."""
        col = self._get_collection(create=False)
        if col is None:
            return {
                "active": False,
                "drawer_count": 0,
                "palace_path": str(self.palace_path),
                "collection": self.collection_name,
                "note": "Palace uninitialized or empty",
            }
        try:
            count = col.count()
            return {
                "active": True,
                "drawer_count": count,
                "palace_path": str(self.palace_path),
                "collection": self.collection_name,
            }
        except Exception as e:
            return {"active": False, "error": str(e), "palace_path": str(self.palace_path)}

    def search(
        self,
        query: str,
        n_results: int = 5,
        wing: Optional[str] = None,
        room: Optional[str] = None,
    ) -> List[PalaceDrawer]:
        """
        Perform semantic search across memory drawers.

        Args:
            query: Natural language search string.
            n_results: Maximum drawers to retrieve.
            wing: Optional wing filter (e.g. 'technical').
            room: Optional room filter (e.g. 'python', 'rust').

        Returns:
            List[PalaceDrawer] ordered by semantic relevance.
        """
        col = self._get_collection(create=False)
        if col is None:
            return []

        where_filter = {}
        if wing and room:
            where_filter = {"$and": [{"wing": wing}, {"room": room}]}
        elif wing:
            where_filter = {"wing": wing}
        elif room:
            where_filter = {"room": room}

        kwargs: Dict[str, Any] = {
            "query_texts": [query],
            "n_results": min(n_results, max(1, col.count())),
        }
        if where_filter:
            kwargs["where"] = where_filter

        try:
            results = col.query(**kwargs)
            drawers = []
            if results and "ids" in results and results["ids"]:
                ids = results["ids"][0]
                docs = results.get("documents", [[]])[0]
                metas = results.get("metadatas", [[]])[0]
                dists = results.get("distances", [[]])[0] if "distances" in results else [0.0] * len(ids)

                for i, d_id in enumerate(ids):
                    doc = docs[i] if i < len(docs) else ""
                    meta = metas[i] if i < len(metas) else {}
                    dist = dists[i] if i < len(dists) else 0.0
                    drawers.append(
                        PalaceDrawer(
                            drawer_id=d_id,
                            wing=meta.get("wing", "general"),
                            room=meta.get("room", "general"),
                            content=doc,
                            metadata=meta,
                            distance=float(dist) if dist is not None else 0.0,
                        )
                    )
            return drawers
        except Exception as e:
            LOGGER.error("Search failed in MemPalace: %s", e)
            return []

    def add_drawer(
        self,
        content: str,
        wing: str = "technical",
        room: str = "general",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        File a new memory drawer into a specified wing and room.

        Returns:
            str: Generated unique drawer ID.
        """
        col = self._get_collection(create=True)
        if col is None:
            raise RuntimeError(f"Could not connect to palace collection at {self.palace_path}")

        meta = metadata.copy() if metadata else {}
        meta["wing"] = wing
        meta["room"] = room

        drawer_hash = hashlib.sha256(f"{wing}:{room}:{content}".encode("utf-8")).hexdigest()[:16]
        drawer_id = f"drawer_{drawer_hash}"

        col.upsert(
            ids=[drawer_id],
            documents=[content],
            metadatas=[meta],
        )
        return drawer_id


# Standalone quick sanity check
if __name__ == "__main__":
    bridge = MempalaceBridge()
    print("Bridge Status:", bridge.status())
