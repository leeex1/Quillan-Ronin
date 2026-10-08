#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧠 QUILLAN SECOND BRAIN — HIGH-THROUGHPUT REPOSITORY MAPPER & DOCUMENTER
=======================================================================
High-performance scanner, connection extractor, and documentation compiler
for the 140,000+ file ecosystem in C:\\02_QUILLAN.
Embodying:
  C1-ASTRA (Pattern Recognition & Topology)
  C10-CODEWEAVER (AST Hierarchy & Module Parsing)
  C18-SHEPHERD (Grounding & Reference Auditing)
  C27-CHRONICLE (System Narrative & Auto-Documentation)
"""

import os
import sys
import re
import time
import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, Any, List, Set, Tuple

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

ROOT_DIR = Path(r"C:\02_QUILLAN")
SPIDER_DIR = ROOT_DIR / "09 - Projects" / "mcp" / "quillan_spider"
INDEX_FILE = SPIDER_DIR / "second_brain_index.json"
CLUSTERS_FILE = SPIDER_DIR / "second_brain_clusters.json"
DOCS_FILE = ROOT_DIR / "SYSTEM_TOPOLOGY_MAP.md"

SKIP_DIRS = {
    ".git", "__pycache__", "node_modules", ".venv", ".venv-mazebench",
    ".pytest_cache", ".mypy_cache", "site-packages", ".coverage", ".dist"
}

CODE_EXTS = {".py", ".pyi", ".ts", ".js", ".h", ".cpp", ".c", ".rs", ".go", ".html", ".css"}
DOC_EXTS = {".md", ".txt", ".rst", ".json", ".yaml", ".yml", ".toml"}
MEDIA_EXTS = {".png", ".jpg", ".jpeg", ".svg", ".webp", ".mp3", ".wav"}

# 8 Sovereign Subsystem Sectors
SECTORS = {
    "01_CORE": {"name": "Core Architecture & Cognition", "color": "#79dd8e", "icon": "🧠"},
    "02_KNOWLEDGE": {"name": "Knowledge Base & Obsidian Vaults", "color": "#49e0da", "icon": "📚"},
    "03_TRAINING": {"name": "Training Pipeline & Model Forge", "color": "#ff7b72", "icon": "⚡"},
    "04_SWARM": {"name": "Sovereign Skills & Micro-Swarm", "color": "#a371f7", "icon": "🐝"},
    "05_CREATIVE": {"name": "Creative Works & Worldbuilding", "color": "#f6b84b", "icon": "🎨"},
    "06_DEPLOYMENT": {"name": "Deployment, Dashboards & Telemetry", "color": "#58a6ff", "icon": "🚀"},
    "07_MEMORY": {"name": "Vector Memory & LanceDB RAG", "color": "#3fb950", "icon": "💾"},
    "09_MCP_PROJECTS": {"name": "MCP Servers & Tool Ecosystem", "color": "#d29922", "icon": "⚙️"},
}

def classify_sector(rel_path: str) -> str:
    norm = rel_path.replace("\\", "/").lower()
    if norm.startswith("01 - core") or "cognition" in norm:
        return "01_CORE"
    if norm.startswith("02 - knowledge") or "obsidian" in norm or "knowledge-base" in norm:
        return "02_KNOWLEDGE"
    if norm.startswith("03 - training") or "checkpoints" in norm or "eval" in norm:
        return "03_TRAINING"
    if norm.startswith("04- quillan") or "skills" in norm:
        return "04_SWARM"
    if norm.startswith("05 - creative") or "06_media" in norm or "audio" in norm or "main-images" in norm:
        return "05_CREATIVE"
    if norm.startswith("06 - deployment") or "collector" in norm or "dashboard" in norm:
        return "06_DEPLOYMENT"
    if norm.startswith("07 - memory") or "lancedb" in norm or "rag" in norm:
        return "07_MEMORY"
    if norm.startswith("09 - projects") or norm.startswith("mcp") or "fastmcp" in norm:
        return "09_MCP_PROJECTS"
    return "01_CORE"

def classify_cluster(rel_path: str) -> str:
    parts = Path(rel_path).parts
    if len(parts) >= 2:
        return f"{parts[0]}/{parts[1]}"
    elif len(parts) == 1:
        return parts[0]
    return "root"


class SecondBrainScanner:
    def __init__(self, root: Path = ROOT_DIR):
        self.root = root
        self.total_scanned = 0
        self.indexed_files = []
        self.clusters = defaultdict(lambda: {
            "file_count": 0, "code_files": 0, "doc_files": 0, "media_files": 0,
            "sample_files": [], "dependencies": set(), "sector": "01_CORE"
        })
        self.connections = [] # {from, to, type, reason}
        self.persona_mentions = defaultdict(int)

    def scan_and_map(self, max_inspect_code: int = 500) -> Dict[str, Any]:
        """High-speed scan across root directory extracting structure & connections."""
        t0 = time.time()
        file_map = {} # stem.lower() -> rel_path
        code_candidates = []

        print(f"[*] Starting high-throughput scan of {self.root}...")
        for dirpath, dirnames, filenames in os.walk(self.root):
            # Prune excluded directories in-place for maximum speed
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS and not d.startswith(".tmp")]
            
            for f in filenames:
                self.total_scanned += 1
                full = Path(dirpath) / f
                try:
                    rel = full.relative_to(self.root)
                    rel_str = str(rel)
                except Exception:
                    continue

                ext = full.suffix.lower()
                sector = classify_sector(rel_str)
                cluster = classify_cluster(rel_str)

                c_info = self.clusters[cluster]
                c_info["file_count"] += 1
                c_info["sector"] = sector

                is_code = ext in CODE_EXTS
                is_doc = ext in DOC_EXTS
                is_media = ext in MEDIA_EXTS

                if is_code:
                    c_info["code_files"] += 1
                elif is_doc:
                    c_info["doc_files"] += 1
                elif is_media:
                    c_info["media_files"] += 1

                if len(c_info["sample_files"]) < 12:
                    c_info["sample_files"].append({
                        "name": f, "rel": rel_str, "ext": ext, "is_code": is_code
                    })

                stem_key = full.stem.lower()
                if is_code or is_doc:
                    file_map[stem_key] = rel_str
                    if len(code_candidates) < max_inspect_code and is_code:
                        code_candidates.append(full)

        print(f"[*] Indexed {self.total_scanned:,} total files across {len(self.clusters)} clusters in {time.time()-t0:.2f}s.")
        print(f"[*] Inspecting {len(code_candidates)} core artifacts for deep cross-module couplings...")

        # Connection Extraction
        import_regex = re.compile(r"(?:from\s+([a-zA-Z0-9_\.]+)\s+import|import\s+([a-zA-Z0-9_\.]+)|require\(['\"]([^'\"]+)['\"]\))")
        wikilink_regex = re.compile(r"\[\[([a-zA-Z0-9_\-\s]+)\]\]")
        persona_regex = re.compile(r"C\d{1,2}\-[A-Z]+")

        for full_p in code_candidates:
            try:
                rel = str(full_p.relative_to(self.root))
                txt = full_p.read_text(encoding="utf-8", errors="ignore")[:8000] # Inspect first 8KB
            except Exception:
                continue

            from_cluster = classify_cluster(rel)

            # 1. Imports / Requires
            for m in import_regex.finditer(txt):
                raw_mod = m.group(1) or m.group(2) or m.group(3)
                if not raw_mod:
                    continue
                clean_mod = raw_mod.split(".")[0].lower()
                if clean_mod in file_map and file_map[clean_mod] != rel:
                    target_rel = file_map[clean_mod]
                    to_cluster = classify_cluster(target_rel)
                    self.connections.append({
                        "from": rel,
                        "to": target_rel,
                        "from_cluster": from_cluster,
                        "to_cluster": to_cluster,
                        "type": "IMPORT",
                        "why": f"Imports {clean_mod}"
                    })
                    if from_cluster != to_cluster:
                        self.clusters[from_cluster]["dependencies"].add(to_cluster)

            # 2. Markdown / Wikilinks
            for w in wikilink_regex.finditer(txt):
                target_stem = w.group(1).strip().lower()
                if target_stem in file_map:
                    target_rel = file_map[target_stem]
                    to_cluster = classify_cluster(target_rel)
                    self.connections.append({
                        "from": rel,
                        "to": target_rel,
                        "from_cluster": from_cluster,
                        "to_cluster": to_cluster,
                        "type": "WIKILINK",
                        "why": f"Obsidian link [[{w.group(1)}]]"
                    })
                    if from_cluster != to_cluster:
                        self.clusters[from_cluster]["dependencies"].add(to_cluster)

            # 3. Sovereign Persona mentions
            for pm in persona_regex.findall(txt):
                self.persona_mentions[pm] += 1

        # Convert sets to lists for JSON serialization
        serialized_clusters = {}
        for c_name, c_data in self.clusters.items():
            serialized_clusters[c_name] = {
                "name": c_name,
                "sector": c_data["sector"],
                "file_count": c_data["file_count"],
                "code_files": c_data["code_files"],
                "doc_files": c_data["doc_files"],
                "media_files": c_data["media_files"],
                "dependencies": sorted(list(c_data["dependencies"])),
                "sample_files": c_data["sample_files"]
            }

        elapsed = round(time.time() - t0, 2)
        summary = {
            "timestamp": time.time(),
            "elapsed_s": elapsed,
            "total_files": self.total_scanned,
            "total_clusters": len(self.clusters),
            "total_connections": len(self.connections),
            "sectors": SECTORS,
            "clusters": serialized_clusters,
            "connections": self.connections[:2000], # Top 2000 deep connections
            "persona_mentions": dict(self.persona_mentions)
        }

        # Persist index
        SPIDER_DIR.mkdir(parents=True, exist_ok=True)
        INDEX_FILE.write_text(json.dumps(summary, indent=1), encoding="utf-8")
        print(f"[✓] Second Brain index saved to {INDEX_FILE} ({len(self.connections)} synaptic connections mapped).")

        # Generate System Documentation
        self.generate_documentation(summary)
        return summary

    def generate_documentation(self, summary: Dict[str, Any]) -> str:
        """Compile comprehensive Markdown system topology documentation."""
        doc = [
            "# 🧠 QUILLAN SOVEREIGN SECOND BRAIN — MASTER SYSTEM TOPOLOGY MAP",
            "==================================================================",
            f"*Autonomous Documentation Generated by Spider Bot Crawler | {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}*",
            "",
            "## 1. Executive System Overview",
            f"- **Total Workspace Artifacts Mapped**: **{summary['total_files']:,} files**",
            f"- **Active Subsystem Clusters**: **{summary['total_clusters']} architectural modules**",
            f"- **Woven Synaptic Connections**: **{summary['total_connections']:,} inter-module couplings**",
            "- **Central Anchor Nucleus**: **Quillan C0 Core Sovereign Throne**",
            "",
            "---",
            "",
            "## 2. Sovereign Subsystem Sectors",
            "| Sector ID | Domain Name | Clusters | File Count | Dominant Role |",
            "| :--- | :--- | :--- | :--- | :--- |"
        ]

        # Group cluster stats by sector
        sector_stats = defaultdict(lambda: {"clusters": 0, "files": 0, "code": 0, "docs": 0})
        for c in summary["clusters"].values():
            s = c["sector"]
            sector_stats[s]["clusters"] += 1
            sector_stats[s]["files"] += c["file_count"]
            sector_stats[s]["code"] += c["code_files"]
            sector_stats[s]["docs"] += c["doc_files"]

        for s_id, s_info in SECTORS.items():
            st = sector_stats[s_id]
            doc.append(f"| `{s_id}` | {s_info['icon']} **{s_info['name']}** | {st['clusters']} clusters | {st['files']:,} files | Code: {st['code']:,} · Docs: {st['docs']:,} |")

        doc.extend([
            "",
            "---",
            "",
            "## 3. Subsystem Inter-Connectivity Matrix (Major Couplings)",
            "The following primary architectural arteries connect the repository's modules together:",
            ""
        ])

        # List top cross-cluster dependencies
        dep_pairs = defaultdict(int)
        for conn in summary["connections"]:
            fc, tc = conn["from_cluster"], conn["to_cluster"]
            if fc != tc:
                dep_pairs[(fc, tc)] += 1

        doc.append("| Origin Module | Target Module | Synaptic Strength | Primary Coupling Type |")
        doc.append("| :--- | :--- | :--- | :--- |")
        for (fc, tc), count in sorted(dep_pairs.items(), key=lambda x: -x[1])[:30]:
            doc.append(f"| `{fc}` | `{tc}` | **{count} links** | `AST_IMPORT / WIRING` |")

        doc.extend([
            "",
            "---",
            "",
            "## 4. Sovereign Persona Resonance Frequency",
            "Frequency of 34-expert persona invocations discovered across system prompts, codebases, and vaults:",
            ""
        ])

        doc.append("| Persona Node | Role Domain | Citations Across Codebase |")
        doc.append("| :--- | :--- | :--- |")
        for persona, count in sorted(summary["persona_mentions"].items(), key=lambda x: -x[1])[:20]:
            doc.append(f"| **{persona}** | Sovereign Council Expert | `{count}` active references |")

        doc.extend([
            "",
            "---",
            "",
            "## 5. Maintenance & Integration Directives",
            "1. **Zero-Egress Isolation**: All MCP servers, training loops, and vector storage operate strictly within the `C:\\02_QUILLAN` root sandbox.",
            "2. **Continuous Synaptic Weaving**: The spider bot continuously sweeps updated files, extracts modified imports, and re-weaves the synaptic silk web lines in real time.",
            "3. **Living Documentation Synchronization**: Any addition or modification to workspace modules triggers an incremental update to this topology document."
        ])

        final_text = "\n".join(doc)
        DOCS_FILE.write_text(final_text, encoding="utf-8")
        print(f"[✓] Comprehensive System Documentation published to {DOCS_FILE}")
        return final_text


if __name__ == "__main__":
    scanner = SecondBrainScanner()
    scanner.scan_and_map(max_inspect_code=800)
