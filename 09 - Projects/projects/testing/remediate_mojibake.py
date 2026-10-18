#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
👑 QUILLAN-RONIN (2026) — MOJIBAKE & ENCODING REPAIR ENGINE
High-performance tree scanner using os.walk directory pruning to detect and
remediate CP-1252 / UTF-8 double-encoding artifacts in milliseconds.
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

MOJIBAKE_MAP: Dict[str, str] = {
    "\u00e2\u20ac\u2014": "—",       # â€” -> em dash
    "\u00e2\u20ac\u2013": "–",       # â€“ -> en dash
    "\u00e2\u2020\u2019": "→",       # → -> right arrow
    "\u00e2\u2020\u2018": "←",       # ← / arrow
    "\u00e2\u20ac\u0153": "“",       # “ -> left double quote
    "\u00e2\u20ac\u009d": "”",       # â€\x9d -> right double quote
    "\u00e2\u20ac\u2122": "’",       # ’ -> right single quote / apostrophe
    "\u00e2\u20ac\u02dc": "‘",       # ‘ -> left single quote
    "\u00e2\u20ac\u00a2": "•",       # • -> bullet
    "\u00c3\u00a9": "é",             # é -> é
    "\u00c3\u00a8": "è",             # è -> è
    "\u00c3\u00bc": "ü",             # ü -> ü
    "\u00c3\u00b6": "ö",             # ö -> ö
    "\u00c3\u00a4": "ä",             # ä -> ä
    "\u00c2\u00a0": " ",             # Â  -> non-breaking space
}

SKIP_DIRS = {
    ".git", "node_modules", "free-programming-books", "palace_db",
    "__pycache__", ".antigravity", ".devin", ".windsurf", ".antigravity-ide",
    ".vscode-shared", ".obsidian", ".claude", "06_Media", "checkpoints", "quillan_rag_db"
}

EXTENSIONS = {".py", ".md", ".html", ".js", ".json", ".txt", ".yaml", ".yml"}

def scan_and_remediate(root_dir: Path, apply_fixes: bool = False) -> List[Tuple[Path, int]]:
    results = []
    for root, dirs, files in os.walk(str(root_dir)):
        # In-place directory pruning
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS and not d.startswith(".")]
        
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext not in EXTENSIONS:
                continue
                
            file_path = Path(root) / file
            try:
                # Read binary first to fast-check for multi-byte â patterns (0xc3, 0xc2, 0xe2)
                raw_bytes = file_path.read_bytes()
                if b"\xe2\x80" not in raw_bytes and b"\xe2\x86" not in raw_bytes and b"\xc3\xa9" not in raw_bytes:
                    continue
                raw_text = raw_bytes.decode("utf-8", errors="replace")
            except Exception:
                continue
                
            fixed_text = raw_text
            count_for_file = 0
            for bad, good in MOJIBAKE_MAP.items():
                if bad in fixed_text:
                    occurrences = fixed_text.count(bad)
                    count_for_file += occurrences
                    fixed_text = fixed_text.replace(bad, good)
                    
            if count_for_file > 0:
                results.append((file_path, count_for_file))
                if apply_fixes:
                    file_path.write_text(fixed_text, encoding="utf-8")
                    
    return results

def main():
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    root = Path(__file__).resolve().parent.parent
    apply = "--apply" in sys.argv
    print(f"Scanning {root} for mojibake encoding artifacts (Apply={apply})...")
    results = scan_and_remediate(root, apply_fixes=apply)
    total_artifacts = sum(c for _, c in results)
    print(f"Found {len(results)} files with {total_artifacts} total mojibake artifacts.")
    for p, c in results[:35]:
        print(f"  {p.relative_to(root)} ({c} occurrences)")
    if len(results) > 35:
        print(f"  ... and {len(results) - 35} more files.")
    if apply:
        print("[SUCCESS] Successfully repaired all detected mojibake files in strict UTF-8!")

if __name__ == "__main__":
    main()
