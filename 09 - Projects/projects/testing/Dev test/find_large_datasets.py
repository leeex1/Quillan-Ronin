#!/usr/bin/env python3
import os, sys
from pathlib import Path

print("==================================================", flush=True)
print("  SCANNING FOR ALL DATASET FILES & REPOSITORIES", flush=True)
print("==================================================", flush=True)

scan_roots = [
    r"C:\02_QUILLAN",
    r"C:\Users\Admin\Downloads",
    r"C:\Users\Admin\Documents",
    r"C:\Users\Admin\Desktop",
    r"C:\Users\Admin\data",
    r"C:\data",
    r"C:\datasets",
]

exts = {".jsonl", ".parquet", ".arrow", ".bin", ".txt", ".json", ".csv", ".tar", ".gz", ".zip", ".zst", ".7z"}

found = []

for root_path in scan_roots:
    if not os.path.exists(root_path): continue
    print(f"[*] Scanning: {root_path}...", flush=True)
    for root, dirs, files in os.walk(root_path):
        # skip git and venv
        if ".git" in root or "node_modules" in root or "__pycache__" in root:
            continue
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if ext in exts:
                fpath = os.path.join(root, f)
                try:
                    sz = os.path.getsize(fpath)
                    if sz > 1 * 1024 * 1024:  # > 1 MB
                        found.append((sz, fpath))
                except:
                    pass

found.sort(reverse=True, key=lambda x: x[0])

print(f"\n[+] Total Large Dataset Files Found: {len(found)}", flush=True)
total_sz = sum(x[0] for x in found)
print(f"[+] Total Combined Size: {total_sz / (1024**3):.2f} GB\n", flush=True)

print("TOP DATASET FILES BY SIZE:", flush=True)
for sz, p in found[:40]:
    print(f"  {sz / (1024**2):8.2f} MB | {p}", flush=True)
