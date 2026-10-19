#!/usr/bin/env python3
"""Delete 05_Training mirror dupes ONLY when (name,size) matches root canonical.
Logs every delete + total freed. Root files never touched."""
import os
from pathlib import Path
MIRROR = Path(r"C:\02_QUILLAN\09 - Projects\projects\05_Training")
ROOT = Path(r"C:\02_QUILLAN")
# one index pass over canonical root (skip mirror + vendor)
index = {}
for cdp, cdns, cfns in os.walk(ROOT, topdown=True):
    cdns[:] = [d for d in cdns if d not in ("node_modules", "__pycache__",
        "05_Training", "Nextverse-protype-app--main") and not d.startswith(".")]
    if "05_Training" in cdp or "Nextverse" in cdp:
        continue
    for f in cfns:
        if f.lower().endswith((".pt", ".bin", ".qbin", ".gguf",
                               ".safetensors")):
            try:
                key = (f.lower(), os.path.getsize(os.path.join(cdp, f)))
                index.setdefault(key, os.path.join(cdp, f))
            except OSError:
                pass
print(f"index: {len(index)} canonical files", flush=True)
freed, n = 0, 0
for dp, dns, fns in os.walk(MIRROR, topdown=True):
    dns[:] = [d for d in dns if d not in ("node_modules", "__pycache__")]
    for f in fns:
        if not f.lower().endswith((".pt", ".bin", ".qbin", ".gguf",
                                   ".safetensors")):
            continue
        mp = os.path.join(dp, f)
        try:
            key = (f.lower(), os.path.getsize(mp))
            if key in index:
                os.remove(mp)
                freed += key[1]
                n += 1
                print(f"DEL {key[1] // 1024 // 1024}MB {mp}", flush=True)
        except OSError as e:
            print(f"SKIP {mp}: {e}", flush=True)
print(f"DONE files={n} freed={freed // 1024 // 1024}MB", flush=True)
