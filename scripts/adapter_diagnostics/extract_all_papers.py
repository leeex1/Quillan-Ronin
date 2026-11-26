#!/usr/bin/env python3
"""Extract ALL formal papers to text on F: for systematic keyword audit."""

from pathlib import Path
import fitz, sys, time

BASE = Path(r"C:\02_QUILLAN\10 - Formal Papers")
OUT = Path(r"F:\quillan_cache\papers_text")
OUT.mkdir(parents=True, exist_ok=True)
pdfs = list(BASE.rglob("*.pdf"))
print(f"found {len(pdfs)} PDFs", flush=True)
t0 = time.time()
for i, p in enumerate(pdfs):
    # stable name incl subfolder to avoid collisions
    rel = p.relative_to(BASE).with_suffix(".txt")
    out = OUT / str(rel).replace("\\", "__")
    if out.exists() and out.stat().st_size > 0:
        continue
    try:
        doc = fitz.open(str(p))
        text = "\n".join(pg.get_text() for pg in doc)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8", errors="ignore")
        doc.close()
    except Exception as e:
        print(f"  FAIL {p.name}: {str(e)[:80]}", flush=True)
    if (i + 1) % 25 == 0:
        print(f"  {i + 1}/{len(pdfs)} ({time.time() - t0:.0f}s)", flush=True)
print(f"DONE {len(pdfs)} PDFs in {time.time() - t0:.0f}s -> {OUT}", flush=True)
