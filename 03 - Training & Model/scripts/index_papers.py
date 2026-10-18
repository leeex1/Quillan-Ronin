import json
import re
from pathlib import Path
p = Path(r"C:\02_QUILLAN\training_data\pdf_papers_corpus.jsonl")
rows = [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
out = []
for r in rows:
    if not r.get("source"):
        m = re.search(r"Paper Title / File:\s*(\S+)", r.get("text", "")[:500])
        if m:
            r["source"] = "file:" + m.group(1)
        m2 = re.search(r"Paper Title / File:[^\n]*\n\n(.{0,120})", r.get("text", "")[:800])
        if m2 and not r.get("title"):
            r["title"] = " ".join(m2.group(1).split())[:150]
    out.append(r)
with open(p, "w", encoding="utf-8") as f:
    for r in out:
        f.write(json.dumps(r) + "\n")
print(f"indexed {len(out)} rows", flush=True)
for r in out:
    s = (r.get("source") or "?")[:60]
    t = (r.get("title") or "")[:70]
    if "evo" in (s + t).lower() or "onto" in (s + t).lower() or "moe" in (s + t).lower() or "evol" in (s + t).lower():
        print(f"EVO-ish: {s} | {t}", flush=True)
