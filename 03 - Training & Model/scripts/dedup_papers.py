"""Content-hash dedup across ALL paper locations. Ingest truly-missing unique docs."""
import hashlib
import json
import re
from pathlib import Path
from pypdf import PdfReader
REPO = Path(r"C:\02_QUILLAN")
TD = REPO / "training_data"
DIRS = [REPO / "10 - Formal Papers" / "Formal Papers",
        REPO / "02 - Knowledge Foundation" / "knowledge" / "papers" / "Formal Papers",
        REPO / "02 - Knowledge Foundation" / "Quillan Knowledge files",
        REPO / "10 - Formal Papers",
        REPO / "10 - Formal Papers" / "_archive_papers",
        REPO / "10 - Formal Papers" / "Formal Papers" / "big deal folder",
        REPO / "09 - Projects" / "projects" / "Nextverse-protype-app--main" / "research papers"]
allpdfs = []
for d in DIRS:
    if d.exists():
        allpdfs.extend(d.glob("*.pdf"))
print(f"pdf files found: {len(allpdfs)}", flush=True)
byhash = {}
for p in allpdfs:
    try:
        h = hashlib.md5(p.read_bytes()).hexdigest()
        byhash.setdefault(h, []).append(p)
    except Exception as e:
        print(f"  UNREADABLE FILE {p.name}: {str(e)[:80]}", flush=True)
print(f"unique by content-hash: {len(byhash)}", flush=True)
dups = sum(len(v) - 1 for v in byhash.values())
print(f"duplicate copies: {dups}", flush=True)
corpus = [json.loads(l) for l in open(TD / "pdf_papers_corpus.jsonl", encoding="utf-8") if l.strip()]
have_src = set()
for r in corpus:
    s = r.get("source", "")
    if s.startswith("file:"):
        have_src.add(s[5:].lower())
    have_src.add((r.get("title") or "")[:60].lower())
n = 0
with open(TD / "pdf_papers_corpus.jsonl", "a", encoding="utf-8") as f:
    for h, paths in sorted(byhash.items()):
        rep = paths[0]
        if rep.name.lower() in have_src:
            continue
        try:
            rdr = PdfReader(str(rep))
            text = "\n\n".join([(pg.extract_text() or "") for pg in rdr.pages])
            if len(text.strip()) < 500:
                print(f"  SKIP (unreadable): {rep.name}", flush=True)
                continue
            m = re.search(r"Paper Title / File:\s*(\S+)", text[:500])
            f.write(json.dumps({"text": text[:120000], "source": "file:" + rep.name,
                                "title": (m.group(0)[:150] if m else rep.stem),
                                "copies": len(paths)}) + "\n")
            n += 1
            print(f"  + {rep.name} ({len(text)} chars, {len(paths)} copies)", flush=True)
        except Exception as e:
            print(f"  FAIL {rep.name}: {str(e)[:100]}", flush=True)
print(f"INGESTED {n} unique; corpus now {len(corpus) + n}", flush=True)
