"""Reconcile: corpus file: sources vs Formal Papers folder vs strays. Ingest what's missing."""
import json
import re
from pathlib import Path
from pypdf import PdfReader
REPO = Path(r"C:\02_QUILLAN")
TD = REPO / "training_data"
corpus = [json.loads(l) for l in open(TD / "pdf_papers_corpus.jsonl", encoding="utf-8") if l.strip()]
have = set()
for r in corpus:
    s = r.get("source", "")
    if s.startswith("file:"):
        have.add(s[5:].lower())
    elif s.startswith("arxiv:"):
        have.add((s + ".pdf").lower())
print(f"corpus rows: {len(corpus)}, with file ids: {len(have)}", flush=True)
F10 = REPO / "10 - Formal Papers" / "Formal Papers"
folder = {p.name.lower() for p in F10.glob("*.pdf")}
print(f"10-folder pdfs: {len(folder)}", flush=True)
missing_folder = sorted(folder - have)
print(f"folder PDFs NOT in corpus ({len(missing_folder)}):", flush=True)
for m in missing_folder:
    print("  MISS:", m, flush=True)
strays = []
for d in [REPO / "10 - Formal Papers",
          REPO / "10 - Formal Papers" / "_archive_papers",
          F10 / "big deal folder",
          REPO / "09 - Projects" / "projects" / "Nextverse-protype-app--main" / "research papers"]:
    if d.exists():
        for p in d.glob("*.pdf"):
            if p.name.lower() not in have and p.name.lower() not in folder:
                strays.append(p)
print(f"stray PDFs not in corpus ({len(strays)}):", flush=True)
for p in strays:
    print("  STRAY:", p.name, flush=True)
n = 0
with open(TD / "pdf_papers_corpus.jsonl", "a", encoding="utf-8") as f:
    for name in missing_folder:
        p = F10 / next(x for x in F10.glob("*.pdf") if x.name.lower() == name)
        todo = [(p, "file:" + p.name)]
    for p in strays:
        todo.append((p, "file:" + p.name))
    for p, src in todo:
        try:
            rdr = PdfReader(str(p))
            text = "\n\n".join([(pg.extract_text() or "") for pg in rdr.pages])
            if len(text.strip()) < 500:
                print(f"  SKIP (unreadable): {p.name}", flush=True)
                continue
            m = re.search(r"Paper Title / File:\s*(\S+)", text[:500])
            f.write(json.dumps({"text": text[:120000], "source": src,
                                "title": (m.group(0)[:150] if m else p.stem)}) + "\n")
            n += 1
            print(f"  + {p.name} ({len(text)} chars)", flush=True)
        except Exception as e:
            print(f"  FAIL {p.name}: {str(e)[:120]}", flush=True)
print(f"INGESTED {n}; corpus now {len(corpus) + n}", flush=True)
