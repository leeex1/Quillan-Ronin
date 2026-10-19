import json
from pathlib import Path
from pypdf import PdfReader
REPO = Path(r"C:\02_QUILLAN")
pdf = REPO / "training_data" / "evoontology_2609.15779.pdf"
rdr = PdfReader(str(pdf))
pages = []
for i, pg in enumerate(rdr.pages):
    try:
        pages.append(pg.extract_text() or "")
    except Exception as e:
        print(f"page {i} skip: {e}", flush=True)
text = "\n\n".join(pages)
print(f"pages={len(pages)} chars={len(text)}", flush=True)
row = {"text": text[:120000],
       "source": "arxiv:2609.15779",
       "title": "EvoOntology: A Self-Evolving Ontology Layer for Data Agents",
       "authors": ["Meiduo Chong", "Shaolei Zhang", "Ju Fan", "Xiaoyong Du"],
       "subjects": ["cs.AI", "cs.CL", "cs.DB"]}
with open(REPO / "training_data" / "pdf_papers_corpus.jsonl", "a", encoding="utf-8") as f:
    f.write(json.dumps(row) + "\n")
print("APPENDED to pdf_papers_corpus.jsonl", flush=True)
