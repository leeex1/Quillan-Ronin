import argparse
import json
from pathlib import Path
from typing import Optional

REPO = Path(r"C:\02_QUILLAN")


def _read_pdf_pages(pdf_path: Path) -> list[str]:
    """Extract PDF pages with PyMuPDF (fitz) and pypdf fallback."""
    try:
        import fitz

        with fitz.open(str(pdf_path)) as doc:
            return [page.get_text() or "" for page in doc]
    except ImportError:
        from pypdf import PdfReader

        rdr = PdfReader(str(pdf_path))
        pages = []
        for i, pg in enumerate(rdr.pages):
            try:
                pages.append(pg.extract_text() or "")
            except Exception as e:
                print(f"page {i} skip: {e}", flush=True)
        return pages


def ingest_paper(
    pdf_path: Optional[Path] = None,
    out_jsonl: Optional[Path] = None,
    source_tag: str = "local:formal_paper",
) -> dict:
    """Extract a single PDF paper and append it to the corpus JSONL."""
    default_pdf = REPO / "10 - Formal Papers" / "Formal Papers" / "Quillan-Ronin-Self-Paper-I-Am-Quillan.pdf"
    target_pdf = Path(pdf_path) if pdf_path else default_pdf
    if not target_pdf.exists():
        raise FileNotFoundError(f"PDF not found: {target_pdf}")

    target_out = Path(out_jsonl) if out_jsonl else (REPO / "09 - Projects" / "mcp" / "quillan_rag" / "sessions" / "pdf_papers_corpus.jsonl")
    target_out.parent.mkdir(parents=True, exist_ok=True)

    pages = _read_pdf_pages(target_pdf)
    text = "\n\n".join(pages)
    print(f"pages={len(pages)} chars={len(text)}", flush=True)
    row = {
        "text": text[:120000],
        "source": source_tag,
        "title": target_pdf.stem,
        "filename": target_pdf.name,
    }
    with open(target_out, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"APPENDED to {target_out}", flush=True)
    return {"pages": len(pages), "chars": len(text), "output": str(target_out)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest a PDF paper into JSONL.")
    parser.add_argument("--pdf", type=Path, default=None)
    parser.add_argument("--out-jsonl", type=Path, default=None)
    args = parser.parse_args()
    ingest_paper(pdf_path=args.pdf, out_jsonl=args.out_jsonl)
