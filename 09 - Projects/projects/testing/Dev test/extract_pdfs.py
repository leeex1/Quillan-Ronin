import argparse
import json
from pathlib import Path
from typing import Optional


def _extract_pdf_pages(pdf_path: Path) -> list[str]:
    """Extract text pages from a PDF using PyMuPDF (fitz) with pypdf fallback."""
    try:
        import fitz

        with fitz.open(str(pdf_path)) as doc:
            return [page.get_text() or "" for page in doc]
    except ImportError:
        from pypdf import PdfReader

        reader = PdfReader(str(pdf_path))
        return [page.extract_text() or "" for page in reader.pages]


def extract_pdfs(
    pdf_dir: Optional[Path] = None,
    out_jsonl: Optional[Path] = None,
    max_files: Optional[int] = None,
) -> dict:
    """Extract PDFs recursively into JSONL without deleting or altering sources."""
    repo_root = Path(r"C:\02_QUILLAN")
    default_dir = repo_root / "10 - Formal Papers"
    if not default_dir.exists():
        default_dir = repo_root / "05 - Creative Works"
    target_dir = Path(pdf_dir) if pdf_dir else default_dir
    out_path = Path(out_jsonl) if out_jsonl else (repo_root / "09 - Projects" / "mcp" / "quillan_rag" / "sessions" / "pdf_papers_corpus.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    pdf_files = sorted(target_dir.rglob("*.pdf")) if target_dir.exists() else []
    if max_files is not None and max_files > 0:
        pdf_files = pdf_files[:max_files]
    print(f"[*] Found {len(pdf_files)} PDF files in {target_dir}")

    extracted_count = 0
    total_pages = 0
    errors = []

    with open(out_path, "w", encoding="utf-8") as out_f:
        for pdf_path in pdf_files:
            try:
                pages = _extract_pdf_pages(pdf_path)
                num_pages = len(pages)
                text_blocks = [p.strip() for p in pages if p and len(p.strip()) > 50]
                full_pdf_text = "\n\n".join(text_blocks)
                if len(full_pdf_text) > 200:
                    record = {
                        "text": f"Paper Title / File: {pdf_path.name}\n\n" + full_pdf_text,
                        "meta": {"source": "local_pdf_paper", "filename": pdf_path.name, "pages": num_pages},
                    }
                    out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    extracted_count += 1
                    total_pages += len(text_blocks)
            except Exception as e:
                errors.append(f"{pdf_path.name}: {e}")
                print(f"[!] Error processing {pdf_path.name}: {e}")

    print(f"[COMPLETE] Successfully extracted {extracted_count} PDFs ({total_pages} total pages) into {out_path}")
    return {"extracted_count": extracted_count, "total_pages": total_pages, "output": str(out_path), "errors": errors}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract PDF text corpus using PyMuPDF/pypdf.")
    parser.add_argument("--pdf-dir", type=Path, default=None)
    parser.add_argument("--out-jsonl", type=Path, default=None)
    parser.add_argument("--max-files", type=int, default=None)
    args = parser.parse_args()
    extract_pdfs(pdf_dir=args.pdf_dir, out_jsonl=args.out_jsonl, max_files=args.max_files)
