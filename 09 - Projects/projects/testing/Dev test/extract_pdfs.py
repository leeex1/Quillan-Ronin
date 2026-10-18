import os
import json
from pypdf import PdfReader
from pathlib import Path

pdf_dir = Path(r"C:\02_QUILLAN\01_Knowledge_Base\Wiki\Papers")
out_jsonl = Path(r"C:\02_QUILLAN\training_data\pdf_papers_corpus.jsonl")

pdf_files = list(pdf_dir.glob("*.pdf"))
print(f"[*] Found {len(pdf_files)} PDF files in {pdf_dir}")

extracted_count = 0
total_pages = 0

with open(out_jsonl, "w", encoding="utf-8") as out_f:
    for pdf_path in pdf_files:
        try:
            reader = PdfReader(str(pdf_path))
            num_pages = len(reader.pages)
            text_blocks = []
            for i, page in enumerate(reader.pages):
                page_text = page.extract_text()
                if page_text and len(page_text.strip()) > 50:
                    text_blocks.append(page_text.strip())
            
            full_pdf_text = "\n\n".join(text_blocks)
            if len(full_pdf_text) > 200:
                record = {
                    "text": f"Paper Title / File: {pdf_path.name}\n\n" + full_pdf_text,
                    "meta": {"source": "local_pdf_paper", "filename": pdf_path.name, "pages": num_pages}
                }
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                extracted_count += 1
                total_pages += len(text_blocks)
        except Exception as e:
            print(f"[!] Error processing {pdf_path.name}: {e}")

print(f"[COMPLETE] Successfully extracted {extracted_count} PDFs ({total_pages} total pages) into {out_jsonl}")
