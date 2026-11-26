#!/usr/bin/env python3
"""Extract text from the key formal papers for donor integration."""

from pathlib import Path
import fitz  # PyMuPDF

BASE = Path(r"C:\02_QUILLAN\10 - Formal Papers\Formal Papers")
WSP = Path(
    r"C:\02_QUILLAN\02 - Knowledge Foundation\Quillan Knowledge files\Weight_Space_Programming.pdf"
)
PAPERS = {
    "WSP": WSP,
    "MoDSE": BASE / "MoDSE_2409.12210.pdf",
    "MoHGE": BASE / "MoHGE_2604.23108.pdf",
    "ST-MoE": BASE / "ST-MoE_Stable_Transferable_Sparse_Experts.pdf",
    "MixtureDiverseSize": BASE / "Mixture_of_Diverse_Size_Experts.pdf"
    if (BASE / "Mixture_of_Diverse_Size_Experts.pdf").exists()
    else BASE / "MoDSE_2409.12210.pdf",
    "DeepSeekMoE": BASE / "DeepSeekMoE_2401.06066.pdf",
}
OUT = Path(r"C:\Users\Admin\AppData\Local\Temp\opencode\papers")
OUT.mkdir(exist_ok=True)
KEYWORDS = (
    "merge",
    "fuse",
    "fusion",
    "heterogeneous",
    "diverse",
    "different size",
    "transfer",
    "zero",
    "training-free",
    "train-free",
    "without training",
    "initializ",
    "alignment",
    "mapping",
    "project",
    "router",
    "expert size",
    "distill",
    "knowledge",
    "cross-tokenizer",
    "vocab",
)

for name, path in PAPERS.items():
    if not path.exists():
        print(f"MISSING {name}: {path}")
        continue
    doc = fitz.open(str(path))
    text = "\n".join(
        f"\n=== PAGE {i + 1} ===\n" + p.get_text() for i, p in enumerate(doc)
    )
    (OUT / f"{name}.txt").write_text(text, encoding="utf-8")
    # print keyword context snippets
    lines = text.split("\n")
    hits = []
    for i, ln in enumerate(lines):
        low = ln.lower()
        if any(k in low for k in KEYWORDS):
            hits.append((i, ln.strip()))
    print(f"\n########## {name} ({len(doc)} pages, {len(text)} chars) ##########")
    for i, ln in hits[:40]:
        print(f"  [{i}] {ln[:200]}")
    print(f"  ... {len(hits)} keyword lines total; full text: {OUT / f'{name}.txt'}")
