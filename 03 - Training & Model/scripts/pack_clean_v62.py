#!/usr/bin/env python3
"""Pack CLEAN_V7 with the FIXED 50262 tokenizer (owner's canonical set).
Rules (from tonight's audits): single-ID specials, pad=1 (never EOS),
NO tail amputation (sliding window over full text), keep <|im_end|> intact.
Output: training_data/canonical_standardized/quillan_clean_v62_v1.pt
Format: continued-pretraining (labels=input_ids), seq 256."""
from __future__ import annotations
import json
import sys
from pathlib import Path
import torch

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REPO = Path(r"C:\02_QUILLAN")
sys.path.insert(0, str(REPO / "09 - Projects" / "projects" / "oni"))
SRC = REPO / "training_data" / "hf_samurai" / "quillan_corpus_CLEAN_V7.jsonl"
OUT = REPO / "training_data" / "canonical_standardized" / "quillan_clean_v62_v1.pt"
SEQ, STRIDE = 256, 128


def main() -> int:
    from quillan_tokenizer_unified import UnifiedQuillanTokenizer
    tok = UnifiedQuillanTokenizer()
    assert tok.encode("<|im_end|>") == [50261], "tokenizer not the fixed one"
    ids: list[int] = []
    n_rows = 0
    with open(SRC, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                text = json.loads(line).get("text", "")
            except json.JSONDecodeError:
                continue
            if not text or len(text) < 32:
                continue
            # wrap once per row; <|im_end|> single-ID stop preserved by tokenizer
            ids += tok.encode(text) + [50261]
            n_rows += 1
    print(f"rows={n_rows} tokens={len(ids)}", flush=True)
    seqs = [ids[i:i + SEQ] for i in range(0, len(ids) - SEQ + 1, STRIDE)]
    # pad last to full length with pad id 1
    t = torch.full((len(seqs), SEQ), 1, dtype=torch.long)
    for i, s in enumerate(seqs):
        t[i, :len(s)] = torch.tensor(s[:SEQ], dtype=torch.long)
    torch.save({"input_ids": t, "labels": t.clone()}, OUT)
    print(f"saved {OUT.name}: {t.shape} ({OUT.stat().st_size/1024**2:.0f}MB)",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
