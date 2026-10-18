import json
import sys
import time
from pathlib import Path
import torch

REPO = Path(r"C:\02_QUILLAN")
for p in [str(REPO / "scripts"), str(REPO / "03 - Training & Model"), str(REPO)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from quillan_bpe_tokenizer import QuillanBPETokenizer

TD = REPO / "training_data"
corpus_file = TD / "pdf_papers_corpus.jsonl"
tok = QuillanBPETokenizer()

print("[pack_papers] Loading 139 academic papers from corpus...", flush=True)
paper_rows = []
with open(corpus_file, "r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        p = json.loads(line)
        title = p.get("title", "Academic Research Paper")[:120]
        text = p.get("text", "")
        paras = [para.strip() for para in text.split("\n\n") if len(para.strip()) > 80]
        cur = []
        for para in paras:
            cur.append(para)
            if sum(len(c) for c in cur) > 1200:
                body = "\n\n".join(cur)
                paper_rows.append(f"User: Summarize the findings and architecture in the research paper \"{title}\".\n\nAssistant: {body}")
                cur = []
        if cur:
            body = "\n\n".join(cur)
            paper_rows.append(f"User: What are the principles and methods in \"{title}\"?\n\nAssistant: {body}")

print(f"[pack_papers] Extracted {len(paper_rows):,} paper Q&A training texts.", flush=True)

SEQ, CHUNK = 512, 2000
buf = []
new_shards = []
t0 = time.time()

for idx, t in enumerate(paper_rows):
    e = tok.encode(t)[:SEQ]
    if len(e) < 40:
        continue
    e = e + [1] * (SEQ - len(e))
    buf.append(e)
    if len(buf) >= CHUNK:
        shard_id = len(new_shards)
        p = TD / f"_struct_shard_paper_{shard_id}.pt"
        torch.save(torch.tensor(buf, dtype=torch.long), p)
        new_shards.append(str(p))
        print(f"[pack_papers] Saved paper shard {shard_id}: {len(buf)} rows -> {p.name}", flush=True)
        buf = []

if buf:
    shard_id = len(new_shards)
    p = TD / f"_struct_shard_paper_{shard_id}.pt"
    torch.save(torch.tensor(buf, dtype=torch.long), p)
    new_shards.append(str(p))
    print(f"[pack_papers] Saved final paper shard {shard_id}: {len(buf)} rows -> {p.name}", flush=True)

# Update manifest
manifest_path = TD / "mini_structured.pt"
man = torch.load(manifest_path, map_location="cpu", weights_only=True)
existing_shards = [s for s in man["shards"] if "paper" not in Path(s).name]
all_shards = existing_shards + new_shards

# Compute exact row total
total_rows = 0
for s in all_shards:
    total_rows += torch.load(s, map_location="cpu", weights_only=True).shape[0]

torch.save({
    "shards": all_shards,
    "total": total_rows,
    "seq": SEQ,
    "format": "structured-dialogue-plus-academic-papers-corpus"
}, manifest_path)

print(f"[pack_papers] Updated manifest {manifest_path.name}: {len(all_shards)} shards, {total_rows:,} total rows in {time.time()-t0:.1f}s.", flush=True)
