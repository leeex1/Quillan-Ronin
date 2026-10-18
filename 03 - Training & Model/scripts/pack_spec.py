"""Append Samurai spec chunks (self-knowledge) to structured pack WITHOUT prose filter."""
import sys
from pathlib import Path
import torch
REPO = Path(r"C:\02_QUILLAN")
for p in [str(REPO / "scripts"), str(REPO / "03 - Training & Model"), str(REPO)]:
    if p not in sys.path:
        sys.path.insert(0, p)
from quillan_bpe_tokenizer import QuillanBPETokenizer
tok = QuillanBPETokenizer()
TD = REPO / "training_data"
SPEC = REPO / "06 - Deployment & Platforms" / "system prompts" / "Quillan-Samurai.md"
text = SPEC.read_text(encoding="utf-8")
chunks, cur = [], []
for para in text.split("\n\n"):
    cur.append(para)
    if sum(len(c) for c in cur) > 1800:
        chunks.append("\n\n".join(cur))
        cur = []
if cur:
    chunks.append("\n\n".join(cur))
print(f"spec paragraphs -> {len(chunks)} chunks", flush=True)
SEQ = 512
buf, total, shard = [], 0, 56
shard_paths = []
for c in chunks:
    t = "Quillan System Knowledge:\n\n" + c.strip()
    if len(t) < 120:
        continue
    e = tok.encode(t)[:SEQ]
    if len(e) < 40:
        continue
    e = e + [1] * (SEQ - len(e))
    buf.append(e)
    if len(buf) >= 2000:
        p = TD / f"_struct_shard_{shard}.pt"
        torch.save(torch.tensor(buf, dtype=torch.long), p)
        shard_paths.append(str(p))
        total += len(buf)
        print(f"shard {shard}: {len(buf)} (total {total})", flush=True)
        shard += 1
        buf = []
if buf:
    p = TD / f"_struct_shard_{shard}.pt"
    torch.save(torch.tensor(buf, dtype=torch.long), p)
    shard_paths.append(str(p))
    total += len(buf)
print(f"SPEC DONE rows={total}", flush=True)
man_path = TD / "mini_structured.pt"
man = torch.load(man_path, map_location="cpu", weights_only=True)
man["shards"].extend(shard_paths)
man["total"] += total
man["spec_shards"] = shard_paths
torch.save(man, man_path)
print(f"MANIFEST updated: total={man['total']} shards={len(man['shards'])}", flush=True)
