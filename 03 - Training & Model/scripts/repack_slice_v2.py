import sys
import json
from pathlib import Path
import torch
REPO = Path(r"C:\02_QUILLAN")
for p in [str(REPO / "scripts"), str(REPO / "03 - Training & Model"),
          str(REPO / "09 - Projects" / "projects" / "oni")]:
    if p not in sys.path:
        sys.path.insert(0, p)
torch.set_num_threads(3)
from quillan_bpe_tokenizer import QuillanBPETokenizer
tok = QuillanBPETokenizer()
TD = REPO / "training_data"
mix = [(TD / "quillan_corpus_CLEAN_V7.jsonl", 12000),
       (TD / "Quillan_Refined_Thought_Corpus.jsonl", 4000),
       (TD / "Quillan_Clean_Reasoning_Gold_Dataset.jsonl", 4000)]
rows = []
for path, want in mix:
    got = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            if got >= want:
                break
            line = line.strip()
            if not line:
                continue
            try:
                o = json.loads(line)
            except Exception:
                continue
            txt = o.get("text", o.get("content", o.get("prompt", ""))) if isinstance(o, dict) else str(o)
            if isinstance(txt, list):
                txt = " ".join(str(x) for x in txt)
            txt = str(txt).strip()
            if len(txt) < 40:
                continue
            rows.append(txt)
            got += 1
    print(f"{path.name}: {got}", flush=True)
print(f"rows={len(rows)}", flush=True)
SEQ = 160
ids = []
for t in rows:
    e = tok.encode(t)[:SEQ]
    if len(e) < 16:
        continue
    e = e + [1] * (SEQ - len(e))
    ids.append(e)
print(f"kept={len(ids)}", flush=True)
out = torch.tensor(ids, dtype=torch.long)
torch.save({"input_ids": out, "meta": {"n": len(ids), "seq": SEQ, "vocab": "legacy-50257"}},
           TD / "mini_slice_20k.pt")
print("SLICE DONE " + str(out.shape), flush=True)
