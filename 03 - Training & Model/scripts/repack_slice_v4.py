import sys
import json
from pathlib import Path
import torch
REPO = Path(r"C:\02_QUILLAN")
for p in [str(REPO / "scripts"), str(REPO / "09 - Projects" / "projects" / "oni")]:
    if p not in sys.path:
        sys.path.insert(0, p)
from quillan_tokenizer_unified import UnifiedQuillanTokenizer
tok = UnifiedQuillanTokenizer()
TD = REPO / "training_data"
SAM = TD / "hf_samurai"
mix = [(TD / "quillan_corpus_CLEAN_V7.jsonl", 8000),
       (SAM / "instruct_train.jsonl", 4000),
       (SAM / "code_train.jsonl", 4000),
       (TD / "Quillan_Refined_Thought_Corpus.jsonl", 2000),
       (TD / "sovereign_thinking_gold.jsonl", 2000)]
rows = []
for path, want in mix:
    if not path.exists():
        print(f"MISSING {path.name}", flush=True)
        continue
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
            if isinstance(o, dict):
                if isinstance(o.get("messages"), list):
                    txt = " ".join(str(m.get("content", "")) for m in o["messages"] if isinstance(m, dict))
                elif "question" in o:
                    txt = str(o.get("question", "")) + " " + str(o.get("response", o.get("answer", "")))
                else:
                    txt = o.get("text", o.get("content", o.get("instruction", o.get("prompt", ""))))
                    extra = o.get("response", o.get("output", ""))
                    if extra:
                        txt = str(txt) + " " + str(extra)
            else:
                txt = str(o)
            txt = str(txt).strip()
            if len(txt) < 40:
                continue
            rows.append(txt)
            got += 1
    print(f"{path.name}: {got}", flush=True)
SEQ = 160
ids = []
for t in rows:
    e = tok.encode(t)[:SEQ]
    if len(e) < 16:
        continue
    e = e + [2] * (SEQ - len(e))
    ids.append(e)
print(f"kept={len(ids)}", flush=True)
out = torch.tensor(ids, dtype=torch.long)
print(f"max_id={out.max().item()}", flush=True)
torch.save({"input_ids": out, "meta": {"n": len(ids), "seq": SEQ, "vocab": "unified-50262-v4"}},
           TD / "mini_slice_unified_v4.pt")
print("V4 DONE " + str(out.shape), flush=True)
