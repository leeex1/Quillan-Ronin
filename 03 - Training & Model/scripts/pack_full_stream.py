import sys
import json
from pathlib import Path
import torch
REPO = Path(r"C:\02_QUILLAN")
for p in [str(REPO / "scripts"), str(REPO / "03 - Training & Model"),
          str(REPO / "09 - Projects" / "projects" / "oni")]:
    if p not in sys.path:
        sys.path.insert(0, p)
from quillan_bpe_tokenizer import QuillanBPETokenizer
tok = QuillanBPETokenizer()
TD = REPO / "training_data"
SAM = TD / "hf_samurai"
FILES = [TD / "quillan_corpus_CLEAN_V7.jsonl",
         SAM / "instruct_train.jsonl",
         SAM / "code_train.jsonl",
         TD / "Quillan_Clean_Reasoning_Gold_Dataset.jsonl",
         TD / "Quillan_Master_Combined_Gold.jsonl",
         TD / "Quillan_General_Knowledge_Dataset.jsonl",
         TD / "Quillan_Direct_Answers_Gold.jsonl",
         TD / "Quillan_Explanatory_Prose_Dataset.jsonl",
         TD / "Quillan_Hyper_Tune_Gold_Dataset.jsonl",
         TD / "sovereign_thinking_gold.jsonl",
         TD / "Quillan_Refined_Thought_Corpus.jsonl",
         TD / "Quillan_Universal_Sovereign_Gold_1000.jsonl",
         TD / "Quillan_Canonical_Reasoning_Master_Gold.jsonl",
         TD / "Quillan_70B_Teacher_Distilled_Gold.jsonl",
         TD / "Quillan_Canonical_Pristine_Master_Gold.jsonl",
         TD / "quillan_science_absolute.jsonl",
         TD / "quillan_science_additional.jsonl",
         TD / "Quillan_Universal_100_Percent_Master_Gold.jsonl",
         TD / "Quillan_Ronin_v5.3.1_Samurai_Training_Seed_Dataset.jsonl",
         TD / "harvested_conversations_gold.jsonl",
         TD / "pdf_papers_corpus.jsonl"]
SEQ, CHUNK = 512, 4000
out_path = TD / "mini_full_stream.pt"
buf, total, shard = [], 0, 0
shard_paths = []


def flush():
    global buf, total, shard
    if not buf:
        return
    t = torch.tensor(buf, dtype=torch.long)
    p = TD / f"_stream_shard_{shard}.pt"
    torch.save(t, p)
    shard_paths.append(str(p))
    total += len(buf)
    print(f"shard {shard}: {len(buf)} rows (total {total})", flush=True)
    shard += 1
    buf = []


for path in FILES:
    if not path.exists():
        print(f"MISSING {path.name}", flush=True)
        continue
    n = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
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
            if len(txt) < 80:
                continue
            e = tok.encode(txt)[:SEQ]
            if len(e) < 32:
                continue
            e = e + [1] * (SEQ - len(e))
            buf.append(e)
            n += 1
            if len(buf) >= CHUNK:
                flush()
    print(f"{path.name}: {n}", flush=True)
flush()
print(f"STREAM DONE rows={total} shards={len(shard_paths)}", flush=True)
torch.save({"shards": shard_paths, "total": total, "seq": SEQ}, out_path)
print("MANIFEST " + str(out_path), flush=True)
