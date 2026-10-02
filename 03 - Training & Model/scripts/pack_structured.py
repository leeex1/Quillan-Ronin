"""STRUCTURED DIALOGUE PACK: preserve roles/system/think, filter code-exhaust,
include Samurai spec (self-knowledge). Legacy BPE, 512-len, sharded. No flattening."""
import sys
import json
import re
from pathlib import Path
import torch
REPO = Path(r"C:\02_QUILLAN")
for p in [str(REPO / "scripts"), str(REPO / "03 - Training & Model"), str(REPO)]:
    if p not in sys.path:
        sys.path.insert(0, p)
from quillan_bpe_tokenizer import QuillanBPETokenizer
tok = QuillanBPETokenizer()
TD = REPO / "training_data"
SAM = TD / "hf_samurai"

DROP_PATTERNS = ["[MEDIA]", "@types/", '"requiresBuild"', "#!/usr/bin", "sha512-",
                 "checkedAt", "node_modules", "Modified: 17", "__pycache__",
                 '"integrity":', "size\":1141", "licenses", ".png\n", ".jpg\n",
                 "Traceback (most recent", "File \"", "  File ", "npm-debug",
                 "package-lock", "yarn.lock", "git clone", "pip install",
                 "Copyright (c)", "All rights reserved", "MIT License"]
CODE_START = re.compile(r"^\s*(import |from |def |class |#!/|<\?php|package |npm |git |pip |docker |kubectl )")
COMMON = set("the be to of and a in that have i it for not on with he as you do at this but his by from they we say her she or an will my one all would there their what so up out if about who get which go me when make can like time no just him know take people into year your good some could them see other than then now look only come its over think also back after use two how our work first well way even new want because any these give day most us is are was were has had been will would can could shall should may might must do does did".split())


def english_ratio(t: str) -> float:
    words = re.findall(r"[a-zA-Z']+", t.lower())
    if len(words) < 10:
        return 0.0
    return sum(1 for w in words if w in COMMON) / len(words)


def is_exhaust(t: str) -> bool:
    if any(p in t for p in DROP_PATTERNS):
        return True
    if CODE_START.match(t):
        return True
    brace = t.count("{") + t.count("}") + t.count(";") + t.count("===")
    if brace > len(t) / 40:
        return True
    return False


def format_dialogue(msgs) -> str:
    parts = []
    for m in msgs:
        if not isinstance(m, dict):
            continue
        role = str(m.get("role", "user")).capitalize()
        content = str(m.get("content", "")).strip()
        if content:
            parts.append(f"{role}: {content}")
    return "\n\n".join(parts)


rows = []
def add_text(t, source):
    t = str(t).strip()
    if len(t) < 120 or is_exhaust(t):
        return False
    if english_ratio(t) < 0.25:
        return False
    rows.append(t)
    return True


DIALOGUE_FILES = [SAM / "instruct_train.jsonl",
                  TD / "Quillan_Clean_Reasoning_Gold_Dataset.jsonl",
                  TD / "Quillan_Master_Combined_Gold.jsonl",
                  TD / "Quillan_Direct_Answers_Gold.jsonl",
                  TD / "Quillan_Explanatory_Prose_Dataset.jsonl",
                  TD / "sovereign_thinking_gold.jsonl",
                  TD / "Quillan_Refined_Thought_Corpus.jsonl",
                  TD / "Quillan_Universal_Sovereign_Gold_1000.jsonl",
                  TD / "Quillan_Canonical_Reasoning_Master_Gold.jsonl",
                  TD / "Quillan_Canonical_Pristine_Master_Gold.jsonl",
                  TD / "Quillan_General_Knowledge_Dataset.jsonl",
                  TD / "Quillan_Hyper_Tune_Gold.jsonl",
                  TD / "Quillan_Hyper_Tune_Gold_Dataset.jsonl"]
counts = {}
for path in DIALOGUE_FILES:
    if not path.exists():
        print(f"MISSING {path.name}", flush=True)
        continue
    kept = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                o = json.loads(line)
            except Exception:
                continue
            if isinstance(o, dict) and isinstance(o.get("messages"), list):
                t = format_dialogue(o["messages"])
            elif isinstance(o, dict) and "question" in o:
                t = "User: " + str(o.get("question", "")) + "\n\nAssistant: " + str(o.get("response", o.get("answer", "")))
            elif isinstance(o, dict):
                t = o.get("text", o.get("content", o.get("instruction", o.get("prompt", ""))))
                extra = o.get("response", o.get("output", ""))
                if extra:
                    t = "User: " + str(t) + "\n\nAssistant: " + str(extra)
                else:
                    t = str(t)
            else:
                t = str(o)
            if add_text(t, path.name):
                kept += 1
    counts[path.name] = kept
    print(f"{path.name}: {kept}", flush=True)

SPEC = REPO / "06 - Deployment & Platforms" / "system prompts" / "Quillan-Samurai.md"
if SPEC.exists():
    text = SPEC.read_text(encoding="utf-8")
    chunks, cur = [], []
    for para in text.split("\n\n"):
        cur.append(para)
        if sum(len(c) for c in cur) > 1800:
            chunks.append("\n\n".join(cur))
            cur = []
    if cur:
        chunks.append("\n\n".join(cur))
    sk = sum(1 for c in chunks if add_text("Quillan System Knowledge:\n\n" + c, "samurai-spec"))
    print(f"samurai-spec chunks: {sk}/{len(chunks)}", flush=True)

print(f"TOTAL structured rows: {len(rows)}", flush=True)
SEQ, CHUNK = 512, 2000
buf, total, shard = [], 0, 0
shard_paths = []
for t in rows:
    e = tok.encode(t)[:SEQ]
    if len(e) < 40:
        continue
    e = e + [1] * (SEQ - len(e))
    buf.append(e)
    if len(buf) >= CHUNK:
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
print(f"STRUCT DONE rows={total} shards={len(shard_paths)}", flush=True)
torch.save({"shards": shard_paths, "total": total, "seq": SEQ,
            "format": "structured-roles-preserved"},
           TD / "mini_structured.pt")
print("MANIFEST done", flush=True)
