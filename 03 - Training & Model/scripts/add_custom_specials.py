import json
import shutil
from pathlib import Path
ONI = Path(r"C:\02_QUILLAN\09 - Projects\projects\oni")
NEW_SPECIAL = ["<|start|>", "<|user|>", "<|assistant|>", "<|im_start|>", "<|im_end|>"]
NEW_IDS = range(50257, 50262)

tj = ONI / "tokenizer.json"
shutil.copy2(tj, ONI / "tokenizer.json.bak")
data = json.loads(tj.read_text(encoding="utf-8"))
model = data.get("model", {})
added = data.get("added_tokens", [])
if isinstance(added, dict):
    added = [{"id": int(k), "content": v} if not isinstance(v, dict) else v
             for k, v in added.items()]
have = {t.get("content", t) if isinstance(t, dict) else t for t in added}
for tok, i in zip(NEW_SPECIAL, NEW_IDS):
    if tok not in have:
        added.append({"id": i, "content": tok, "single_word": False,
                      "lstrip": False, "rstrip": False, "normalized": False,
                      "special": True})
data["added_tokens"] = added
tj.write_text(json.dumps(data), encoding="utf-8")
print(f"tokenizer.json: +{len(NEW_SPECIAL)} specials, ids 50257-50261", flush=True)

for name, key, val in [("config.json", "vocab_size", 50262),
                       ("tokenizer_config.json", "vocab_size", 50262)]:
    p = ONI / name
    if p.exists():
        d = json.loads(p.read_text(encoding="utf-8"))
        d[key] = val
        p.write_text(json.dumps(d, indent=2), encoding="utf-8")
        print(f"{name}: {key}={val}", flush=True)

stm = ONI / "special_tokens_map.json"
if stm.exists():
    d = json.loads(stm.read_text(encoding="utf-8"))
    d.update({"im_start_token": "<|im_start|>", "im_end_token": "<|im_end|>",
              "start_token": "<|start|>", "user_token": "<|user|>",
              "assistant_token": "<|assistant|>"})
    stm.write_text(json.dumps(d, indent=2), encoding="utf-8")
    print("special_tokens_map.json: +5 entries", flush=True)

# verify
d2 = json.loads(tj.read_text(encoding="utf-8"))
names = [t.get("content") for t in d2["added_tokens"] if isinstance(t, dict)]
missing = [t for t in NEW_SPECIAL if t not in names]
print("VERIFY missing=" + str(missing), flush=True)
