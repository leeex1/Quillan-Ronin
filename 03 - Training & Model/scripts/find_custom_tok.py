import json
from pathlib import Path
ROOTS = [r"C:\02_QUILLAN\training_data",
         r"C:\02_QUILLAN\03 - Training & Model",
         r"C:\02_QUILLAN\09 - Projects\projects\oni",
         r"C:\02_QUILLAN\09 - Projects\projects\05_Training",
         r"C:\02_QUILLAN\scripts",
         r"C:\02_QUILLAN\checkpoints",
         r"C:\02_QUILLAN\01 - Core Architecture"]
MARKERS = ["<|im_end|>", "<|im_start|>", "<|user|>", "<|assistant|>", "<|start|>"]
found = []
for root in ROOTS:
    for p in Path(root).rglob("tokenizer*.json"):
        if "node_modules" in str(p):
            continue
        try:
            txt = p.read_text(encoding="utf-8", errors="replace")
            hits = [m for m in MARKERS if m in txt]
            data = json.loads(txt)
            vocab = data.get("model", {}).get("vocab", {})
            added = data.get("added_tokens", data.get("added_tokens_decoder", {}))
            print(f"{p} size={p.stat().st_size} markers={hits} "
                  f"vocab_len={len(vocab) if vocab else '?'}", flush=True)
            if hits:
                found.append(str(p))
        except Exception as e:
            print(f"{p} UNREADABLE {type(e).__name__}", flush=True)
print("CUSTOM-SET-FOUND:" if found else "NO-CUSTOM-SET", flush=True)
for f in found:
    print("  " + f, flush=True)
