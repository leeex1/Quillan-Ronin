import os
from pathlib import Path
KEYS = ("qwen", "llama", "bitnet", "mistral", "phi-",
        ".gguf", ".safetensors", ".ptq", "awq", "gptq")
SKIP = ("node_modules", "__pycache__", ".venv", "open-webui",
        "Nextverse", "site-packages", ".git", "emojis", "static")
ROOTS = [r"C:\02_QUILLAN", r"C:\Users\Admin\.cache",
         r"C:\Users\Admin\Downloads", r"C:\03_WORKSPACE", r"C:\04_CONFIG"]
for root in ROOTS:
    rp = Path(root)
    if not rp.exists():
        continue
    for dp, dns, fns in os.walk(rp, topdown=True):
        dns[:] = [d for d in dns if d not in SKIP and not d.startswith(".")]
        for f in fns:
            fl = f.lower()
            if any(k in fl for k in KEYS):
                try:
                    sz = os.path.getsize(os.path.join(dp, f)) // 1024 // 1024
                except OSError:
                    sz = -1
                print(f"{sz}MB :: {dp}\\{f}", flush=True)
print("hunt done", flush=True)
