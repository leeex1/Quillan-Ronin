import os
from pathlib import Path
SKIP = ("node_modules", "__pycache__", ".venv", "open-webui", "Nextverse",
        "site-packages", ".git", "emojis", "static", "assets")
ROOTS = [r"C:\Users\Admin", r"C:\02_QUILLAN", r"C:\03_WORKSPACE",
         r"C:\04_CONFIG", r"C:\ProgramData", r"C:\OllamaModels"]


def sz(p):
    try:
        return os.path.getsize(p)
    except OSError:
        return 0


for root in ROOTS:
    if not Path(root).exists():
        continue
    for dp, dns, fns in os.walk(root, topdown=True):
        dns[:] = [d for d in dns if d not in SKIP and not d.startswith(".")]
        base = os.path.basename(dp).lower()
        if base == "blobs":
            t = sum(sz(os.path.join(dp, f)) for f in fns) // 1024 // 1024
            print(f"BLOBS {t}MB :: {dp} ({len(fns)} files)", flush=True)
        for f in fns:
            fl = f.lower()
            if fl.endswith(".gguf"):
                print(f"GGUF {sz(os.path.join(dp, f)) // 1024 // 1024}MB :: "
                      f"{dp}\\{f}", flush=True)
print("hunt done", flush=True)
