import os
from pathlib import Path
CORPUS = ("quillan_corpus", "holy_grail", "robust", "secure", "merged",
          "pristine", "frontier_intact")
SKIP = ("node_modules", "__pycache__", ".venv", "open-webui", "Nextverse",
        "site-packages", ".git", "emojis")
ROOTS = [r"C:\02_QUILLAN", r"C:\Quillan-Ronin", r"C:\CascadeProjects",
         r"C:\QuillanWorker", r"C:\Users\Admin\Downloads",
         r"C:\03_WORKSPACE", r"C:\Users\Admin\Desktop", r"C:\Users\Admin\Documents"]


def sz(p):
    try:
        return os.path.getsize(p)
    except OSError:
        return 0


print("=== corpus files ===", flush=True)
for root in ROOTS:
    if not Path(root).exists():
        continue
    for dp, dns, fns in os.walk(root, topdown=True):
        dns[:] = [d for d in dns if d not in SKIP and not d.startswith(".")]
        for f in fns:
            fl = f.lower()
            if fl.endswith((".jsonl", ".txt")) and any(k in fl for k in CORPUS):
                print(f"{sz(os.path.join(dp, f)) // 1024 // 1024}MB :: {dp}\\{f}",
                      flush=True)

print("=== top dirs (depth<=2, MB) ===", flush=True)
sizes = []
for root in [r"C:\02_QUILLAN", r"C:\Users\Admin\Downloads"]:
    for dp, dns, fns in os.walk(root, topdown=True):
        depth = dp.count(os.sep) - root.count(os.sep)
        dns[:] = [d for d in dns if d not in SKIP and not d.startswith(".")]
        if depth > 2:
            dns[:] = []
            continue
        if depth == 2:
            t = 0
            for dp2, _, fns2 in os.walk(dp, topdown=True):
                for f in fns2:
                    t += sz(os.path.join(dp2, f))
            sizes.append((t // 1024 // 1024, dp))
for s, p in sorted(sizes, reverse=True)[: 20]:
    print(f"{s}MB :: {p}", flush=True)

print("=== duplicate .pt (same name+size) ===", flush=True)
seen = {}
for root in [r"C:\02_QUILLAN", r"C:\Quillan-Ronin"]:
    if not Path(root).exists():
        continue
    for dp, dns, fns in os.walk(root, topdown=True):
        dns[:] = [d for d in dns if d not in SKIP and not d.startswith(".")]
        for f in fns:
            if f.lower().endswith(".pt"):
                key = (f.lower(), sz(os.path.join(dp, f)) // 1024 // 1024)
                seen.setdefault(key, []).append(dp)
for (name, mb), locs in seen.items():
    if len(locs) > 1:
        print(f"DUP {mb}MB {name} x{len(locs)} :: {locs[0]} (+{len(locs)-1} more)",
              flush=True)
print("audit done", flush=True)
