import os
from pathlib import Path
NAMES = {"quillan_head_v62_best.pt", "quillan_frontier_v2_best.pt",
         "quillan_6l_vocab62_mapped.pt", "quillan_clean_v62_v1.pt",
         "quillan_clean_v62_cpu20k.pt", "quillan_main_122b_init.pt",
         "quillan_6l_step_5311.pt", "quillan_master_gold_training_v1.pt"}
SKIP = ("node_modules", "__pycache__", ".venv", "open-webui", "Nextverse",
        "site-packages", ".git", "emojis", "static")
ROOTS = ["C:\\02_QUILLAN", "C:\\Quillan-Ronin", "C:\\CascadeProjects",
         "C:\\QuillanWorker", "C:\\03_WORKSPACE", "C:\\04_CONFIG",
         "C:\\Users\\Admin\\Downloads", "C:\\Users\\Admin\\Desktop",
         "C:\\Users\\Admin\\Documents"]
for root in ROOTS:
    if not Path(root).exists():
        continue
    for dp, dns, fns in os.walk(root, topdown=True):
        dns[:] = [d for d in dns if d not in SKIP and not d.startswith(".")]
        for f in fns:
            if f in NAMES:
                try:
                    mb = os.path.getsize(os.path.join(dp, f)) // 1024 // 1024
                except OSError:
                    mb = -1
                print(f"{mb}MB :: {dp}\\{f}", flush=True)
print("search done", flush=True)
