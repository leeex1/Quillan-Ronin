import sys
from pathlib import Path
from huggingface_hub import snapshot_download
DST = r"C:\02_QUILLAN\checkpoints\hf_restore"
Path(DST).mkdir(parents=True, exist_ok=True)
for f in ["quillan_frontier_v2_best_loss0.0789_step2500.pt",
          "quillan_oni_5.4.0_step660_5.22GB.pt"]:
    print(f"fetch {f} ...", flush=True)
    snapshot_download(repo_id="CrashOverrideX/Quillan-Ronin",
                      local_dir=DST, allow_patterns=[f])
    print(f"done {f}", flush=True)
print("ALL DONE", flush=True)
