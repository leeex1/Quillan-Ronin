import shutil
from pathlib import Path

src = Path(r"C:\02_QUILLAN\checkpoints\checkpoints_sft\quillan_hyper_tuned_v531.pt")
dst = Path(r"C:\02_QUILLAN\checkpoints\checkpoints_sft\quillan_thinking_reasoning_master.pt")

if src.exists():
    shutil.copy2(src, dst)
    print(f"[SYNC] Successfully synchronized loss=0.3054 checkpoint to: {dst}")
else:
    print(f"[ERROR] Source checkpoint not found: {src}")
