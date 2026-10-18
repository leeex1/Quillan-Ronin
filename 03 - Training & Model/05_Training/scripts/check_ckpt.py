import torch
from pathlib import Path
for d in [r"C:\02_QUILLAN\checkpoints\checkpoints_oni"]:
    for p in Path(d).glob("*.pt"):
        ck = torch.load(str(p), map_location="cpu", weights_only=False)
        print(p.name, "step", ck.get("step"), "best_val", ck.get("best_val"))
        print("  keys:", list(ck.keys())[:5])
