import sys
from pathlib import Path
import torch
import torch.nn.functional as F

REPO_ROOT = Path(r"C:\02_QUILLAN")
MODEL_DIR = REPO_ROOT / "03 - Training & Model"
SCRIPTS_DIR = MODEL_DIR / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from quillan_v5_4_oni import QuillanRoninOni, QuillanConfig

def verify():
    cfg = QuillanConfig(n_layer=6, hidden_dim=1024, ffn_dim=2048, vocab_size=50262)
    model = QuillanRoninOni(cfg)
    sd = model.state_dict()
    print("Model initialized. Total keys:", len(sd))
    print("Has wte.weight:", "wte.weight" in sd, sd["wte.weight"].shape)
    print("Has h.0.attn.c_attn.weight:", "h.0.attn.c_attn.weight" in sd, sd["h.0.attn.c_attn.weight"].shape)
    print("Has h.0.moe.c_fc.weight:", "h.0.moe.c_fc.weight" in sd, sd["h.0.moe.c_fc.weight"].shape)
    print("Has h.0.moe.c_proj.weight:", "h.0.moe.c_proj.weight" in sd, sd["h.0.moe.c_proj.weight"].shape)
    print("Has h.0.moe.evo_moe.experts.0.0.weight:", "h.0.moe.evo_moe.experts.0.0.weight" in sd, sd["h.0.moe.evo_moe.experts.0.0.weight"].shape)

if __name__ == "__main__":
    verify()
