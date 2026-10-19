import sys
from pathlib import Path
import torch
REPO = Path(r"C:\02_QUILLAN")
for p in [str(REPO / "09 - Projects" / "projects" / "oni"),
          str(REPO / "scripts"),
          str(REPO / "scripts" / "hf_code"),
          str(REPO / "scripts" / "hf_code_era")]:
    if p not in sys.path:
        sys.path.insert(0, p)
torch.set_num_threads(3)
import quillan_v5_4_oni as Q
print("core file=" + Q.__file__, flush=True)
from quillan_v5_4_oni import QuillanOniConfig, QuillanRoninOni
CKPT = REPO / "checkpoints" / "hf_restore" / "quillan_oni_5.4.0_step660_5.22GB.pt"
cfg = QuillanOniConfig(vocab_size=50257, hidden_dim=1024, ffn_dim=2048,
                       n_layer=12, num_experts=34, top_k=4, max_seq_len=512)
m = QuillanRoninOni(cfg)
d = torch.load(CKPT, map_location="cpu", weights_only=True)
sd = d.get("model_state_dict", d.get("model", d))
missing, unexpected = m.load_state_dict(sd, strict=False)
print(f"STRICT-AS-IS missing={len(missing)} unexp={len(unexpected)} total={len(sd)}",
      flush=True)
for k in list(missing)[:6]:
    print("MISS " + str(k), flush=True)
for k in list(unexpected)[:6]:
    print("UNEXP " + str(k), flush=True)
