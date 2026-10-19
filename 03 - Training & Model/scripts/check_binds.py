import sys
import torch
from pathlib import Path
REPO = Path(r"C:\02_QUILLAN")
for p in [str(REPO / "scripts" / "hf_code_era"), str(REPO / "scripts")]:
    if p not in sys.path:
        sys.path.insert(0, p)
from quillan_v5_4_oni import QuillanOniConfig, QuillanRoninOni
cfg = QuillanOniConfig(vocab_size=50262, hidden_dim=1024, ffn_dim=2048,
                       n_layer=6, num_experts=34, top_k=4, max_seq_len=512)
for name in ["mini_sft_best.pt", "mini_stream2_best.pt", "mini_mega_best.pt", "mini_full_best.pt"]:
    p = REPO / "checkpoints" / "hf_restore" / name
    if not p.exists():
        print(f"{name}: MISSING", flush=True)
        continue
    try:
        bd = torch.load(p, map_location="cpu", weights_only=True)
        sd = bd.get("model_state_dict", bd)
        m = QuillanRoninOni(cfg)
        missing, unexp = m.load_state_dict(sd, strict=False)
        print(f"{name}: keys={len(sd)} missing={len(missing)} unexp={len(unexp)} "
              f"val={bd.get('meta', {}).get('val', '?')}", flush=True)
    except Exception as e:
        print(f"{name}: CORRUPT ({str(e)[:150]})", flush=True)
