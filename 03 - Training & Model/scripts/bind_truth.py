"""Bind truth: import each code copy by FILE PATH, fresh 6L model, key count + bind."""
import importlib.util
import torch
from pathlib import Path
REPO = Path(r"C:\02_QUILLAN")
COPIES = {
    "era": REPO / "scripts" / "hf_code_era" / "quillan_v5_4_oni.py",
    "oni": REPO / "09 - Projects" / "projects" / "oni" / "quillan_v5_4_oni.py",
    "scripts": REPO / "scripts" / "quillan_v5_4_oni.py",
}
CKPT = torch.load(REPO / "checkpoints" / "hf_restore" / "mini_full_best.pt",
                  map_location="cpu", weights_only=True)
sd = CKPT.get("model_state_dict", CKPT)
print(f"checkpoint keys: {len(sd)}", flush=True)
for name, path in COPIES.items():
    try:
        spec = importlib.util.spec_from_file_location(f"qoni_{name}", str(path))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        cfg = mod.QuillanOniConfig(vocab_size=50262, hidden_dim=1024, ffn_dim=2048,
                                   n_layer=6, num_experts=34, top_k=4, max_seq_len=512)
        m = mod.QuillanRoninOni(cfg)
        fresh_keys = len(m.state_dict())
        missing, unexp = m.load_state_dict(sd, strict=False)
        print(f"{name}: fresh_keys={fresh_keys} missing={len(missing)} unexp={len(unexp)}",
              flush=True)
    except Exception as e:
        print(f"{name}: ERROR {str(e)[:200]}", flush=True)
