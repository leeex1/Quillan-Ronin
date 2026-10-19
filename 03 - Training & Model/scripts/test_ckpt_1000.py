from pathlib import Path
import sys

ONI_DIR = Path(__file__).resolve().parents[2] / "09 - Projects" / "projects" / "oni"
if not (ONI_DIR / "quillan_v5_4_oni.py").is_file():
    raise FileNotFoundError(f"Canonical ONI model not found: {ONI_DIR}")
sys.path.insert(0, str(ONI_DIR))
import torch
from quillan_v5_4_oni import QuillanOniConfig, QuillanRoninOni
from quillan_tokenizer_unified import UnifiedQuillanTokenizer

tok = UnifiedQuillanTokenizer()
cfg = QuillanOniConfig(n_layer=6, max_seq_len=512)
model = QuillanRoninOni(cfg)
ckpt = torch.load(r"C:\02_QUILLAN\checkpoints\checkpoints_oni\quillan_oni_latest.pt", map_location="cpu", weights_only=False)
model.load_state_dict(ckpt["model"])
print("Checkpoint step", ckpt["step"], "best_val", round(ckpt["best_val"],3))
model.eval()

tests = [
    "User: Hello, who are you?\n\nAssistant:",
    "User: Explain quantum entanglement simply.\n\nAssistant:",
    "User: Write a short poem about the sea.\n\nAssistant:",
]
for prompt in tests:
    ids = tok.encode(prompt, domain="dialogue")
    out = model.generate(ids, max_tokens=80, temp=0.8)
    text = tok.decode(out)
    d = model.deliberate(ids, max_rounds=1, max_tokens=40)
    txt2 = tok.decode(d["tokens"])
    print("--- PROMPT:", prompt[:40].replace(chr(10), " "))
    print("generate:", repr(text[-120:]))
    print("deliberate gates:", d["trace"]["gates"]["passed"], "rounds:", len(d["trace"]["rounds"]))
    print()
