"""Gate calibration + deliberate-vs-raw on current sft weights (oni-copy, importlib)."""
import importlib.util
import sys
import torch
from pathlib import Path
REPO = Path(r"C:\02_QUILLAN")
for p in [str(REPO / "03 - Training & Model"), str(REPO)]:
    if p not in sys.path:
        sys.path.insert(0, p)
from quillan_bpe_tokenizer import QuillanBPETokenizer
spec = importlib.util.spec_from_file_location(
    "qoni", str(REPO / "09 - Projects" / "projects" / "oni" / "quillan_v5_4_oni.py"))
qoni = importlib.util.module_from_spec(spec)
spec.loader.exec_module(qoni)
tok = QuillanBPETokenizer()
cfg = qoni.QuillanOniConfig(vocab_size=50262, hidden_dim=1024, ffn_dim=2048,
                            n_layer=6, num_experts=34, top_k=4, max_seq_len=512)
m = qoni.QuillanRoninOni(cfg)
bd = torch.load(REPO / "checkpoints" / "hf_restore" / "mini_sft_best.pt",
                map_location="cpu", weights_only=True)
missing, unexp = m.load_state_dict(bd.get("model_state_dict", bd), strict=False)
print(f"bound missing={len(missing)} unexp={len(unexp)}", flush=True)
m.eval()
PROMPTS = ["User: Say hello in one short sentence.\n\nAssistant:",
           "User: The capital of France is\n\nAssistant:",
           "User: Why is the sky blue?\n\nAssistant:"]
with torch.no_grad():
    for pi, p in enumerate(PROMPTS):
        ids = tok.encode(p)
        pt = torch.tensor([ids[-256:]], dtype=torch.long)
        out = m(pt)
        logits = out[0] if isinstance(out, tuple) else out
        hidden = m.wte(pt)
        g = m.quality_gate(hidden)
        print(f"P{pi}: cov={g['covenant_identity']:.3f} eth={g['ethics_constraint']:.3f} "
              f"passed={g['passed']}", flush=True)
    ids = tok.encode(PROMPTS[0])
    r = m.deliberate(ids, max_rounds=3, max_tokens=30, temp=0.65, num_branches=2)
    print("LOOP rounds=", len(r["trace"]["rounds"]),
          "gate=", r["trace"]["gates"]["passed"],
          "branches=", [(b["branch"], b["passed"]) for b in r["trace"]["branches"]], flush=True)
    print("LOOP TEXT:", repr(tok.decode([t for t in r["tokens"] if t < 50257]))[:300], flush=True)
print("GATE PROBE DONE", flush=True)
