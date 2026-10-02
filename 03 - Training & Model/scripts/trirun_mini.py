"""Mini Tri-Run: 3 head-only configs x 45 steps on mini_slice_20k. BEST-only saves."""
import sys
import time
from pathlib import Path
import torch
REPO = Path(r"C:\02_QUILLAN")
for p in [str(REPO / "scripts" / "hf_code_era"),
          str(REPO / "scripts" / "hf_code"),
          str(REPO / "scripts"),
          str(REPO / "09 - Projects" / "projects" / "oni")]:
    if p not in sys.path:
        sys.path.insert(0, p)
from quillan_v5_4_oni import QuillanOniConfig, QuillanRoninOni
BASE = REPO / "checkpoints" / "hf_restore" / "quillan_6l_vocab62_base.pt"
SLICE = REPO / "training_data" / "mini_slice_20k.pt"
OUTDIR = REPO / "checkpoints" / "hf_restore"
CONFIGS = [{"name": "lr1e5", "lr": 1e-5}, {"name": "lr3e5", "lr": 3e-5},
           {"name": "lr5e5", "lr": 5e-5}]
STEPS, SEQ, EFF = 45, 160, 16
ds = torch.load(SLICE, map_location="cpu", weights_only=True)
ids = ds["input_ids"] if isinstance(ds, dict) else ds
ntr = int(len(ids) * 0.9)
tr, va = ids[:ntr], ids[ntr:]
print(f"train={len(tr)} val={len(va)}", flush=True)
bd = torch.load(BASE, map_location="cpu", weights_only=True)
base_sd = bd.get("model_state_dict", bd.get("model", bd))
results = {}
for c in CONFIGS:
    cfg = QuillanOniConfig(vocab_size=50262, hidden_dim=1024, ffn_dim=2048,
                           n_layer=6, num_experts=34, top_k=4, max_seq_len=512)
    m = QuillanRoninOni(cfg)
    missing, unexp = m.load_state_dict(base_sd, strict=False)
    print(f"[{c['name']}] bound missing={len(missing)} unexp={len(unexp)}",
          flush=True)
    for n, p in m.named_parameters():
        p.requires_grad = ("lm_head" in n or "ln_f" in n)
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],
                            lr=c["lr"])
    m.train()
    best, best_state = 1e9, None
    t0 = time.time()
    for s in range(STEPS):
        bi = (s * 2) % (len(tr) - 2)
        x = tr[bi:bi + 2][:, :SEQ]
        opt.zero_grad()
        o = m(x)
        logits = o[0] if isinstance(o, tuple) else o
        loss = torch.nn.functional.cross_entropy(
            logits[:, :-1, :].reshape(-1, logits.size(-1)), x[:, 1:].reshape(-1))
        loss.backward()
        opt.step()
        if (s + 1) % 15 == 0:
            m.eval()
            with torch.no_grad():
                vx = va[:4][:, :SEQ]
                vo = m(vx)
                vl = vo[0] if isinstance(vo, tuple) else vo
                vv = torch.nn.functional.cross_entropy(
                    vl[:, :-1, :].reshape(-1, vl.size(-1)),
                    vx[:, 1:].reshape(-1)).item()
            print(f"[{c['name']}] step={s + 1} train={loss.item():.4f} val={vv:.4f}",
                  flush=True)
            if vv < best:
                best = vv
                best_state = {k: v.cpu().clone() for k, v in m.state_dict().items()}
            m.train()
    results[c["name"]] = best
    if best_state is not None:
        torch.save({"model_state_dict": best_state,
                    "meta": {"cfg": c["name"], "val": best}},
                   OUTDIR / f"mini_trirun_{c['name']}_best.pt")
        print(f"[{c['name']}] SAVED val={best:.4f} ({time.time() - t0:.0f}s)",
              flush=True)
win = min(results, key=results.get)
print(f"TRIRUN WINNER={win} " + str({k: round(v, 4) for k, v in results.items()}),
      flush=True)
