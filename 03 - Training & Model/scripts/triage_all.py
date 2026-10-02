#!/usr/bin/env python3
"""Triage ALL unevaluated 6L candidates (owner order: use what exists).
Sequential 10-batch LM probe per file, appends verdicts to triage_all.txt.
Light eval only: no optimizer, ~2GB transient. CPU threads=3."""
from __future__ import annotations
import sys
import time
from pathlib import Path
import torch

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REPO = Path(r"C:\02_QUILLAN")
for p in [str(REPO / "scripts"), str(REPO / "09 - Projects" / "projects" / "oni"),
          str(REPO), str(REPO / "03 - Training & Model")]:
    if p not in sys.path:
        sys.path.insert(0, p)

torch.set_num_threads(3)

CANDIDATES = [
    ("checkpoints/checkpoints_oni/quillan_fused_pretrained_master.pt", 6),
    ("checkpoints/checkpoints_oni/quillan_generative_foundation.pt", 6),
    ("checkpoints/production_export/quillan_ronin_v531_sovereign_production.pt", 6),
    ("checkpoints/checkpoints_oni/quillan_oni_inference.pt", 6),
    ("checkpoints/checkpoints_sft/quillan_calibrated_dialogue.pt", 6),
    ("checkpoints/checkpoints_sft/quillan_teacher_tail_latest.pt", 6),
    ("checkpoints/quillan_oni_main_12l.pt", 12),
]


def main() -> int:
    from quillan_v5_4_oni import QuillanOniConfig, QuillanRoninOni
    ds = torch.load(REPO / "training_data" / "canonical_standardized"
                    / "quillan_master_gold_training_v1.pt", map_location="cpu",
                    weights_only=True)
    ids, labels = ds["input_ids"], ds["labels"]
    n = ids.shape[0]
    val = torch.arange(int(n * 0.9), n)
    out = REPO / "training_logs" / "triage_all.txt"
    for rel, nlayers in CANDIDATES:
        ckpt = REPO / rel
        if not ckpt.exists():
            line = f"{ckpt.name}: MISSING, skipped"
            print(line, flush=True)
            with open(out, "a") as f:
                f.write(line + "\n")
            continue
        try:
            cfg = QuillanOniConfig(vocab_size=50257, hidden_dim=1024, ffn_dim=2048,
                                   n_layer=nlayers, num_experts=34, top_k=4,
                                   max_seq_len=512)
            model = QuillanRoninOni(cfg)
            d = torch.load(ckpt, map_location="cpu", weights_only=True)
            sd = d.get("model_state_dict", d.get("model", d))
            missing, unexpected = model.load_state_dict(sd, strict=False)
            model.eval()
            tot, cnt = 0.0, 0
            with torch.no_grad():
                for i in range(0, min(20, len(val)), 2):
                    idx = val[i:i + 2]
                    x = ids[idx][:, :160]
                    y = labels[idx][:, :160]
                    o = model(x)
                    logits = o[0] if isinstance(o, tuple) else o
                    loss = torch.nn.functional.cross_entropy(
                        logits[:, :-1, :].reshape(-1, logits.size(-1)),
                        y[:, 1:].reshape(-1), ignore_index=-100)
                    tot += loss.item()
                    cnt += 1
            mean = tot / max(1, cnt)
            verdict = ("MATURE-restore?" if mean < 3.0
                       else "MID-needs-train" if mean < 6.0 else "COLD")
            line = (f"{ckpt.name} [{nlayers}L] mean={mean:.4f} "
                    f"missing={len(missing)} unexp={len(unexpected)} -> {verdict}")
        except Exception as e:
            line = f"{ckpt.name}: EVAL-FAIL {type(e).__name__}"
        print(line, flush=True)
        with open(out, "a") as f:
            f.write(line + "\n")
        del model
        import gc
        gc.collect()
    print("triage_all DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
