#!/usr/bin/env python3
"""Triage eval: is quillan_6l_step_5311.pt the mature 0.9-loss weights?
Loads it + 20 val batches, reports mean LM loss. Decides restore vs long road.
Writes result to triage_verdict.txt (relay/outbox picks it up if dropped there)."""
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


def main() -> int:
    from quillan_v5_4_oni import QuillanOniConfig, QuillanRoninOni
    ckpt = REPO / "checkpoints" / "checkpoints_oni" / "quillan_6l_step_5311.pt"
    print(f"loading {ckpt.name} ...", flush=True)
    cfg = QuillanOniConfig(vocab_size=50257, hidden_dim=1024, ffn_dim=2048,
                           n_layer=6, num_experts=34, top_k=4, max_seq_len=512)
    model = QuillanRoninOni(cfg)
    d = torch.load(ckpt, map_location="cpu", weights_only=True)
    sd = d.get("model_state_dict", d.get("model", d))
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"bound missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    model.eval()
    ds = torch.load(REPO / "training_data" / "canonical_standardized"
                    / "quillan_master_gold_training_v1.pt", map_location="cpu",
                    weights_only=True)
    ids, labels = ds["input_ids"], ds["labels"]
    n = ids.shape[0]
    val = torch.arange(int(n * 0.9), n)  # last 10% = held-out split
    tot, cnt, t0 = 0.0, 0, time.time()
    with torch.no_grad():
        for i in range(0, min(20, len(val)), 2):
            idx = val[i:i + 2]
            x = ids[idx][:, :160]
            y = labels[idx][:, :160]
            out = model(x)
            logits = out[0] if isinstance(out, tuple) else out
            loss = torch.nn.functional.cross_entropy(
                logits[:, :-1, :].reshape(-1, logits.size(-1)),
                y[:, 1:].reshape(-1), ignore_index=-100)
            tot += loss.item()
            cnt += 1
            print(f"batch {cnt}/10 lm_loss={loss.item():.4f}", flush=True)
    mean = tot / max(1, cnt)
    verdict = ("MATURE - restore candidate" if mean < 3.0
               else "MID - needs training" if mean < 6.0 else "COLD - long road")
    line = f"step_5311 mean LM loss={mean:.4f} over {cnt} batches ({time.time()-t0:.0f}s) -> {verdict}"
    print(line, flush=True)
    (REPO / "training_logs" / "triage_verdict.txt").write_text(line + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
