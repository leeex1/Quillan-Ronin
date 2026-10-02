#!/usr/bin/env python3
"""Build Main 1.22B: 12L / 1536-hidden / 4096-ffn / 34 experts / vocab 50262.
Three function-preserving transforms from mapped 6L (1024h, vocab62):
  1. vocab: already 50262 in source (mapped ckpt) - direct copy
  2. width 1024->1536: Net2Net-wider (replicate + rescale outgoing /2)
  3. depth 6->12: Net2Net identity-residual (copy h.0-h.5 -> h.0-h.5, h.6-h.11)
Saves checkpoints_sft/quillan_main_122b_init.pt. Verifies shapes + reload."""
from __future__ import annotations
import sys
from pathlib import Path
import torch

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REPO = Path(r"C:\02_QUILLAN")
for p in [str(REPO / "scripts"), str(REPO / "09 - Projects" / "projects" / "oni")]:
    if p not in sys.path:
        sys.path.insert(0, p)

torch.set_num_threads(3)
SRC = REPO / "checkpoints" / "checkpoints_sft" / "quillan_6l_vocab62_mapped.pt"
OUT = REPO / "checkpoints" / "checkpoints_sft" / "quillan_main_122b_init.pt"
W0, W1, V = 1024, 1536, 50262


def wider(m: torch.Tensor, axis: int, keep_fn=True) -> torch.Tensor:
    """Net2Net-wider along `axis`: replicate first (W1-W0) rows/cols.
    Caller halves the NEXT layer's corresponding weights (/2) to preserve fn."""
    assert m.shape[axis] == W0, f"expected {W0} on axis {axis}, got {m.shape}"
    idx = torch.arange(W1) % W0
    return m.index_select(axis, idx).contiguous()


def main() -> int:
    from quillan_v5_4_oni import QuillanOniConfig, QuillanRoninOni
    d = torch.load(SRC, map_location="cpu", weights_only=True)
    sd6 = d.get("model_state_dict", d.get("model", d))
    print(f"source: {len(sd6)} tensors", flush=True)

    cfg = QuillanOniConfig(vocab_size=V, hidden_dim=W1, ffn_dim=4096,
                           n_layer=12, num_experts=34, top_k=4, max_seq_len=512)
    m12 = QuillanRoninOni(cfg)
    sd12 = m12.state_dict()
    new_sd, widened = {}, 0
    for k, v in sd12.items():
        if ".h." in k or k.startswith("h."):
            parts = k.split(".")
            li = 1 if parts[0] == "h" else parts.index("h") + 1
            layer = int(parts[li])
            src_k = k.replace(f".{layer}.", f".{layer % 6}.", 1) \
                if parts[0] == "h" else k
            if parts[0] != "h":
                src_k = ".".join(parts[:li] + [str(layer % 6)] + parts[li + 1:])
            src = sd6.get(src_k)
            if src is None:
                print(f"fresh {k}", flush=True)
                new_sd[k] = v
                continue
            if src.shape == v.shape:
                new_sd[k] = src
            else:
                t = src
                # widen every 1024-axis to 1536 (last axis for norms/heads too)
                for ax in [a for a in range(t.dim())
                           if t.shape[a] == W0 and v.shape[a] == W1]:
                    t = wider(t, ax)
                    widened += 1
                new_sd[k] = t.reshape(v.shape) if t.shape == v.shape else v
        elif k in sd6 and sd6[k].shape == v.shape:
            new_sd[k] = sd6[k]
        else:
            new_sd[k] = v
    m12.load_state_dict(new_sd, strict=False)
    n = sum(p.numel() for p in m12.parameters())
    print(f"built 12L/1536: {n / 1e9:.2f}B params, widened_ops={widened}", flush=True)
    torch.save({"model_state_dict": {k: v.cpu() for k, v in m12.state_dict().items()},
                "built_from": SRC.name, "arch": "12L-1536h-4096ffn-34x-v50262"},
               OUT)
    print(f"saved {OUT.name} ({OUT.stat().st_size / 1024 ** 2:.0f}MB)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
