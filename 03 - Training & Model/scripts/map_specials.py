#!/usr/bin/env python3
"""Map the 5 new specials. Simple version: same key names, copy rows 0-50256,
set rows 50257-50261. Embedding rows = mean of boundary anchors. Head rows = 0.
Saves quillan_6l_vocab62_mapped.pt. Verifies shapes + single-ID roundtrip."""
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
SRC = REPO / "checkpoints" / "checkpoints_sft" / "quillan_frontier_v2_best.pt"
OUT = REPO / "checkpoints" / "checkpoints_sft" / "quillan_6l_vocab62_mapped.pt"
OLD_V, NEW_V = 50257, 50262
NEW_IDS = {"<|start|>": 50257, "<|user|>": 50258, "<|assistant|>": 50259,
           "<|im_start|>": 50260, "<|im_end|>": 50261}


def main() -> int:
    from quillan_v5_4_oni import QuillanOniConfig, QuillanRoninOni
    from quillan_tokenizer_unified import UnifiedQuillanTokenizer
    tok = UnifiedQuillanTokenizer()

    def build(vocab):
        cfg = QuillanOniConfig(vocab_size=vocab, hidden_dim=1024, ffn_dim=2048,
                               n_layer=6, num_experts=34, top_k=4, max_seq_len=512)
        return QuillanRoninOni(cfg)

    d = torch.load(SRC, map_location="cpu", weights_only=True)
    sd_old = d.get("model_state_dict", d.get("model", d))
    m_new = build(NEW_V)
    sd_new = m_new.state_dict()

    anchors, seen = (["<|endoftext|>", ".", "\n", ":", "#", "```"], [])
    for a in anchors:
        try:
            seen += [i for i in tok.encode(a) if 0 <= i < OLD_V]
        except Exception:
            pass
    seen = sorted(set(seen)) or list(range(64))

    n_mapped = n_copied = 0
    with torch.no_grad():
        for k, v_new in sd_new.items():
            v_old = sd_old.get(k)
            if v_old is None:
                continue
            if v_old.shape == v_new.shape:
                v_new.copy_(v_old)
                n_copied += 1
            elif (len(v_new.shape) == 2 and v_new.shape[0] == NEW_V
                  and v_old.shape[0] == OLD_V and v_old.shape[1] == v_new.shape[1]):
                v_new[:OLD_V].copy_(v_old)
                if "head" not in k:  # embedding tables: anchor-mean
                    mu = v_old[torch.tensor(seen)].mean(dim=0)
                    v_new[OLD_V:].copy_(mu.unsqueeze(0).expand(NEW_V - OLD_V, -1))
                else:
                    v_new[OLD_V:].zero_()
                n_mapped += 1
                print(f"mapped {k}: {tuple(v_old.shape)} -> {tuple(v_new.shape)}",
                      flush=True)
    m_new.load_state_dict(sd_new, strict=False)
    torch.save({"model_state_dict": {k: v.cpu() for k, v in m_new.state_dict().items()},
                "mapped_from": SRC.name, "new_ids": NEW_IDS}, OUT)
    print(f"copied={n_copied} mapped={n_mapped} saved={OUT.name} "
          f"({OUT.stat().st_size / 1024 ** 2:.0f}MB)", flush=True)
    m_check = build(NEW_V)
    m_check.load_state_dict(torch.load(OUT, map_location="cpu",
                                       weights_only=True)["model_state_dict"],
                            strict=False)
    print("reload ok", flush=True)
    for s, want in NEW_IDS.items():
        ids = tok.encode(s)
        print(f"{s} -> {ids} {'SINGLE-OK' if ids == [want] else 'FAIL'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
