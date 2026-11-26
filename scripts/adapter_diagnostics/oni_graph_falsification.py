#!/usr/bin/env python3
"""Read-only falsification of the ONI body before donor transplant.
Checks owner config, geometry feasibility, and downstream graph delta.
"""

import sys
import torch

SCRIPTS = r"F:\02_QUILLAN\03 - Training & Model\scripts"
sys.path.insert(0, SCRIPTS)

from oni_v2_config import owner_config
from quillan_v5_4_oni import QuillanOniConfig, QuillanRoninOni

for name in ("mini", "main"):
    c = owner_config(name)
    print(
        name,
        "vocab",
        c.vocab_size,
        "H",
        c.hidden_dim,
        "F",
        c.ffn_dim,
        "layers",
        c.n_layer,
        "ctx",
        c.max_seq_len,
        flush=True,
    )
    assert c.vocab_size == 151648, "WRONG CONFIG: default 50262 leaked in"

ONI = {"mini": (1152, 3072), "main": (1536, 3072)}
DONOR = {
    "qwen": (896, 4864),
    "llama": (2048, 8192),
    "bitnet": (2560, 6912),
    "falcon": (2048, 8192),
}
for d, (h, f) in DONOR.items():
    for o, (H, F) in ONI.items():
        ok_h, ok_f = h == H, f == F
        print(
            f"{d}->{o}: H {h}/{H} {'OK' if ok_h else 'FAIL'} "
            f"F {f}/{F} {'OK' if ok_f else 'FAIL'}",
            flush=True,
        )

cfg = QuillanOniConfig(
    vocab_size=1000,
    hidden_dim=64,
    n_layer=2,
    n_head=4,
    head_dim=16,
    ffn_dim=128,
    num_experts=34,
    max_seq_len=128,
    device="cpu",
)
m = QuillanRoninOni(cfg).eval()
ids = torch.randint(1, 999, (1, 16))
torch.manual_seed(0)
with torch.inference_mode():
    clean = m(ids, compute_aux=False, deliberation=False)
    aux = m(ids, compute_aux=True, deliberation=False)
    delib = m(ids, compute_aux=True, deliberation=True)

print(
    "clean-aux max/mean",
    float((clean - aux).abs().max()),
    float((clean - aux).abs().mean()),
    flush=True,
)
print(
    "aux-delib max/mean",
    float((aux - delib).abs().max()),
    float((aux - delib).abs().mean()),
    flush=True,
)
print(
    "top1 clean/aux/delib",
    int(clean[0, -1].argmax()),
    int(aux[0, -1].argmax()),
    int(delib[0, -1].argmax()),
    flush=True,
)
