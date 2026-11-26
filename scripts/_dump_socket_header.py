#!/usr/bin/env python3
"""Read-only dump of socket-v2.safetensors header: exact keys, shapes, dtypes."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

from safetensors import safe_open

PATH = Path(
    r"C:\02_QUILLAN\exports\quillan-mini-sovereign-bitnet158-socket-v2.safetensors"
)

with safe_open(str(PATH), framework="pt", device="cpu") as f:
    keys = list(f.keys())
    meta: dict[str, tuple[tuple[int, ...], str]] = {}
    for k in keys:
        sl = f.get_slice(k)
        meta[k] = (tuple(sl.get_shape()), sl.get_dtype())

print("TOTAL_KEYS", len(keys))
print("WITH_SCALE", sum(1 for k in keys if k.endswith(".bitnet_scale")))
print("BASE_KEYS", sum(1 for k in keys if not k.endswith(".bitnet_scale")))

# group by top-level prefix
prefix_counts: Counter[str] = Counter()
for k in keys:
    prefix_counts[k.split(".")[0]] += 1
print("PREFIX_COUNTS", json.dumps(dict(sorted(prefix_counts.items())), indent=2))

# full dump
out = {k: {"shape": list(meta[k][0]), "dtype": meta[k][1]} for k in keys}
Path(r"C:\02_QUILLAN\scripts\_socket_header_full.json").write_text(
    json.dumps(out, indent=2), encoding="utf-8"
)

# base keys only dump
base = {k: list(meta[k][0]) for k in keys if not k.endswith(".bitnet_scale")}
Path(r"C:\02_QUILLAN\scripts\_socket_header_base.json").write_text(
    json.dumps(base, indent=2), encoding="utf-8"
)
print("WROTE full + base json")
