#!/usr/bin/env python3
"""Protected-invariant audit for the hifi g4 checkpoint (Nullion/Predator gate).
Verifies the reconversion did not break 2M context, hierarchy, council, swarm.
"""

from __future__ import annotations
import sys, json
from pathlib import Path

ROOT = Path(r"C:\02_QUILLAN")
for p in (
    str(ROOT / "scripts"),
    str(ROOT / "03 - Training & Model"),
    str(ROOT / "scripts" / "adapter_diagnostics"),
):
    if p not in sys.path:
        sys.path.insert(0, p)
from wsp_math_eval import load_model

FILES = [
    "quillan-mini-sovereign-bitnet.safetensors",
    "quillan-mini-hifi-g4-councilfixed.safetensors",
]


def main():
    report = {}
    for f in FILES:
        m = load_model(ROOT / "exports" / f)
        c = m.cfg
        d = {
            "max_position_embeddings": getattr(c, "max_position_embeddings", None),
            "orig_max_position_embeddings": getattr(
                c, "orig_max_position_embeddings", None
            ),
            "rope_theta": getattr(c, "rope_theta", None),
            "num_experts": getattr(c, "num_experts", None),
            "moe_layer_indices": list(getattr(c, "moe_layer_indices", ()) or ()),
            "eggroll_rank": c.eggroll_rank,
            "group_size": c.group_size,
            "n_layers": len(m.layers),
            "council_at_11_22": [
                m.layers[11].council_moe is not None,
                m.layers[22].council_moe is not None,
            ],
        }
        # council gamma alive?
        cg = {}
        for li in (11, 22):
            moe = m.layers[li].council_moe
            if moe is not None:
                cg[f"L{li}.w1_gamma_max"] = float(moe.w1_gamma.abs().max())
                cg[f"L{li}.w2_gamma_max"] = float(moe.w2_gamma.abs().max())
        d["council_gamma"] = cg
        report[f] = d
        print(f"=== {f} ===", flush=True)
        for k, v in d.items():
            print(f"  {k}: {v}", flush=True)
    # 2M context invariant
    print("\n=== 2M CONTEXT INVARIANT ===", flush=True)
    ok = all(r["max_position_embeddings"] == 2097152 for r in report.values())
    print(f"  all checkpoints max_position_embeddings=2097152: {ok}", flush=True)
    # council invariant
    cok = all(
        all(r["council_at_11_22"]) and all(v > 0 for v in r["council_gamma"].values())
        for r in report.values()
    )
    print(f"  council present + gamma alive at 11/22: {cok}", flush=True)
    (ROOT / "exports" / "hifi_g4_protected_invariant.json").write_text(
        json.dumps(report, indent=2)
    )
    print("\nsaved hifi_g4_protected_invariant.json", flush=True)


if __name__ == "__main__":
    main()
