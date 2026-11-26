#!/usr/bin/env python3
"""GLASS BOX — surgical freeze/unfreeze edit ledger (no backprop).

The user's protocol:
  1. FREEZE known-good sections (identity params + already-fixed sections).
  2. UNFREEZE one domain section (e.g. math, science, identity).
  3. Apply a KNOWN closed-form edit to only that section.
  4. Re-measure on DOMAIN-SCOPED gates + a global protected baseline.
  5. If improved AND no regression -> FREEZE that section (commit to ledger),
     move to next section. If not -> revert, log why it failed.

Everything recorded to a JSON ledger: what was edited, why, what changed,
what caused the fix or failure. This is the anti-black-box audit trail.

Note: 'frozen' here means CHANGE-CONTROL (do not touch again), not
requires_grad=False. Actual weight storage is in buffers (uint8 packed,
fp16 gamma, int8 embedding); all edits are direct closed-form tensor writes.
"""

from __future__ import annotations
import sys, json
from pathlib import Path
from datetime import datetime

ROOT = Path(r"C:\02_QUILLAN")
LEDGER = ROOT / "exports" / "glassbox_ledger.json"

# Domain sections of the model (logical groups for surgical editing)
SECTIONS = {
    "identity": ["embed_tokens", "norm", "layers.0.input_layernorm"],
    "attention": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "math_ffn": ["gate_proj", "up_proj", "down_proj"],
    "council": ["council_moe"],
}


def load_ledger():
    if LEDGER.exists():
        return json.loads(LEDGER.read_text())
    return {
        "version": "1.0",
        "created": datetime.now().isoformat(),
        "frozen": [],
        "edits": [],
        "principle": "freeze-good / edit-one-domain / measure / freeze-improvement",
    }


def save_ledger(L):
    LEDGER.write_text(json.dumps(L, indent=2))


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--freeze",
        type=str,
        default=None,
        help="section name to freeze (commit as known-good)",
    )
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--note", type=str, default="")
    a = ap.parse_args()
    L = load_ledger()
    if a.list:
        print(json.dumps(L, indent=2))
        return
    if a.freeze:
        if a.freeze not in SECTIONS:
            print(f"unknown section {a.freeze}; choose from {list(SECTIONS)}")
            return
        if a.freeze not in L["frozen"]:
            L["frozen"].append(a.freeze)
        L["edits"].append(
            {
                "time": datetime.now().isoformat(),
                "action": "FREEZE",
                "section": a.freeze,
                "note": a.note,
            }
        )
        save_ledger(L)
        print(f"FROZEN section={a.freeze} note={a.note}")
        print(f"frozen set now: {L['frozen']}")
        return
    print("GLASSBOX ledger:", LEDGER)
    print("frozen:", L["frozen"], "| edits logged:", len(L["edits"]))


if __name__ == "__main__":
    main()
