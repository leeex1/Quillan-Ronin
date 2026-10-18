#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Clean README rebuild: raw restore from 45cc7be + ONI banner + version sweep.
NO reflow - the original formatting is already correct."""
import re
import subprocess
from pathlib import Path

REPO = r"C:\02_QUILLAN"
p = Path(REPO) / "README.md"

# 1. Raw restore (binary-safe via subprocess, no PowerShell encoding damage)
s = subprocess.run(["git", "-C", REPO, "show", "45cc7be:README.md"],
                   capture_output=True).stdout.decode("utf-8", errors="replace")
orig_lines = s.count("\n") + 1
print(f"[1] restored 45cc7be: {len(s)} chars, {orig_lines} lines")
assert "Success Stories" in s and "Model Specs" in s

# 2. Insert styled ONI banner before Model Specs
banner = Path(REPO + r"\scripts\_oni_banner.md").read_text(encoding="utf-8")
anchor = "## Model Specs"
assert anchor in s, "Model Specs anchor missing"
s = s.replace(anchor, banner + "\n" + anchor, 1)
print(f"[2] ONI banner inserted ({len(banner)} chars)")

# 3. Version sweep (surgical, same as before)
def rep(old, new):
    global s
    if old in s:
        s = s.replace(old, new, 1)
        return "OK"
    return "MISS"

results = []
results.append(("headline", rep(
    "**Quillan-Ronin v5.3.1**",
    "**Quillan-Ronin v5.4.0 \u201cONI\u201d**")))
results.append(("version", rep(
    "- **Version**: 5.2.2",
    "- **Version**: 5.4.0-oni")))
results.append(("params", rep(
    "- **Parameter Count**: 3 Billion (3B)",
    "- **Parameter Count**: 234M (6-layer proof) / ~390M (12-layer flagship) \u2014 ternary footprint ~90\u2013200 MB deployed")))
results.append(("architecture", rep(
    "- **Architecture**: Multi-Modal Mixture-of-Experts (MoE) with 33 specialized experts",
    "- **Architecture**: Unrolled Throne+Council \u2014 dense persona-pull Council MoE, 34 specialists (C1\u2013C34), rank-8 LoRA + rank-8 EGGROLL swarm per member")))
results.append(("quantization", rep(
    "- **Quantization**: 1.58-bit BitNet",
    "- **Quantization**: 1.58-bit BitNet ternary + STE \u00b7 INT8 activations \u00b7 fp32 routers \u00b7 INT8 KV-cache")))
results.append(("updates-current", rep(
    "    Current: v5.3.1 - Ronin",
    "    Current: v5.4.0-oni - Unified standalone model\n    - Canonical build: oni/ package + LINEAGE.md registry\n    - Unified custom BPE tokenizer (fixes all legacy vocab mismatches)\n    - Previous: v5.3.1 - Ronin (prompt era)")))
results.append(("version-history", rep(
    "**v5.3.1 (Samurai - Current Epoch)**",
    "**v5.4.0 (ONI \u2014 Current Epoch)**\n- **Standalone unified model:** every Samurai mechanism in weights \u2014 Throne pull-gate, dense 34-member deliberation, RoPE, Couil hybrid attention, Langevin diffusion, PID governor, analytic E_ICE, quality gates, deliberate() loop.\n- **Unified tokenizer:** single custom BPE \u2014 retires multi-tokenizer chaos.\n- **Canonical registry:** LINEAGE.md; one counter; branches retired.\n\n**v5.3.1 (Samurai \u2014 Prompt Era)**")))
results.append(("roadmap", rep(
    "## Coming Soon: v5.3.1:", "## Roadmap (v5.4.0-oni):")))
results.append(("samurai-edition", rep(
    "Quillan-Ronin (v5.3.1 Samurai Edition), architected by",
    "Quillan-Ronin (v5.4.0 Oni Edition \u2014 successor to the v5.3.1 Samurai prompt architecture), architected by")))
for name, r in results:
    print(f"[3] {name}: {r}")

# 4. Write (no reflow!)
p.write_text(s, encoding="utf-8", newline="\n")
final_lines = s.count("\n") + 1
print(f"[4] written: {len(s)} chars, {final_lines} lines")
for marker in ("Success Stories", "Comprehensive Benchmark Table", "Model Specs",
               "ONI", "Getting Help", "Usage Examples", "Key Takeaways"):
    print(f"    contains {marker!r}: {marker in s}")
