#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Restyle the README canonical banner to match the doc's HTML-styled aesthetic."""
from pathlib import Path

p = Path(r"C:\02_QUILLAN\README.md")
s = p.read_text(encoding="utf-8", errors="replace")
start = s.find("## \u26a1 CURRENT BUILD")
end = s.find("## Model Specs")
assert start > 0 and end > start, (start, end)
old_block = s[start:end]

new_block = """<div align="center">

# \u26a1 QUILLAN-RONIN v5.4.0 \u2014 \u201cONI\u201d \u26a1
### <i>The standalone model is live.</i>

<b>Repo description (GitHub About):</b><br/>
<i>Quillan-Ronin v5.4 \u201cOni\u201d \u2014 a 34-member council-deliberation LLM: BitNet 1.58-bit ternary substrate,
dense persona-pull MoE, 9-vector prism attention, thermodynamic ethics (E_ICE),
PID token-velocity governor, and Langevin diffusion refinement \u2014 built &amp; trained on consumer hardware.</i>

</div>

---

### \U0001f4e6 The Canonical Package \u2014 [`oni/`](oni/)

| File | Role |
| :--- | :--- |
| [`quillan_v5_4_oni.py`](oni/quillan_v5_4_oni.py) | The model \u2014 Throne + Council C1\u2013C34 + all cognitive engines |
| [`quillan_tokenizer_unified.py`](oni/quillan_tokenizer_unified.py) + [`tokenizer.json`](oni/tokenizer.json) | Unified custom BPE 50,257 (EOS=0) \u2014 modular domains, legacy-ID translation |
| [`train_oni.py`](oni/train_oni.py) | Trainer \u2014 aux losses, governor wiring, weight EMA, resume |
| [`LINEAGE.md`](LINEAGE.md) | <b>Canonical version registry</b> \u2014 one counter, full organ-by-organ lineage |

---

### \U0001f9e0 Architecture at a Glance

<table>
  <tr><td align="center">\U0001f451 <b>THRONE</b><br/><sub>Quillan Core \u2014 intake \u00b7 prism-shard \u00b7 pull assignment \u00b7 audit \u00b7 [diffusion round | quality gates] \u00b7 Typist polish</sub></td></tr>
  <tr><td align="center">\u2694\ufe0f <b>COUNCIL \u2014 C1\u2013C34</b> <i>(dense pull-weighted deliberation \u2014 every persona always parses)</i><br/>
  <sub>\U0001f9e0 Cognitive C1\u2013C8 \u00b7 \U0001f5e3\ufe0f Communication C9\u2013C16 \u00b7 \U0001f300 Meta C17\u2013C24 \u00b7 \u2699\ufe0f Systems C25\u2013C34<br/>
  each member: rank-8 LoRA adapter + rank-8 EGGROLL swarm (world-sim fabric)</sub></td></tr>
  <tr><td align="center">\U0001f504 <b>BLOCKS</b> \u00d7 N<br/>
  <sub>RoPE Couil-attention (hybrid dense/sparse heads, 9-Vector Prism branch) + dense SwiGLU \u2295 council deliberation (tanh-gated)</sub></td></tr>
  <tr><td align="center">\U0001f9ee <b>ENGINES</b><br/>
  <sub>E_ICE (learned + analytic Landauer bound) \u00b7 MARTA \u00b7 DQSO (Kuramoto) \u00b7 Covenant \u00b7 CCRL \u00b7 10 Quantum Formulas \u00b7 Complexity router</sub></td></tr>
  <tr><td align="center">\U0001f321\ufe0f <b>GOVERNORS + REFINEMENT</b><br/>
  <sub>Lee-Mach-6 latency (sigma/EMA/recency) \u00b7 PID token-velocity (hard-token mask) \u00b7 ModalityIsolatedThermoDiffusion (confidence-gated, Langevin inv-\u221at)</sub></td></tr>
  <tr><td align="center">\U0001f4a0 <b>SUBSTRATE</b><br/>
  <sub>BitLinear ternary + STE \u00b7 INT8 activations \u00b7 fp32 routers (ST-MoE) \u00b7 INT8 KV-cache \u00b7 tied embeddings \u00b7 ~90\u2013200 MB deployed</sub></td></tr>
</table>

### \U0001f512 Versioning Rule (binding)

<b>One counter:</b> <code>v5.4.x-oni</code> \u2014 patch bumps at checkpoint boundaries only \u00b7 <code>v6.0-oni</code> reserved for HF packaging \u00b7
new names require a LINEAGE.md entry first \u00b7 the v8/v9/v10 filename chaos is <i>retired</i> (archived branches = reference only).

### \U0001f4ca Status

<b>\u2705</b> Gate A: 16/16 architecture tests &nbsp;\u2022&nbsp; <b>\U0001f7e1</b> Phase B: 6-layer proof run in flight &nbsp;\u2022&nbsp; <b>\u23f3</b> Phase C: 12-layer flagship + Quintessence wrapper + world-model engine

<i>Everything below this section is the historical v5.3.1 prompt-era documentation \u2014 kept for the record.
The architecture it describes is exactly what Oni now implements in weights.</i>

---

"""
s = s.replace(old_block, new_block, 1)
p.write_text(s, encoding="utf-8", newline="\n")
print("restyled OK, size:", len(s))
