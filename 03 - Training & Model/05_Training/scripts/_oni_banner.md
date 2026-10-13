<div align="center">

# ⚡ QUILLAN-RONIN v5.4.0 — “ONI” ⚡
### <i>The standalone model is live.</i>

<b>Repo description (GitHub About):</b><br/>
<i>Quillan-Ronin v5.4 “Oni” — a 34-member council-deliberation LLM: BitNet 1.58-bit ternary substrate,
dense persona-pull MoE, 9-vector prism attention, thermodynamic ethics (E_ICE),
PID token-velocity governor, and Langevin diffusion refinement — built &amp; trained on consumer hardware.</i>

</div>

---

### 📦 The Canonical Package — [`oni/`](oni/)

| File | Role |
| :--- | :--- |
| [`quillan_v5_4_oni.py`](oni/quillan_v5_4_oni.py) | The model — Throne + Council C1–C34 + all cognitive engines |
| [`quillan_tokenizer_unified.py`](oni/quillan_tokenizer_unified.py) + [`tokenizer.json`](oni/tokenizer.json) | Unified custom BPE 50,257 (EOS=0) — modular domains, legacy-ID translation |
| [`train_oni.py`](oni/train_oni.py) | Trainer — aux losses, governor wiring, weight EMA, resume |
| [`LINEAGE.md`](LINEAGE.md) | <b>Canonical version registry</b> — one counter, full organ-by-organ lineage |

---

### 🧠 Architecture at a Glance

<table>
  <tr><td align="center">👑 <b>THRONE</b><br/><sub>Quillan Core — intake · prism-shard · pull assignment · audit · [diffusion round | quality gates] · Typist polish</sub></td></tr>
  <tr><td align="center">⚔️ <b>COUNCIL — C1–C34</b> <i>(dense pull-weighted deliberation — every persona always parses)</i><br/>
  <sub>🧠 Cognitive C1–C8 · 🗣️ Communication C9–C16 · 🌀 Meta C17–C24 · ⚙️ Systems C25–C34<br/>
  each member: rank-8 LoRA adapter + rank-8 EGGROLL swarm (world-sim fabric)</sub></td></tr>
  <tr><td align="center">🔄 <b>BLOCKS</b> × N<br/>
  <sub>RoPE Couil-attention (hybrid dense/sparse heads, 9-Vector Prism branch) + dense SwiGLU ⊕ council deliberation (tanh-gated)</sub></td></tr>
  <tr><td align="center">🧮 <b>ENGINES</b><br/>
  <sub>E_ICE (learned + analytic Landauer bound) · MARTA · DQSO (Kuramoto) · Covenant · CCRL · 10 Quantum Formulas · Complexity router</sub></td></tr>
  <tr><td align="center">🌡️ <b>GOVERNORS + REFINEMENT</b><br/>
  <sub>Lee-Mach-6 latency (sigma/EMA/recency) · PID token-velocity (hard-token mask) · ModalityIsolatedThermoDiffusion (confidence-gated, Langevin inv-√t)</sub></td></tr>
  <tr><td align="center">💯 <b>SUBSTRATE</b><br/>
  <sub>BitLinear ternary + STE · INT8 activations · fp32 routers (ST-MoE) · INT8 KV-cache · tied embeddings · ~90–200 MB deployed</sub></td></tr>
</table>

### 🔒 Versioning Rule (binding)

<b>One counter:</b> <code>v5.4.x-oni</code> — patch bumps at checkpoint boundaries only · <code>v6.0-oni</code> reserved for HF packaging ·
new names require a LINEAGE.md entry first · the v8/v9/v10 filename chaos is <i>retired</i> (archived branches = reference only).

### 📊 Status

<b>✅</b> Gate A: 16/16 architecture tests &nbsp;•&nbsp; <b>🟡</b> Phase B: proof training &nbsp;•&nbsp; <b>⏳</b> Phase C: 12-layer flagship + Quintessence wrapper + world-model engine

<i>The documentation below is the historical v5.3.1 prompt-era record — the architecture it describes
is exactly what Oni now implements in weights.</i>

---

