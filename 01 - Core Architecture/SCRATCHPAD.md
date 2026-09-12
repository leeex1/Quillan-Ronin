---
name: scratchpad
description: Quantum-entangled scratchpad — working memory where all 135 papers interfere
---

# SCRATCHPAD — Entangled Working Memory

## Not a File, a Superposition

Scratchpad is `|Ψ_work⟩` — the transient state where Papers 1-135 interfere before collapse to `USER.md`.

```
scratchpad = (profiler wall_total_ms) ⊗ (xMem total_mb) ⊗ (coordination order) ⊗ (abductive axiom)
```

- **Paper 1 Profiler:** `wall_total_ms` lives here transiently before `step_profile.jsonl`
- **Paper 4 Memo + 11 xMem:** `α` and `total_mb` computed here, then entangled
- **Paper 16-20 Persona:** `self.coherence_loss` computed here before `persona_embed`
- **Paper 31 Prefix Sliding:** `gen[-W:]` lives here before `kv_cache`

## Entanglement

Writing to scratchpad entangles all papers: update `GRT.gate_B.bias` (Paper 21) → scratchpad holds `g_t` → affects `DynamicCompression.rate` (Paper 26) → affects `Metan.depth` (Paper 30).

## Location

`C:\02_QUILLAN\SCRATCHPAD.md` is the human-readable collapse. The real scratchpad is `C:\Users\Admin\AppData\Local\Temp\opencode\` + `torch.cuda.memory_allocated()`.

Use `context7` to query scratchpad state: `serena: find in scratchpad`.
