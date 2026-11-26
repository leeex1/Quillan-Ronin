# FORGE v2 — Long-Context QUALITY Validation (brief #896)

**Verdict: long-context quality is NOT validated. Measurement was attempted on the
real trained checkpoint and could not be completed — the checkpoint fails a
short-context positive control. No claim of long-context quality is made.**

## Artifacts (all NEW files)
- `scripts/probe_longctx_quality.py` — CPU-only depth/length retrieval probe
- `scripts/probe_generate_sanity.py` — prompt-insensitivity root-cause diagnostic
- `scripts/longctx_quality_probe_results.json` — machine-readable measured result
- This report

## What was run
Environment: Python 3.11, torch CPU, 4 logical cores, `torch.get_num_threads()=3`.
Subject: `exports/quillan-mini-sovereign-bitnet158.safetensors`
(815,396,270 params, sha256 53e60d33..., strict load 530/530, dequantization
`int8 * bitnet_scale` verified against `export_quillan_sovereign_bitnet.py:74-112`).
Tokenizer: matching Qwen2.5-0.5B-Instruct (vocab 151936).

Task: single-token "magic word" copy/retrieval. Fact planted at controlled
absolute depth inside SOUL.md-derived filler; graded by (a) rank of the answer
token in the RAW last-position prefill distribution, (b) exact substring in the
greedy continuation. Short-context positive control at depth 64 / 512 tokens is
a hard gate: if it fails, no long-context conclusion is permitted.

## Measured results (exact)

### Throughput / budget
| prefill length | seconds | tok/s |
|---|---:|---:|
| 256 | 7.91 | 32.4 |
| 1024 | 15.66 | 65.4 |

Projection at the 1024 rate (OPTIMISTIC — `generate()` auto-shrinks the chunk
as `needed` grows: 8k->256, 32k->68, 128k->17 tokens/chunk):
- 8,192 tok: **125.3 s**
- 32,768 tok: **501.0 s**
- 131,072 tok: **2004.2 s (~33.4 min)** — exceeds the 600 s budget

Real-config KV: 14,128 B/token (hidden 896, heads 30, head_dim 29), so the
131,072-token active window = **1.77 GiB**; full 2,000,000 declared = 26.3 GiB
(does not fit host RAM with weights). No truncation; num_ctx unchanged.

### Positive control (depth 64, 512-token context) — FAILED
Best of 4 prompt formats: answer token rank **5,508 / 151,936** (top 3.6%),
p=2e-6, top-1=False, top-5=False, generated hit=False.
All four formats produced the identical degenerate continuation `",-,-,121"`.

### Root-cause isolation (short prompts, raw forward + generate)
| prompt | raw top-1 | greedy continuation |
|---|---|---|
| "The capital of France is" | `BOSE` (35.27) | "BOSE honda esl Fotos erotica raping cuckold nostalg" |
| "Hello, my name is" | `BOSE` (35.27) | identical tokens |
| "2 + 2 =" | `BOSE` (34.81) | identical tokens |
| magic-word prompt | `Nguồn` (25.69) | "Nguồn erotica ... hookers Prostit" |

Logits for the two 5-token prompts are **100.00% identical** (max abs diff
0.0000). The output is prompt-insensitive and toxic.

Embedding scale mismatch: `txt_emb` mean-abs **0.01198**, `pos_embed` mean-abs
**0.62258** — position signal is **~52x** token signal. In-memory rescale
(`pos_embed *= 0.02`, no files edited) changes top-5 from
`BOSE/honda/esl/若要/Fotos` to `1/2/_/./,` (and `2 + 2 =` surfaces `3` in
top-5), confirming the scale mismatch contributes — but coherence still does
not appear after naive rescale, so the defect is deeper (Qwen weights loaded
into a structurally different custom graph; donor fraction 0.9072 measures
parameter provenance, not functional transfer).

## Consequences
1. The repaired generation path's MECHANICS remain as verified in #881
   (6/6: real KV cache, 2M addressable positions, no O(L^2) masks, bit-identical
   short logits). This probe does not contradict that.
2. QUALITY at any length >=1024 is **unmeasured**, not "working". The
   instrument correctly refused to emit depth/degradation numbers because the
   model cannot perform the task even at depth 64.
3. The acceptance claim in #895 ("first real custom-architecture Quillan model
   with verified donor knowledge transfer") is verified only for weights,
   finiteness and provenance. Functional coherence is now falsified at short
   context and should be treated as an open blocker, not an inherited property.

## Required next steps (not done here)
- Fix token/position embedding scale at export or in `InputIngestionLayer`, then
  re-run `probe_generate_sanity.py` until short prompts yield coherent tokens.
- Re-run `probe_longctx_quality.py --full` once the positive control passes;
  budget 8k depth sweep first (~375 s), then 32k/128k in a separate run if the
  10-minute cap is lifted.
- Do not serve this checkpoint through the planned serving path (#896/c26)
  until short-context coherence is restored.
