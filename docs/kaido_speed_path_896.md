# C14-KAIDO — Speed path: prefix caching + keep_alive (brief #896)

**Agent:** c14-kaido · **Request:** `8d2bb927-7f04-47a2-b685-f6e4a750ace0` · **Date:** 2026-11-20
**Scope:** NEW files only (`scripts\kaido_prefix_cache_serve.ps1`, `scripts\kaido_prefix_cache_measure.py`, this doc, two result JSONs). No model, GGUF, Modelfile, or live config changed.

---

## 1. Exact serving configuration

Runnable file: `C:\02_QUILLAN\scripts\kaido_prefix_cache_serve.ps1`
Model: `C:\02_QUILLAN\exports\quillan-1b-sovereign-wsp-v23-q8_0.gguf`
Prompt: `C:\02_QUILLAN\06 - Deployment & Platforms\system prompts\Quillan-Samurai.md` — 171,434 tokens (Llama-3.2 tokenizer, board #874). **Never truncated.**

| Flag / env | Value | Why (evidence) |
|---|---|---|
| `-c` / `OLLAMA_CONTEXT_LENGTH` | `262144` | Active window. Model log confirms `n_ctx_seq (262144) < n_ctx_train (2097152)` — the 2M declared context is preserved. q4_0 KV = 9,216 B/tok (board #863): 171,434 tok = 1.47 GiB; 256k = 2.25 GiB. A real 2M allocation = 18.0 GiB and fails on 7.78 GiB free RAM / 4 GiB GPU. |
| `-ctk`, `-ctv` / `OLLAMA_KV_CACHE_TYPE` | `q4_0` | 4x KV compression; 18 GiB → 2.25 GiB at 256k. |
| `-fa on` / `OLLAMA_FLASH_ATTENTION=1` | on | **Required**: measured server error `V cache quantization requires flash_attn` when `-fa off` with q4_0 V. Board #863: FA auto-enables on sm_61. |
| `-ngl 99` | all layers | Board #863: `offloaded 17/17 layers to GPU` on GTX 1050 sm_61. |
| `-kvo` | host KV | Weights stay on CUDA0, KV in host RAM. Needed for 256k. **Measured cost: see §2 — decode drops ~16x.** |
| `--cache-prompt` | on | The lever. Server log: `prompt cache is enabled, size limit: 8192 MiB`; 1.47 GiB full-prompt KV fits that limit. Default on, pinned explicit. |
| `--cache-reuse 256` | 256 tokens | `--help`: "min chunk size to attempt reusing from the cache via KV shifting" (default `0` = off). |
| `-np 1` / `OLLAMA_NUM_PARALLEL=1` | one slot | One cacheable prefix; N slots fragment KV. |
| `--metrics`, `--slots` | on | Observe cache state / slot prefix retention. |
| `-t 4 -tb 4 --no-mmap` | 4 threads | Board #863 config; avoids pageouts under low free RAM. |
| `OLLAMA_KEEP_ALIVE=24h`, per-request `{"keep_alive":-1}` | pin | llama.cpp has **no** `--keep-alive` flag; process residency is permanent and the slot retains its KV prefix. Ollama equivalent keeps model resident so the cache is not evicted. `OLLAMA_MAX_LOADED_MODELS=1` prevents eviction by another model. |
| `OLLAMA_HOST` | `127.0.0.1:11434` | Ollama-compatible API surface. |

Declared `num_ctx=2097152` remains in the Modelfile (metadata); `-c 262144` is the realized attention window. Declared capacity ≠ realized span — stated plainly.

---

## 2. Bounded measurement (real system-prompt file, cold vs cache reuse)

Harness: `scripts\kaido_prefix_cache_measure.py`. Two identical 30,000-char slices of `Quillan-Samurai.md` (offset 200,000 → **7,513 tokens**), different 1-token user probes. First-token latency ≈ `prompt_ms` (prefill completes before token 1). All numbers below are from `llama-server` `timings` + its own log lines — nothing invented.

### 2a. Deployment-shaped config, host KV (`-kvo`, 32k measurement window)
`-c 32768 -ctk q4_0 -ctv q4_0 -fa on -ngl 99 -kvo --cache-prompt --cache-reuse 256 -np 1 -t 4 -tb 4 --no-mmap`
Result: `docs\kaido_speed_measurement.json`

| Run | Prefill tokens | First-token latency | Prefill rate | Decode (16 tok) | Decode rate |
|---|---|---|---|---|---|
| cold | 7,513 | **18,980.86 ms** | 395.82 t/s | 12,722.96 ms | **1.26 t/s** |
| cache reuse | **2** | **401.93 ms** | (overhead-dominated) | 10,430.66 ms | **1.53 t/s** |

Server log: `prompt eval time = 18980.86 ms / 7513 tokens (395.82 tokens per second)` → `prompt eval time = 401.93 ms / 2 tokens`. **7,511 of 7,513 tokens were served from cache.**

### 2b. GPU KV (`-nkvo`, same 32k window) — throughput comparison
Result: `docs\kaido_speed_measurement_gpu_kv.json`

| Run | Prefill tokens | First-token latency | Prefill rate | Decode (16 tok) | Decode rate |
|---|---|---|---|---|---|
| cold | 7,513 | **12,290.17 ms** | 611.30 t/s | 685.80 ms | **23.33 t/s** |
| cache reuse | **2** | **43.31 ms** | (overhead-dominated) | 630.59 ms | **25.37 t/s** |

Cache-hit proof, verbatim from server log: `selected slot by LCP similarity, sim_best = 1.000 (> 0.100 thold), f_keep = 0.998`.

**Key finding — correction to the prior audit:** prefix caching is a huge **first-token-latency** lever (18.98 s → 0.40 s with host KV; 12.29 s → 0.043 s with GPU KV) and is quality-neutral, but it does **not** change decode tok/s. The dominant speed lever on this host is KV placement: host KV offload (required for the 256k active window) measured 1.26–1.53 t/s vs 23.33–25.37 t/s with GPU KV — a ~16x difference. Both are real measurements; neither is the 46.3 t/s short-prompt CLI figure, which used a 2k context and no long KV.

---

## 3. Full 171,434-token projection (measurement aborted as too heavy)

A literal full-prompt cold run was launched with the §1 256k config and aborted after ~4 min of CLI wall time. Its own progress log gives measured cumulative prefill points:

| Prefill tokens | Cumulative time | Rate |
|---|---|---|
| 2,048 | 12.96 s | 158.04 t/s |
| 4,096 | 34.34 s | 119.28 t/s |
| 6,144 | 65.19 s | 94.24 t/s |
| 8,192 | 101.05 s | 81.07 t/s |
| 10,240 | 137.75 s | 74.34 t/s |

Marginal time for the last 2,048-token chunk = 36.70 s → **55.8 t/s**. Lower-bound projection for the unprocessed 161,194 tokens:
`137.75 s + 161,194 / 55.8 = 137.75 + 2,888.8 ≈ 3,027 s ≈ 50.4 min`.

This is a **lower bound**: attention/bandwidth cost grows with prefix length, so the real cold full-prompt prefill is ≥50 min. With the cache warm, the same request reuses up to all 171,434 tokens and only the new user tokens are prefilled — measured at 43.31 ms (GPU KV) / 401.93 ms (host KV) for a 2-token suffix at 7.5k context; at 171k the cache lookup itself is unmeasured, so do not promise sub-second there.

---

## 4. Caveats / gates

- **Hard constraints honored:** `num_ctx` declaration unchanged; full 171,434-token prompt untouched; no truncation; no invented numbers.
- The 256k launch succeeded once, then two relaunches hit `CUDA error: out of memory` (host free RAM had fallen to ~3.9–6.0 GiB). Production needs a dedicated process / RAM headroom check before serving.
- `-fa off` is **not** a valid fallback with q4_0 V cache (`V cache quantization requires flash_attn`). If FA rejects a kernel, quantize only K or keep FA on.
- GPU KV at 32k works; GPU KV at 256k was not validated and is likely over the 3.36 GiB free VRAM once weights + compute are added.
- Custom-arch checkpoint (#894) still has no llama.cpp backend (#875); all serving here uses the standard-arch WSP GGUF. No claim otherwise.
