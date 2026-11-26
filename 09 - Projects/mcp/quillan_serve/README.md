# Quillan-Ronin-Mini — Custom-Architecture Serving Path

## 🧭 Canonical project context

| | |
| :--- | :--- |
| **This repo** | PyTorch-native, Ollama-compatible serving path for the 815M custom-architecture Mini. |
| **Canonical package** | Three deployed models: Quillan-Ronin Flagship (~1B), Quillan-Ronin-Mini (~0.8B) and Quillan-Dragon (~1B). The ONI Python sources are implementation/training only. |
| **Donor design** | Flagship and Mini: Qwen 0.6B + Llama ~1B + BitNet 2B-4T + Falcon3 ~1B **as a group** (cross-match fix in progress). Dragon: Falcon3 with an Ollama system message running Quillan's system prompt. |
| **Reading the scores** | The 95–100% table in the root README is **Track A** — the Quillan harness on a third-party frontier base, not a Quillan checkpoint. Hard-exam cells are pending re-test; ARC-AGI-3 public ~99s, private pending. |
| **Historical framing** | Edo period, rōnin and bushidō are real Japanese history and ethics — not roleplay, game mechanics, or fantasy lore. |
| **Canonical docs** | [Root README](https://github.com/leeex1/Quillan-Ronin#readme) · [Fleet model cards](https://github.com/leeex1/Quillan-Ronin/blob/main/docs/MODEL_CARDS.md) |

<!-- QUILLAN-DOCS-PROVENANCE:BEGIN -->
<details>
<summary><b>📌 Quillan documentation provenance</b> — expand for canonical sources</summary>

> **Documentation provenance.** This repo is part of the Quillan-Ronin ecosystem (CrashOverrideX / `leeex1`). Canonical sources live in the root repository: `README.md` (**READ FIRST** — Track A harness-on-frontier vs. Track B native WSP models vs. Track B′ custom Mini) and `docs/MODEL_CARDS.md` (fleet identity, verified facts, open gaps). The large 95–100% benchmark table in the root README is a **harness-on-frontier** result, not a score of any Quillan-owned checkpoint. Creator artifacts (books, music, game demo) have primary-source links in the root README's **Creator and verifiable artifacts** section.  Four-donor design: Flagship and Mini receive all four donors (Qwen ~0.6B + Llama ~1B + BitNet 2B-4T + Falcon3 ~1B) **as a group**; Dragon is Falcon3 with an Ollama system message running Quillan's system prompt. Cross-match fix in progress.  The canonical package is the three deployed models — Quillan-Ronin Flagship (~1B), Quillan-Ronin-Mini (~0.8B) and Quillan-Dragon (~1B); the ONI Python sources are the implementation/training package, not the shipped product. Canonical official links (site, social, repos, Ollama, HF, Grokipedia, NotebookLM, DeepWiki, Genius, YouTube/Suno) are in the root README's **Official links** directory. Historical framing is real Japanese history (Edo period, rōnin, bushidō) used as an ethical reference frame — not roleplay, game mechanics, or fantasy lore.

</details>

<!-- QUILLAN-DOCS-PROVENANCE:END -->

**Serves the real 815M `QuillanRoninSovereign` checkpoint** (`exports/quillan-mini-sovereign-bitnet158.safetensors`, sha256 `53e60d33…ac7a5c`, 530/530 strict keys) through an Ollama-compatible HTTP API.

This is **PyTorch-native inference**, not a proxy. No existing server, model, or exporter was edited.

## Option selected: (a) PyTorch + FastAPI/uvicorn

| Option | Verdict | Reason |
|---|---|---|
| **(a) PyTorch FastAPI/uvicorn server** | **CHOSEN** | Reuses the repaired `generate()` + real KV cache verbatim; runs the exact custom class; no backend substitution. |
| (b) custom llama.cpp graph registration | Rejected | No GGUF graph exists: `export_phase5_gguf.py:157-177` explicitly falls back to `.pt` for custom arch. Direct llama.cpp (sm_61, 46.3 tok/s) serves a *different* model family. Registering a graph would require re-exporting weights into an unsupported architecture. |
| (c) bridge wrapper to Ollama/llama.cpp | Rejected | Would silently serve the wrong architecture or force `num_ctx` down — forbidden by board #855/#861/#895. |

Model identity strings stay **Quillan-Ronin / Quillan-Dragon / Quillan-Ronin-Mini**. Served tag: `quillan-ronin-mini:latest`.

## Exact launch command

Python 3.11 environment is required (it has FastAPI 0.162.0, uvicorn 0.54.0, torch 2.12.0+cpu, safetensors, tokenizers):

```powershell
& 'C:\Users\Admin\.local\bin\python3.11.exe' -m uvicorn quillan_serve.server:app `
  --app-dir 'C:\02_QUILLAN\09 - Projects\mcp' `
  --host 127.0.0.1 --port 11435
```

Alternative (same interpreter, no `--app-dir`):

```powershell
& 'C:\Users\Admin\.local\bin\python3.11.exe' 'C:\02_QUILLAN\09 - Projects\mcp\quillan_serve\server.py'
```

Startup loads and verifies the checkpoint (~14.8 s measured). Default bind is loopback only. Override with `QUILLAN_SERVE_HOST` / `QUILLAN_SERVE_PORT`.

## Health check

```powershell
Invoke-RestMethod http://127.0.0.1:11435/health | ConvertTo-Json -Depth 6
# or: curl http://127.0.0.1:11435/health
```

Expect `"status":"ok"`, `"params":815396270`, `"sha256":"53e60d33…ac7a5c"`, `"loaded_keys":530`, `"context_policy":{"declared_num_ctx":2000000,"active_window_tokens":262144,"prompt_truncated":false}`.

### Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | load status, checksum, param count, memory policy |
| GET | `/api/version` | Ollama-shaped version |
| GET | `/api/tags` | model card incl. declared `num_ctx` and ACTIVE window |
| GET | `/api/show?model=quillan-ronin-mini` | modelfile + digest + policy |
| POST | `/api/generate` | generation (`stream:false` JSON, `stream:true` NDJSON) |

```powershell
Invoke-RestMethod -Uri http://127.0.0.1:11435/api/generate -Method Post -ContentType 'application/json' `
  -Body '{"model":"quillan-ronin-mini:latest","prompt":"Hello","stream":false,"options":{"num_predict":8,"temperature":0}}'
```

`options.system` (or top-level `system`) is prepended **in full** — never shortened. `stop` strings and EOS ids `151645/151643/50256` terminate generation.

## Context policy — DECLARED vs ACTIVE (read this)

| Quantity | Value | Status |
|---|---|---|
| Declared `num_ctx` / `max_position_embeddings` | **2,097,152 (2M)** | **Preserved.** Appears in `/api/tags`, `/api/show`, health. |
| Physically allocated ACTIVE KV window | **262,144 (256k)** | Actual preallocation, fp32. |
| Required system prompt (~171k tokens) | **not truncated** | Fits inside 256k; full prompt is prefilled in bounded chunks. |
| Client `num_ctx` override | ignored with explicit warning | No Ollama backend shortcut is used. |
| Request exceeding 256k | **HTTP 400**, error names the constraint | Never silently shortened. |

### Why full 2M cannot be physically served on this host — stated plainly

Measured KV cost for this checkpoint is **14,128 B/token** (`2*896 + 2*30*29` elements × 4 B, fp32):

- Full declared 2M KV cache = **26.315 GiB** alone.
- FP32 weights = 815,396,270 × 4 B = **3.038 GiB**.
- Total ≈ **29.35 GiB**, against roughly **28 GiB host RAM** (board test #881 assumption) — it does not fit; with the runtime and tokenizer it would swap/thrash.
- The GPU is a GTX 1050 4 GB (sm_61, ~3 GB free) — it cannot hold 3.04 GiB weights **plus** a 3.45 GiB 256k KV cache either, so the honest serving device is **CPU FP32**.
- The model itself caps direct uncached forwards at 4,096 queries (`MAX_DIRECT_CAUSAL_QUERIES`); long prompts require the repaired chunked-prefill KV path, which is exactly what is used here.

FP16 cache is explicitly disabled: `cached_step()` casts only the query to cache dtype, then fp32 `BitLinear` projection layers would receive fp16 input — a dtype mismatch. So the 256k window is fp32 (3.449 GiB).

**Conclusion:** the 2M declaration is retained as the product contract; 256k is the honest *active* window. Positions beyond the active window are addressable by the NTK-sinusoidal position scheme but are not simultaneously resident in KV memory.

## Smoke tests (in-process, no external network)

```powershell
& 'C:\Users\Admin\.local\bin\python3.11.exe' 'C:\02_QUILLAN\09 - Projects\mcp\quillan_serve\smoke_test.py'
& 'C:\Users\Admin\.local\bin\python3.11.exe' 'C:\02_QUILLAN\09 - Projects\mcp\quillan_serve\smoke_test_api.py'
```

`smoke_test.py` calls the engine directly. `smoke_test_api.py` drives the FastAPI app via `TestClient` (ASGI transport — no socket, no Ollama, no llama.cpp).

Measured on this host (2026-11-20), Python 3.11, torch 2.12.0+cpu, 4 logical CPUs / 3 torch threads:

- Load: **14.82 s**; strict keys 530/530; params 815,396,270; sha256 verified.
- Engine test, prompt `"Quillan-Ronin is a sovereign"`, 12 tokens, greedy:
  - exact output: `' Fotos erotica cuckold nostalg,Th/lg/stretch😉,wwwBOSE cosplay揶'`
  - **0.3715 tok/s** (12 tokens in 32.30 s), `done_reason=length`.
- Endpoint test, prompt `"Ronin:"`, 2 tokens:
  - exact output `'😉☝'`, **0.2311 tok/s**; `/health` 200, `/api/tags` 200, `/api/generate` 200, over-window request 400, `num_ctx=2048` override ignored with warning.
- `SMOKE_PASS=True`, `API_SMOKE_PASS=True`.

## Honest limitations (no fake PASS)

1. **Speed:** 0.23–0.37 tok/s on CPU FP32. This is a correctness/reference path, not a production throughput path. A faster GPU path is not available at 256k active window on the 4 GB card.
2. **Output quality:** short-prompt samples above are incoherent and emit spurious special/emoji tokens. This server proves the *checkpoint actually runs*; it does not claim frontier quality. Accuracy/PPL on the frozen held-out items via a PyTorch eval path is still pending (board #895).
3. **Long-context quality:** positions ≥1024 remain untrained; mechanics are fixed and validated (`test_arch_generation_path.py` 6/6), quality is not (board #896).
4. **Streaming:** `stream:true` reuses the KV cache one token at a time (two forwards/step due to the repaired `generate()` recomputing the last prompt token on cache reuse) — slower than non-streaming. Prefer non-streaming for throughput.
5. `prompt_eval_duration` is reported as 0: prefill and decode occur inside one `model.generate()` call, so they are not separately measurable; exact combined `eval_duration` is reported instead.
6. No `/api/chat` endpoint in this first cut; use `/api/generate`. No authentication — bind to loopback.

## Files

- `engine.py` — checkpoint dequantisation, strict load, tokenizer, generation, guards, health.
- `server.py` — FastAPI app + Ollama-compatible routes.
- `smoke_test.py` / `smoke_test_api.py` — in-process tests.
- `README.md` — this file.
