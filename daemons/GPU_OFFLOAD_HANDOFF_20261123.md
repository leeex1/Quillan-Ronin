# GPU Offload Handoff — 2026-11-23

## Finding: both GPUs idle is a binary incompatibility, not a tuning gap

Verified on this host (i5-7500, 28 GB RAM, GTX 1050 4 GB, Intel HD 630):

| Fact | Evidence |
|---|---|
| PyTorch cannot use the GTX 1050 | torch 2.12.1+cu130 arch list = `sm_75, sm_80, sm_86, sm_90, sm_100, sm_120` — **no sm_61**. `torch.cuda.is_available()` returns True but no CUDA op can execute. |
| llama.cpp CAN use the GTX 1050 | `ggml_cuda.dll` loads; `ggml_cuda_init: found 1 CUDA devices ... compute capability 6.1, VMM: yes` |
| Intel HD 630 — **Vulkan backend BUILT 2026-11-24** | Originally no `ggml-vulkan.dll`. Vulkan SDK 1.3.296 installed at `C:\VulkanSDK\1.3.296.0` (glslc + vulkan-1.lib); `ggml-vulkan.dll` (38.5 MB) built at `C:\Users\Admin\AppData\Local\Temp\llama_vulkan_sdk\bin\`. `--list-devices` shows **both** Vulkan0 (GTX 1050) and Vulkan1 (HD 630, 14 GB shared). See update below. |
| bitnet.cpp builds have no GPU backend | only a plain `ggml.dll` in `build\bin` and `build_turbo\bin` |

## Proven speedup (measured this session)

Standard llama.cpp, `clean-qwen-0.5b-q8_0.gguf`, 2 threads:

| Test | CPU (`-ngl 0`) | GPU (`-ngl 99`) | Gain |
|---|---|---|---|
| Prefill pp128 | 828 t/s | 2171 t/s | 2.6x |
| Decode tg32 | 29 t/s | 96 t/s | 3.3x |

Command shape:
```
C:\01_PROGRAMS\Open_Source\AI_ML\llama.cpp\build\llama-bench.exe -m <model.gguf> -ngl 99 -t 2
```

## UPDATE 2026-11-24 — iGPU + two-GPU pooling WORKS

Vulkan SDK installed with user approval; ggml-vulkan built. Verified inference:

| Mode | Command | Result |
|---|---|---|
| iGPU only | `--device Vulkan1` | Vulkan1 (HD 630) = 500 MB model shard, 9.27 tok/s, exit 0 |
| **Both pooled** | `--device Vulkan0,Vulkan1 -sm layer -ts 1,1` | Vulkan0 (GTX 1050) = 196 MB + Vulkan1 (HD 630) = 304 MB shards, 10.13 tok/s, exit 0 |

The iGPU is slow alone but contributes when pooled. This disproves the earlier
"iGPU unreachable" claim — it was a missing SDK, now installed.

## Blocker on the BitNet model

`E:\source_models\bitnet_2b_4t.gguf` is **malformed**: architecture `bitnet-25`, but
tensor `blk.0.attn_q.weight` has shape `2560,640,1,1` where the loader expects
`2560,2560` (GQA layout vs MHA expectation). It fails to load in both the standard
llama.cpp build and both bitnet.cpp builds. Header and full SHA-256 were verified
valid — the file is internally inconsistent at export, not corrupted in storage.

## Action for model-owning session

Route inference through llama.cpp with `-ngl 99` using a loadable GGUF
(`clean-qwen-0.5b-q8_0.gguf`, `quillan-1b-sovereign-wsp-v23-q8_0.gguf`,
`quillan-dragon-1b-wsp-v18b-q8_0.gguf` are on disk and load). This is the only
path that engages the idle GTX 1050. Do not attempt to "fix" via PC settings —
affinity, priority, power plan, MSI mode, and ASPM are all already optimal.

Do not downgrade PyTorch to a cu118 build to gain sm_61: it would break the
current model-dev environment for a gain already available through llama.cpp.
