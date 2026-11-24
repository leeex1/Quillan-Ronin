# sm_61 GPU Capability — TESTED BREAKTHROUGH (2026-11-23)

## ⚠️ Correction to prior claim

Prior handoff said "PyTorch cannot use the GTX 1050 — hardware limit." **That was wrong.**
It is true only for the *installed* torch 2.12.1+cu130 build (no sm_61 kernels →
`CUDA error: no kernel image is available`). CUDA 11.8 DOES support sm_61.

## Direct evidence (isolated venv, no system changes)

Venv: `C:\02_QUILLAN\venv_sm61_test` (Python 3.11.14 + torch 2.7.1+cu118, installed from
`https://download.pytorch.org/whl/cu118`). Install exit 0.

```
torch 2.7.1+cu118 | is_available True | device NVIDIA GeForce GTX 1050 | capability (6,1)
ALLOC OK cuda:0 torch.Size([256, 256])
MATMUL OK cuda:0 275.20635986328125
VERDICT: real CUDA ops EXECUTE on sm_61
```

Benchmark (1024x1024 matmul, 30 iters):
```
BENCH cpu=18.92ms gpu=1.90ms speedup=9.95x
```

## Why this was missed

The installed system torch is 2.12.1+cu130. PyTorch dropped Pascal (sm_50/sm_60)
from cu128+ wheels. `torch.cuda.is_available()` returns True misleadingly, but actual
ops fail with "no kernel image." The web search was blocked by bot detection; a direct
runtime test was required — and it initially showed the failure on cu130. Testing the
older CUDA line with the official previous-versions index revealed the working path.

## What this means / what is NOT done

- The GTX 1050 CAN accelerate PyTorch work at ~10x for matmul-class ops.
- The venv is ISOLATED — the model agent's Python 3.14 / torch 2.12 environment is untouched.
- Do NOT replace system torch with cu118: model scripts target Python 3.14 and torch 2.12 APIs;
  downgrading would break the active model work.
- To exploit this: either (a) run sm61-compatible inference/training inside this venv, or
  (b) have the model agent add a GPU inference path that imports from this venv.
- iGPU (HD 630) still has no backend; it cannot join. Only the NVIDIA card is usable.

## Setup command (reproduce)
```
py311="...\uv\python\cpython-3.11.14-windows-x86_64-none\python.exe"
& $py311 -m venv C:\02_QUILLAN\venv_sm61_test
& C:\02_QUILLAN\venv_sm61_test\Scripts\python.exe -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu118
```
