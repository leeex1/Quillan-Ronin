#!/usr/bin/env bash
# build_torch_sm61.sh — Path B: from-source torch build for sm_61 (Pascal).
# The ONLY way to run torch newer than 2.14 with working CUDA on Pascal.
# Run on the machine with the card. Expect several hours and ~100 GB of disk.
# Usage:  PYTORCH_VERSION=v2.14.0 ./build_torch_sm61.sh
#
# Preflight: CUDA 12.x toolkit required (12.6-12.9 can target sm_61;
# CUDA 13.x REMOVED Pascal codegen — nvcc 13 fails outright on compute_61).
set -euo pipefail

PYTORCH_VERSION="${PYTORCH_VERSION:-v2.14.0}"
BUILD_DIR="${BUILD_DIR:-$HOME/torch-sm61-build}"

echo "=== preflight ==="
command -v nvcc >/dev/null || { echo "FAIL: nvcc not found — install CUDA 12.x toolkit"; exit 1; }
NVCC_VER="$(nvcc --version | grep -oE 'release [0-9]+\.[0-9]+' | grep -oE '[0-9]+\.[0-9]+')"
NVCC_MAJOR="${NVCC_VER%%.*}"
echo "nvcc CUDA: $NVCC_VER"
if [ "$NVCC_MAJOR" -ge 13 ]; then
  echo "FAIL: CUDA $NVCC_VER cannot target sm_61 (Pascal removed in CUDA 13). Use 12.6-12.9."
  exit 1
fi
python -c "import torch" 2>/dev/null && echo "NOTE: existing torch will be replaced by the build." || true
nvidia-smi --query-gpu=compute_cap --format=csv,noheader | grep -q "6.1" \
  || { echo "WARN: no sm_6.1 GPU visible to nvidia-smi — continuing anyway."; }

mkdir -p "$BUILD_DIR" && cd "$BUILD_DIR"
if [ ! -d pytorch ]; then
  git clone --recursive https://github.com/pytorch/pytorch.git
fi
cd pytorch
git fetch --tags
git checkout "$PYTORCH_VERSION"
git submodule sync && git submodule update --init --recursive

export TORCH_CUDA_ARCH_LIST="6.1"
export CMAKE_CUDA_ARCHITECTURES="61"
export MAX_JOBS="${MAX_JOBS:-$(nproc)}"
export USE_CUDA=1 USE_CUDNN=1 USE_MKLDNN=1 BUILD_TEST=0

echo "=== building torch $PYTORCH_VERSION for sm_61 (jobs=$MAX_JOBS) ==="
python setup.py install

echo "=== verify with a real kernel launch ==="
python -c "import torch; print(torch.__version__, torch.cuda.get_arch_list()); \
x=torch.randn(64,64,device='cuda'); print('matmul OK', float((x@x.t()).sum()))"
echo "=== done ==="
