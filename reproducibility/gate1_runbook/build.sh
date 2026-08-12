#!/bin/bash
# Gate 1 microbenchmark build — runs inside an NVIDIA CUDA devel container so
# the rented box needs nothing but Docker + NVIDIA Container Toolkit.
#
# Usage:
#   ./build.sh                 # -arch=native (recommended on the target box)
#   ARCH=sm_103a ./build.sh    # explicit B300 target
#   IMAGE=nvidia/cuda:13.0.0-devel-ubuntu24.04 ./build.sh
set -euo pipefail
cd "$(dirname "$0")"

ARCH="${ARCH:-native}"
IMAGE="${IMAGE:-nvidia/cuda:13.0.0-devel-ubuntu24.04}"

echo "[build] image=$IMAGE arch=$ARCH"
docker run --rm --gpus all -v "$PWD":/w -w /w "$IMAGE" bash -c "
  set -e
  nvcc --version | tail -1
  nvcc -O3 -arch=$ARCH imma_peak.cu -o imma_peak
  echo '[build] imma_peak OK'
"
echo "[build] done. run with: docker run --rm --gpus all -v \$PWD:/w -w /w $IMAGE ./imma_peak"
