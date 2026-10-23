#!/usr/bin/env bash
set -euo pipefail
PROJECT=/mnt/d/isfcr/litept
VENV=/opt/litept-venv
OUTPUT=/mnt/c/litept-exp/scannet/official-litept-s-tta-v1m1
WEIGHT=/mnt/c/litept-exp/pretrained-scannet-litept-s.pth
mkdir -p "$OUTPUT/model"
cd "$PROJECT"
export CUDA_HOME=/usr/local/cuda-12.4
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$PROJECT"
exec "$VENV/bin/python" tools/test.py \
  --config-file configs/scannet/semseg-litept-small-full-official-tta-v1m1.py \
  --num-gpus 1 --options save_path="$OUTPUT" weight="$WEIGHT"
