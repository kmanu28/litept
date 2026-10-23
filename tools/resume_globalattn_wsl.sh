#!/usr/bin/env bash
set -euo pipefail

PROJECT=/mnt/d/isfcr/litept
VENV=/opt/litept-venv
OUTPUT=/mnt/c/litept-exp/scannet/semseg-litept-small-full-globalattn-v1m1

cd "$PROJECT"
export CUDA_HOME=/usr/local/cuda-12.4
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$PROJECT"

exec "$VENV/bin/python" tools/train.py \
  --config-file "$OUTPUT/config.py" \
  --num-gpus 1 --num-machines 1 --machine-rank 0 --dist-url auto \
  --options save_path="$OUTPUT" resume=True weight="$OUTPUT/model/model_last.pth"
