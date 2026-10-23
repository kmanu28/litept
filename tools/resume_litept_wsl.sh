#!/usr/bin/env bash
set -euo pipefail

PROJECT=/mnt/d/isfcr/litept
VENV=/opt/litept-venv
OUTPUT=/mnt/c/litept-exp/scannet/semseg-litept-small-full-baseline-seed0

while ! "$VENV/bin/python" -c "import torch, flash_attn, pointrope, spconv; assert torch.cuda.is_available()" >/dev/null 2>&1; do
  echo "Waiting for completed LitePT CUDA runtime..."
  sleep 60
done

mkdir -p "$OUTPUT/model"
cp "$PROJECT/artifacts/scannet_full/baseline/runtime_config.py" "$OUTPUT/config.py"
cp "$PROJECT/artifacts/scannet_full/baseline/checkpoints/model_last_cycle20.pth" "$OUTPUT/model/model_last.pth"

cd "$PROJECT"
export CUDA_HOME=/usr/local/cuda-12.4
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$PROJECT"

exec "$VENV/bin/python" tools/train.py \
  --config-file "$OUTPUT/config.py" \
  --num-gpus 1 \
  --num-machines 1 \
  --machine-rank 0 \
  --dist-url auto \
  --options save_path="$OUTPUT" resume=True weight="$OUTPUT/model/model_last.pth"
