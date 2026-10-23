#!/usr/bin/env bash
set -euo pipefail

PROJECT=/mnt/d/isfcr/litept
VENV=/opt/litept-venv
OUTPUT_ROOT=/mnt/c/litept-exp/scannet
export CUDA_HOME=/usr/local/cuda-12.4
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$PROJECT"

# The full ScanNet configuration intentionally uses a project-relative
# data_root (data/scannet_full), so training must start from the repository.
cd "$PROJECT"

run_variant() {
  local config="$1"
  local name="semseg-litept-small-full-${config}-v1m1"
  local out="$OUTPUT_ROOT/$name"
  mkdir -p "$out/model"
  "$VENV/bin/python" "$PROJECT/tools/train.py" \
    --config-file "$PROJECT/configs/scannet/$name.py" \
    --num-gpus 1 --num-machines 1 --machine-rank 0 --dist-url auto \
    --options save_path="$out"
}

run_variant globalattn
run_variant relflex
