#!/usr/bin/env bash
set -euo pipefail

while [ ! -x /usr/local/cuda-12.4/bin/nvcc ]; do
  echo "Waiting for CUDA Toolkit installation..."
  sleep 30
done

export CUDA_HOME=/usr/local/cuda-12.4
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export PIP_DEFAULT_TIMEOUT=1800
export MAX_JOBS=4

PROJECT=/mnt/d/isfcr/litept
VENV=/opt/litept-venv

python3 -m venv "$VENV"
source "$VENV/bin/activate"
python -m pip install --upgrade pip
python -m pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu124
python -m pip install h5py pyyaml sharedarray tensorboard tensorboardx wandb yapf addict einops scipy plyfile termcolor timm colorhash torchmetrics packaging psutil ninja wheel
python -m pip install torch-cluster torch-scatter torch-sparse torch-geometric==2.5.3 -f https://data.pyg.org/whl/torch-2.4.0+cu124.html
python -m pip install spconv-cu124
python -m pip install flash-attn --no-build-isolation

cd "$PROJECT/libs/pointrope"
TORCH_CUDA_ARCH_LIST=8.9 python setup.py install

cd "$PROJECT"
PYTHONPATH="$PROJECT" python - <<'PY'
import torch
import flash_attn
import pointrope
import spconv

assert torch.cuda.is_available(), "CUDA is not visible to PyTorch"
print("GPU:", torch.cuda.get_device_name(0))
print("Torch:", torch.__version__)
print("FlashAttention and PointROPE imports: OK")
PY
