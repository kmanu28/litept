#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$(dirname "$0")")"

if [[ $# -ne 1 ]]; then
    echo "Usage: RUN_CONFIRM=official-scannet $0 <pretrained_model_best.pth>" >&2
    exit 2
fi

checkpoint="$1"
python_bin="${CONDA_PREFIX:-}/bin/python"
train_root="data/scannet/train"
val_root="data/scannet/val"

if [[ "${RUN_CONFIRM:-}" != "official-scannet" ]]; then
    echo "Refusing a long run without RUN_CONFIRM=official-scannet." >&2
    exit 2
fi
if [[ ! -f "$checkpoint" ]]; then
    echo "Checkpoint not found: $checkpoint" >&2
    exit 2
fi
if [[ ! -x "$python_bin" ]]; then
    echo "Activate the litept conda environment before running this script." >&2
    exit 2
fi

train_scenes="$(find "$train_root" -mindepth 1 -maxdepth 1 -type d | wc -l)"
val_scenes="$(find "$val_root" -mindepth 1 -maxdepth 1 -type d | wc -l)"
if (( train_scenes < 1200 || val_scenes < 300 )); then
    echo "Official ScanNet preflight failed." >&2
    echo "Found ${train_scenes} train and ${val_scenes} validation scenes." >&2
    echo "Expected approximately 1201 train and 312 validation scenes." >&2
    exit 2
fi

configs=(
    semseg-litept-small-subset-finetune-v1m1
    semseg-litept-small-globalattn-subset-finetune-v1m1
    semseg-litept-small-globalattn-stage45-subset-finetune-v1m1
    semseg-litept-small-relflex-subset-finetune-v1m1
    semseg-litept-small-relflex-stage45-subset-finetune-v1m1
)
names=(
    official-baseline-finetune
    official-global-s5-finetune
    official-global-s45-finetune
    official-relflex-s5-finetune
    official-relflex-s45-finetune
)

echo "ScanNet preflight passed: ${train_scenes} train / ${val_scenes} validation."
echo "Running five matched five-epoch fine-tunes from: $checkpoint"
for index in "${!configs[@]}"; do
    sh scripts/train.sh \
        -p "$python_bin" \
        -g 1 \
        -d scannet \
        -c "${configs[$index]}" \
        -n "${names[$index]}" \
        -w "$checkpoint"
done
