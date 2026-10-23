# Official ScanNet v2 baseline: intermediate result

## Status

The full official ScanNet v2 pipeline completed successfully:

- raw assets validated for 1,513 labeled scans;
- preprocessing completed for 1,201 training, 312 validation, and 100 test
  scenes;
- LitePT-S baseline trained through evaluation cycle 20 of 100;
- training stopped cleanly after saving a resumable cycle-20 checkpoint.

Each logged evaluation cycle contains 12 dataset loops, so cycle 20 represents
240 of the configured 1,200 effective epochs (20%).

## Protocol

| Setting | Value |
|---|---|
| Dataset | Official ScanNet v2 |
| Split | 1,201 train / 312 validation |
| Model | LitePT-S baseline, local grouped attention |
| Parameters | 12.7M (12,714,392 in the training log) |
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU, 8 GiB |
| GPUs | 1 |
| Batch size | 1 |
| Data workers | 4 |
| AMP | Enabled |
| Configured schedule | 1,200 effective epochs / 100 evaluation cycles |
| Completed schedule | 240 effective epochs / 20 cycles |
| Validation | Single pass; no multi-view TTA |

The model, optimizer, augmentations, scheduler, and evaluation logic inherit
from `semseg-litept-small-v1m1.py`. The full-data child config changes only the
data root and laptop resource settings (batch size, worker count, seed, and
W&B disablement).

## Intermediate results

| Checkpoint | Cycle | Effective epochs | mIoU | mAcc | allAcc |
|---|---:|---:|---:|---:|---:|
| Best | 18 | 216 | **0.6400** | **0.7475** | **0.8610** |
| Last | 20 | 240 | 0.5770 | 0.6582 | 0.8304 |

All cycle-level measurements are in
[`scannet_full_baseline_cycles.csv`](scannet_full_baseline_cycles.csv), and the
best checkpoint's 20-class breakdown is in
[`scannet_full_baseline_best_per_class.csv`](scannet_full_baseline_best_per_class.csv).

## Interpretation limits

These values are real full-validation measurements, but they are intermediate
training checkpoints. They must not be presented as the completed LitePT paper
result. The paper reports 76.5% ScanNet validation mIoU for LitePT-S under its
own larger-batch, multi-GPU training and test protocol. This laptop run uses
batch size 1 and single-pass validation, so it is a compute-constrained baseline
for controlled local/global/RelFlex comparisons rather than an exact paper
reproduction.

The ScanNet dataset is not included in Git because redistribution is restricted.
