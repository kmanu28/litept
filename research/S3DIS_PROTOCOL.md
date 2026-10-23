# S3DIS backup protocol

S3DIS is a separate backup experiment while official ScanNet access is
pending. Its metrics must never be presented as ScanNet metrics or compared
directly with ScanNet validation numbers.

## Access and preprocessing

1. Submit Stanford's official access form:
   https://docs.google.com/forms/d/e/1FAIpQLScDimvNMCGhy_rmBA2gHfDu3naktRm6A8BPwAWWDv-Uhm6Shw/viewform
2. Download and extract `Stanford3dDataset_v1.2.zip` from the confirmation
   page. Do not commit or redistribute the archive or its derivatives.
3. Correct the known malformed value in
   `Area_5/office_19/Annotations/ceiling_1.txt` if preprocessing reports it.
4. Preprocess all six areas with one worker on this laptop:

```bash
conda activate litept
python datasets/preprocessing/s3dis/preprocess_s3dis.py \
  --dataset-root <Stanford3dDataset_v1.2> \
  --output-root data/s3dis \
  --align-angle \
  --num-workers 1
python tools/validate_s3dis_processed.py --data-root data/s3dis
```

The expected processed structure is
`data/s3dis/Area_N/room_name/{coord,color,segment,instance}.npy`.

## Controlled setup

- Standard Area-5 protocol: Areas 1, 2, 3, 4, and 6 train; Area 5 validates.
- Thirteen semantic classes.
- Single-pass validation and testing (`aug_transform = [[]]`).
- Batch size 1 and four workers for the 8 GiB laptop.
- The official point annotations provide XYZRGB but no normals. Every variant
  therefore uses normalized RGB plus transformed XYZ as six input features.
- The published LitePT model collection has no S3DIS checkpoint. Initial
  S3DIS comparisons therefore train from scratch unless a separately
  documented transfer protocol is introduced.
- Use identical seeds, schedules, batches, and transforms for every attention
  variant.

## Configurations

| Variant | Config |
|---|---|
| Baseline | `configs/s3dis/semseg-litept-small-v1m1.py` |
| Exact global, stage 5 | `configs/s3dis/semseg-litept-small-globalattn-v1m1.py` |
| Exact global, stages 4+5 | `configs/s3dis/semseg-litept-small-globalattn-stage45-v1m1.py` |
| RelFlex, stage 5 | `configs/s3dis/semseg-litept-small-relflex-v1m1.py` |
| RelFlex, stages 4+5 | `configs/s3dis/semseg-litept-small-relflex-stage45-v1m1.py` |

Run baseline first after preprocessing:

```bash
sh scripts/train.sh -p "$CONDA_PREFIX/bin/python" -g 1 -d s3dis \
  -c semseg-litept-small-v1m1 -n s3dis-baseline
```

Do not launch all five long runs until the baseline loader, forward/backward,
checkpoint, and Area-5 evaluation paths have passed.

## Laptop pilot completed on 2026-07-30

The local preprocessing validation passed for all 272 rooms and 273,546,487
points:

| Area | Rooms | Points |
|---|---:|---:|
| 1 | 44 | 44,026,810 |
| 2 | 40 | 47,315,372 |
| 3 | 23 | 18,662,173 |
| 4 | 49 | 43,470,014 |
| 5 | 68 | 78,719,063 |
| 6 | 48 | 41,353,055 |

The controlled pilot uses the corresponding `*-pilot-v1m1.py` configs. Each
model was trained from scratch with seed 0 for ten passes (2,040 optimizer
steps), batch size 1, AMP, and no mix augmentation. The reported metrics come
from the saved best checkpoint and a full, single-pass evaluation of all 68
Area-5 rooms. This is a compute-constrained pipeline comparison, not a
paper-matching S3DIS result.

| Variant | Target | mIoU | mAcc | allAcc | Eval s/room |
|---|---|---:|---:|---:|---:|
| Baseline | none | 0.3004 | 0.3754 | 0.7083 | 2.505 |
| Exact global | stage 5 | 0.3003 | 0.3783 | 0.7197 | 2.605 |
| Exact global | stages 4+5 | 0.3006 | 0.3781 | 0.7159 | 2.905 |
| RelFlex | stage 5 | 0.2878 | 0.3640 | 0.7068 | 2.823 |
| RelFlex | stages 4+5 | 0.2890 | 0.3665 | 0.7086 | 3.891 |

The original RelFlex AMP run first became non-finite at step 269. RelFlex
reductions are now explicitly excluded from CUDA autocast while the rest of
LitePT remains under AMP. The regression completed 408/408 real-data steps
with finite losses, and both 2,040-step pilots subsequently completed without
NaNs. This boundary adds no parameters or state-dict keys.

Training-step measurements include loading a real 102,400-point crop, host to
GPU transfer, forward, backward, gradient scaling, and the optimizer step.
They use five warm-up and twenty measured steps, batch size 1, four workers,
and synchronized CUDA timing on an NVIDIA GeForce RTX 4060 Laptop GPU
(8,188 MiB, driver 581.86):

| Variant | Target | s/step | steps/s | points/s | Peak allocated GiB | Peak reserved GiB |
|---|---|---:|---:|---:|---:|---:|
| Baseline | none | 0.2792 | 3.581 | 366,737 | 0.774 | 0.836 |
| Exact global | stage 5 | 0.2640 | 3.788 | 387,939 | 0.774 | 0.836 |
| Exact global | stages 4+5 | 0.2656 | 3.765 | 385,581 | 0.775 | 0.838 |
| RelFlex | stage 5 | 0.2811 | 3.557 | 364,266 | 0.824 | 0.881 |
| RelFlex | stages 4+5 | 0.2977 | 3.359 | 343,938 | 1.069 | 1.291 |

These short timing measurements are sensitive to data-loader and laptop
variance. The evaluation latency over all 68 rooms is the stronger inference
comparison; none of the variants approached the 8 GiB VRAM limit.

`torch.profiler` was also invoked with CPU and CUDA activities,
`record_shapes`, `profile_memory`, and `with_stack`. In this WSL2 session it
recorded CUDA memory/runtime activity but exposed zero CUDA device events, so
a top-operators-by-CUDA-time table cannot be reported honestly from this
machine. The synchronized wall times and allocator peaks above remain valid;
kernel attribution requires a working CUPTI trace in a different host setup.

Reproduce a training-step profile with:

```bash
conda activate litept
python tools/profile_s3dis_training.py \
  configs/s3dis/semseg-litept-small-pilot-v1m1.py
```

The machine-readable table is in `research/s3dis_pilot_results.csv`. Generated
datasets, checkpoints, and experiment logs remain outside version control.
