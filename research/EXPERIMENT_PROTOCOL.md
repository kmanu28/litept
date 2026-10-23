# LitePT attention ablation protocol

## Variants

| Variant | Config |
|---|---|
| Baseline local | `configs/scannet/semseg-litept-small-v1m1.py` |
| Exact global, stage 5 | `configs/scannet/semseg-litept-small-globalattn-v1m1.py` |
| Exact global, stages 4+5 | `configs/scannet/semseg-litept-small-globalattn-stage45-v1m1.py` |
| RelFlex, stage 5 | `configs/scannet/semseg-litept-small-relflex-v1m1.py` |
| RelFlex, stages 4+5 | `configs/scannet/semseg-litept-small-relflex-stage45-v1m1.py` |

Stage numbers in reports are one-based. Internally these correspond to encoder
indices 4 and 3+4.

## Controlled settings

- Use one RTX 4060 Laptop GPU with 8 GiB VRAM.
- Use the existing `litept` conda environment; do not reinstall dependencies.
- Use the same pretrained checkpoint, data split, random seed, voxel size,
  optimizer, schedule, batch size, gradient accumulation, precision, and
  evaluation transforms for every variant.
- The current ScanNet evaluation config uses one test pass (`aug_transform =
  [[]]`) with no rotation, flip, or multi-view TTA.
- Record the exact commit/diff and full command line for each run.
- Report paper values only as external context, never as reproduced values.

## RelFlex definition used here

- PointROPE rotates query and key tensors first.
- The feature map is ReLU.
- Eight fixed normally distributed Fourier frequencies are used per head.
- Fourier modulation is `exp(-lambda * ||frequency||)`, with `lambda=1`.
- Uniform quadrature coefficients `1/S` are an explicit assumption because the
  paper does not disclose its experimental coefficients.
- Frequencies and coefficients add no parameters or state-dict entries.
- Direct constant-size Fourier quadrature is used rather than a NUFFT package.
- Metric coordinates are scaled by `0.01`.
- FP32 Fourier accumulation and a signed relative normalization floor of
  `0.01` are used for mixed-precision stability.

## Required validation before training

For every variant:

1. Load the same checkpoint with `strict=True`.
2. Confirm identical parameter count and output shape.
3. Confirm finite forward output, loss, and non-zero QKV gradients.
4. Record per-stage token counts and peak allocated/reserved CUDA memory.
5. Stop before an unsafe run if the 8 GiB preflight fails.

## Training

Official ScanNet training is blocked until the complete dataset is available.
The current mirror contains only 64 scenes (54 train and 10 validation), so it
is used only to exercise the pipeline and produce explicitly preliminary
comparisons. Start from the same checkpoint. Determine the largest safe common
batch size from the most memory-intensive variant, then use that same effective
batch size for every comparison. If gradient accumulation is needed, record
both micro-batch and effective batch sizes.

## Evaluation and profiling

Evaluate every fine-tuned checkpoint on the complete ScanNet validation split
with single-pass inference. Record:

- mIoU, mAcc, allAcc, and per-class IoU;
- end-to-end latency per scene and total evaluation time;
- scenes/second and processed points/second;
- peak allocated and reserved VRAM.

Profile the actual training loop using CPU and CUDA `torch.profiler`
activities, scheduled warmup followed by 10–20 active steps, with shapes,
memory, and stacks enabled. Sample `nvidia-smi dmon -s u` concurrently to
distinguish compute saturation from input-pipeline idle time.

S3DIS results, if added later, must be reported as a separate experiment. They
are not a substitute for ScanNet results in the same comparison table.

## Preliminary demo-cloud benchmark

This table is an implementation smoke/profile result, not an accuracy result.
It used the cached NuScenes demo cloud (34,688 raw points; 21,421 voxelized
tokens), two warmup passes, and five synchronized inference passes.

| Variant | Latency (ms) | Scenes/s | Tokens/s | Peak allocated (GiB) |
|---|---:|---:|---:|---:|
| Baseline | 80.269 | 12.458 | 266,865.8 | 0.274 |
| Exact global, stage 5 | 82.266 | 12.156 | 260,385.5 | 0.260 |
| Exact global, stages 4+5 | 111.054 | 9.005 | 192,888.0 | 0.260 |
| RelFlex, stage 5 | 106.224 | 9.414 | 201,658.1 | 1.171 |
| RelFlex, stages 4+5 | 184.323 | 5.425 | 116,214.2 | 1.171 |

All variants produced shape `(21421, 72)` and contained 12,712,932
parameters. Run:

```bash
conda run -n litept python tools/benchmark_attention_variants.py \
  --warmup 2 --iterations 5
```

## Single-scene zero-shot regression

This is only an end-to-end regression on `scene0011_00`, not a ScanNet
validation result. Every variant loaded the same local-attention ScanNet
checkpoint from epoch 83 without fine-tuning and processed the same 11
single-pass fragments.

| Variant | Fragment batch time (s) | mIoU | mAcc | allAcc |
|---|---:|---:|---:|---:|
| Baseline | 2.226 | 0.4155 | 0.4572 | 0.8918 |
| Exact global, stage 5 | 1.784 | 0.4180 | 0.4600 | 0.8935 |
| Exact global, stages 4+5 | 1.815 | 0.4040 | 0.4467 | 0.8892 |
| RelFlex, stage 5 | 1.959 | 0.3916 | 0.4288 | 0.8705 |
| RelFlex, stages 4+5 | 2.355 | 0.0676 | 0.1196 | 0.5030 |

The stage-4+5 RelFlex collapse is a zero-shot checkpoint mismatch signal. It
does not establish how a fine-tuned model will perform. The corresponding logs
are under `exp/scannet/regression-*-single-scene/test.log`.

The tester currently raises an existing post-evaluation exception after
writing metrics because `wandb.run` is `None` but `wandb.run.step` is accessed.
This does not invalidate the completed metrics above, but should be corrected
or disabled consistently before automated full-validation runs.

## Mini-ScanNet pipeline smoke test

The mirror was checksum-verified, selectively extracted, preprocessed, and
validated as 54 training scenes with 7,896,094 points and 10 validation scenes
with 2,091,456 points. These are mini-subset diagnostics, not official ScanNet
benchmark results.

Both variants used seed 0, batch size 1, one epoch (54 optimizer steps), one
evaluation pass, and the same official pretrained checkpoint. The original
one-cycle learning-rate schedule was retained only to verify the unmodified
training path. It is too aggressive for interpreting a one-epoch fine-tune:
loss did not decrease from the first ten to the final ten iterations.

| Variant | mIoU | mAcc | allAcc | Result scope |
|---|---:|---:|---:|---|
| Baseline | 0.3858 | 0.4367 | 0.8142 | 10-scene mini validation |
| RelFlex, stage 5 | 0.3166 | 0.3973 | 0.8176 | 10-scene mini validation |

Both runs completed forward, backward, checkpoint saving, and precise testing
without OOMs or NaNs.

## Controlled five-epoch mini-subset fine-tune

For a less destructive checkpoint adaptation, all variants were rerun from the
same official checkpoint for five epochs (270 optimizer steps) with maximum
learning rates reduced by 10x to `0.0006` for non-block parameters and
`0.00006` for block parameters. All other mini-subset controls remained
identical. Strict loading reported no missing or unexpected keys.

| Variant | mIoU | mAcc | allAcc | Avg. precise-test scene time (s) | Scenes/s |
|---|---:|---:|---:|---:|---:|
| Baseline | 0.4983 | 0.5821 | 0.9028 | 0.995 | 1.005 |
| Exact global, stage 5 | 0.4991 | 0.5820 | 0.9031 | 0.936 | 1.068 |
| Exact global, stages 4+5 | 0.4966 | 0.5711 | 0.9051 | 0.946 | 1.057 |
| RelFlex, stage 5 | 0.4966 | 0.5771 | 0.9054 | 0.985 | 1.015 |
| RelFlex, stages 4+5 | 0.2890 | 0.3734 | 0.7796 | 1.334 | 0.750 |

The baseline, both exact-global variants, and stage-5 RelFlex are close on
these ten validation scenes; none of the differences is evidence of
equivalence or an accuracy trend. Stage-4+5 RelFlex remained substantially
worse after this short adaptation, although its training loss decreased from
2.3333 in epoch 1 to 0.8732 in epoch 5. The subset is too small and incomplete
for benchmark claims or statistical conclusions.

Run each row with the same explicit checkpoint argument:

```bash
sh scripts/train.sh -p "$CONDA_PREFIX/bin/python" -g 1 -d scannet \
  -c semseg-litept-small-subset-finetune-v1m1 \
  -n baseline-subset-finetune-pretrained -w <model_best.pth>

sh scripts/train.sh -p "$CONDA_PREFIX/bin/python" -g 1 -d scannet \
  -c semseg-litept-small-globalattn-subset-finetune-v1m1 \
  -n global-s5-subset-finetune-pretrained -w <model_best.pth>

sh scripts/train.sh -p "$CONDA_PREFIX/bin/python" -g 1 -d scannet \
  -c semseg-litept-small-globalattn-stage45-subset-finetune-v1m1 \
  -n global-s45-subset-finetune-pretrained -w <model_best.pth>

sh scripts/train.sh -p "$CONDA_PREFIX/bin/python" -g 1 -d scannet \
  -c semseg-litept-small-relflex-subset-finetune-v1m1 \
  -n relflex-s5-subset-finetune-pretrained -w <model_best.pth>

sh scripts/train.sh -p "$CONDA_PREFIX/bin/python" -g 1 -d scannet \
  -c semseg-litept-small-relflex-stage45-subset-finetune-v1m1 \
  -n relflex-s45-subset-finetune-pretrained -w <model_best.pth>
```

## Real-loop training profile on the mini subset

The following uses the actual training loader and optimization step with
`torch.profiler`: one wait step, two warmup steps, and ten active steps, with
CPU and CUDA activities, shapes, memory, and stacks enabled. The reported step
time includes profiler overhead and should be used only for the paired
comparison.

| Variant | Profiled step (ms) | Steps/s | Peak allocated (GiB) | Peak reserved (GiB) |
|---|---:|---:|---:|---:|
| Baseline | 212.221 | 4.712 | 1.037 | 1.141 |
| Exact global, stage 5 | 216.603 | 4.617 | 1.037 | 1.141 |
| Exact global, stages 4+5 | 217.192 | 4.604 | 1.028 | 1.131 |
| RelFlex, stage 5 | 233.511 | 4.282 | 1.089 | 1.303 |
| RelFlex, stages 4+5 | 308.273 | 3.244 | 1.339 | 1.816 |

Relative to baseline, exact global stage 5 was 2.1% slower, exact global
stages 4+5 was 2.3% slower, RelFlex stage 5 was 10.0% slower, and RelFlex
stages 4+5 was 45.3% slower in the profiled windows. The largest reserved
footprint was 1.816 GiB, safely below the 8 GiB limit.

One-second `nvidia-smi dmon -s u` sampling saw bursty compute during these
short runs, with active samples reaching 91--95% SM utilization but varying
widely between samples. Since steady data time was approximately 2 ms versus
212--308 ms per profiled step, there is no evidence that this batch-size-one
configuration is data-loader-bound.

On this WSL2 driver/runtime combination, the profiler recorded CUDA runtime
events and CUDA memory but emitted no GPU kernel events or per-operation CUDA
time column. Therefore no per-op CUDA-time ranking is claimed. The complete
traces and CPU/runtime tables are under
the corresponding `exp/scannet/*-subset-profile/profile/` directories. The
consolidated numeric rows are also stored in
`research/mini_scannet_attention_ablation.csv`.

## Official ScanNet handoff

Once the complete processed dataset is present, activate `litept` and run the
guarded five-variant launcher. It refuses to run on the 54/10-scene mirror and
requires at least 1200 training and 300 validation scene directories:

```bash
conda activate litept
RUN_CONFIRM=official-scannet \
  scripts/run_scannet_attention_ablation.sh <pretrained_model_best.pth>
```

This launches the same five-epoch checkpoint-adaptation protocol used for the
mini-subset table. It does not claim to reproduce the paper's 1200-epoch
training protocol. Longer schedules, if used, must be applied identically to
all five configurations and documented as a separate experiment.
