# Preliminary 64-scene ScanNet mirror results

These results use the incomplete mirror only: 54 training scenes and 10
validation scenes. They are useful for a controlled local ablation, but they
are not official ScanNet benchmark results and must not be compared directly
with published full-split numbers.

## Protocol

- Same fixed 54/10 split for every variant.
- Pretrained ScanNet LitePT-S checkpoint followed by five short fine-tuning
  passes, batch size 1, seed 0, and the same optimizer/data transforms.
- Best checkpoint selected by validation mIoU.
- Single-pass test evaluation (`aug_transform = [[]]`), without rotations,
  flips, or multi-view TTA.
- Every result below was recomputed on 2026-07-31 in a new empty result
  directory to prevent the evaluator's prediction cache from reusing outputs
  from another checkpoint.
- RelFlex was retrained and re-profiled on 2026-08-01 after adding its explicit
  FP32 autocast boundary. The table uses those corrected checkpoints and
  measurements; baseline and exact-global values are unchanged.
- GPU: NVIDIA GeForce RTX 4060 Laptop GPU, 8,188 MiB.

## Results

| Variant | Target | mIoU | mAcc | allAcc | Eval s/scene | Scenes/s | Total real s | Profile step ms | Peak allocated GiB |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline | none | **0.4983** | **0.5821** | 0.9028 | 1.036 | 0.965 | 23.34 | 212.221 | 1.037 |
| Exact global | stage 5 | 0.4968 | 0.5796 | 0.9026 | **1.011** | **0.989** | **19.88** | 216.603 | 1.037 |
| Exact global | stages 4+5 | 0.4966 | 0.5711 | 0.9051 | 1.046 | 0.956 | 20.54 | 217.192 | **1.028** |
| RelFlex | stage 5 | 0.4937 | 0.5737 | **0.9055** | 1.075 | 0.930 | **19.66** | 235.614 | 1.120 |
| RelFlex | stages 4+5 | 0.2937 | 0.3747 | 0.7803 | 1.432 | 0.698 | 23.14 | 317.971 | 1.569 |

Baseline, exact-global stage 5, exact-global stages 4+5, and RelFlex stage 5
are within 0.0046 absolute mIoU of one another on this very small split.
RelFlex stages 4+5 is substantially worse and slower. No variant approaches
the laptop's 8 GiB VRAM limit.

The corrected RelFlex rerun shows that the S4+5 failure was not caused solely
by AMP overflow: mIoU moved only from 0.2890 to 0.2937. RelFlex S5 also stayed
close to its pre-fix result. This points to a mechanism or optimization issue
when RelFlex is applied at both late stages, not a simple OOM or NaN failure.

## Three-seed comparison

Baseline, exact-global S5, and corrected RelFlex S5 were repeated with seeds
0, 1, and 2. Each checkpoint was evaluated in its own empty result directory.
Values below are mean +/- sample standard deviation across the three seeds.

| Variant | mIoU | mAcc | allAcc |
|---|---:|---:|---:|
| Baseline | **0.5044 +/- 0.0080** | **0.5887 +/- 0.0061** | 0.9039 +/- 0.0026 |
| Exact global, stage 5 | 0.5040 +/- 0.0072 | 0.5869 +/- 0.0064 | **0.9039 +/- 0.0022** |
| RelFlex, stage 5 | 0.4952 +/- 0.0013 | 0.5773 +/- 0.0051 | 0.9020 +/- 0.0047 |

The baseline/global S5 mIoU means differ by only 0.0004, far below their
roughly 0.007--0.008 seed standard deviations. On this mirror there is no
evidence that exact-global S5 changes accuracy. RelFlex S5 is 0.0092 below the
baseline mean, though three seeds and ten biased validation scenes are still
too weak for a general benchmark claim.

Individual seed results are in `research/scannet_mirror_seed_results.csv` and
the aggregate values are in `research/scannet_mirror_seed_summary.csv`.

## Validation class coverage

The ten validation scenes contain 1,527,473 labelled points. Six of the 20
classes are completely absent: bed, picture, curtain, shower curtain, toilet,
and bathtub. Sink occurs in only two scenes and represents 0.086% of points.
Wall, floor, and chair together account for 68.66% of all validation points.

| Class | Scenes containing class | Points | Fraction |
|---|---:|---:|---:|
| Wall | 10/10 | 444,696 | 29.11% |
| Floor | 10/10 | 384,670 | 25.18% |
| Chair | 10/10 | 219,388 | 14.36% |
| Table | 6/10 | 105,557 | 6.91% |
| Sink | 2/10 | 1,315 | 0.086% |
| Six absent classes | 0/10 | 0 | 0% |

The complete coverage table is in
`research/scannet_mirror_class_coverage.csv`. This severe imbalance explains
why high allAcc coexists with much lower mIoU and reinforces that the mirror
cannot substitute for the official validation set.

Several classes report zero IoU/accuracy because this ten-scene validation
subset has extremely poor class coverage. This is one concrete reason the
aggregate values are biased and cannot replace the official ScanNet split.

The complete per-class IoU and accuracy table is in
`research/scannet_mirror_per_class.csv`; the machine-readable summary is in
`research/scannet_mirror_results.csv`. Full evaluator logs remain under the
corresponding `exp/scannet/subset-clean-eval-*` directories.

## Cache issue corrected

An older automatic baseline test row reported 0.1210 mIoU while its in-training
validation was about 0.50. The evaluator retains per-scene prediction files in
the experiment result directory. A clean directory recomputation with the same
baseline checkpoint produced 0.4983 mIoU, consistent with training validation.
All five rows in this document come from separate clean directories.
