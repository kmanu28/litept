# RelFlexformer attention prototype

The checks in this directory test the attention mechanism independently before
model integration. The dependency-free implementation lives in
`litept/relflexformer_attention.py`.

Implemented from Algorithms 1 and 2 of the RelFlexformer paper:

- ReLU Performer feature map.
- A finite Fourier quadrature mask evaluated without an `N x N` matrix.
- Eight fixed, normally distributed frequencies per head by default.
- Fourier modulation `exp(-lambda * ||frequency||)`.
- Independent variable-length point clouds using cumulative offsets.

The implementation evaluates the constant-size quadrature directly with real
sines and cosines. Its cost is `O(N * S)` for fixed `S`; it does not claim to
use a NUFFT library. This is mathematically the finite sum in Algorithm 1 and
is practical when `S=8`.

## Explicit assumption

The paper introduces quadrature coefficients `a_s`, but does not disclose the
coefficients used in its segmentation experiments. The prototype therefore
uses fixed uniform coefficients `1/S`. Frequencies and coefficients are
buffers, not trainable parameters, and are deliberately excluded from the
state dict so existing pretrained checkpoints can still load strictly. This
assumption must remain visible in any later comparison with the paper.

The integrated mixed-precision path also exposes two numerical controls that
are not specified by the paper:

- `relflex_coord_scale` scales metric coordinates before Fourier phases.
- `relflex_eps` supplies a relative floor for signed row normalization.

The ScanNet experiment config currently uses `0.01` for both. Fourier
accumulations and normalization stay in FP32 before the result returns to the
model dtype. These choices prevented overflow on the cached 184-metre-wide
demo cloud and must be included in any reported experimental protocol.

Run the CPU checks from the repository root:

```bash
conda run -n litept python -m research.test_relflexformer_attention
```

The checks compare the factorized implementation with an exact dense
materialization of the same finite-quadrature operator, compare gradients,
verify cloud-boundary isolation, and exercise a zero-denominator edge case.
