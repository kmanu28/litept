"""Pure-PyTorch fallback for flash_attn.flash_attn_varlen_qkvpacked_func.

flash-attn ships no prebuilt Windows wheels, and its from-source build
requires a working MSVC toolchain; on machines without one, the CUDA
extension cannot be built at all. This module provides a numerically
equivalent (not bit-identical) but unfused substitute -- plain scaled-dot-
product attention per packed variable-length segment instead of
FlashAttention's fused/tiled kernel -- auto-selected only when the real
`flash_attn` package cannot be imported. This mirrors the pattern
`libs/pointrope` already uses for its own CUDA/PyTorch fallback.

This is a disclosed, deliberate compute-environment substitution, not a
transparent drop-in: results are mathematically equivalent modulo floating-
point summation order, but they are not bit-identical to a flash_attn-
produced checkpoint. Any resumed or fine-tuned run using this fallback
should be reported as such (see research/EXPERIMENT_PROTOCOL.md).
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

try:
    import flash_attn as _flash_attn

    HAS_FLASH_ATTN = True
except Exception as exc:  # pragma: no cover - environment dependent
    print(
        f"[LitePT] flash_attn unavailable ({type(exc).__name__}: {exc}). "
        "Using slower PyTorch fallback for varlen attention."
    )
    HAS_FLASH_ATTN = False


def _sdpa_varlen_fallback(
    qkv: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
    dropout_p: float = 0.0,
    softmax_scale: Optional[float] = None,
) -> torch.Tensor:
    """Pure-PyTorch equivalent of flash_attn_varlen_qkvpacked_func.

    qkv: [total_N, 3, num_heads, head_dim], packed variable-length sequences.
    cu_seqlens: 1D int tensor of cumulative sequence-length boundaries, length B+1.
    Returns: [total_N, num_heads, head_dim].
    """
    if qkv.ndim != 4 or qkv.shape[1] != 3:
        raise ValueError(f"Expected qkv shape [N,3,H,D], got {tuple(qkv.shape)}")
    total_n, _, num_heads, head_dim = qkv.shape
    q, k, v = qkv.unbind(dim=1)  # each [N, H, D]

    seqlens = cu_seqlens[1:] - cu_seqlens[:-1]
    if seqlens.numel() == 0:
        return qkv.new_zeros((0, num_heads, head_dim))

    if bool((seqlens == seqlens[0]).all()):
        # Fast batched path: every segment is the same length. Always true for
        # LitePT's local/patchified attention (patches are padded to a fixed
        # patch_size), and true for global attention whenever every point
        # cloud in the batch has the same point count -- e.g. batch_size=1,
        # which every config in this project uses.
        batch = seqlens.numel()
        length = int(seqlens[0].item())

        def _to_batched(x: torch.Tensor) -> torch.Tensor:
            return x.reshape(batch, length, num_heads, head_dim).permute(0, 2, 1, 3)

        out = F.scaled_dot_product_attention(
            _to_batched(q),
            _to_batched(k),
            _to_batched(v),
            dropout_p=dropout_p,
            scale=softmax_scale,
        )  # [B, H, L, D]
        return out.permute(0, 2, 1, 3).reshape(total_n, num_heads, head_dim)

    # General path: irregular segment lengths, looped. Rare in this project
    # (only possible for global attention with a mixed-size batch).
    outputs = []
    starts = cu_seqlens[:-1].tolist()
    lengths = seqlens.tolist()
    for start, length in zip(starts, lengths):
        qs = q[start : start + length].transpose(0, 1).unsqueeze(0)
        ks = k[start : start + length].transpose(0, 1).unsqueeze(0)
        vs = v[start : start + length].transpose(0, 1).unsqueeze(0)
        out = F.scaled_dot_product_attention(
            qs, ks, vs, dropout_p=dropout_p, scale=softmax_scale
        )
        outputs.append(out.squeeze(0).transpose(0, 1))
    return torch.cat(outputs, dim=0)


def flash_attn_varlen_qkvpacked_func(
    qkv: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
    dropout_p: float = 0.0,
    softmax_scale: Optional[float] = None,
) -> torch.Tensor:
    """Drop-in for flash_attn.flash_attn_varlen_qkvpacked_func.

    Dispatches to the real CUDA kernel when flash_attn is importable, else
    the pure-PyTorch fallback above. Same signature and return shape either
    way, so call sites do not need to branch.
    """
    if HAS_FLASH_ATTN:
        return _flash_attn.flash_attn_varlen_qkvpacked_func(
            qkv,
            cu_seqlens,
            max_seqlen=max_seqlen,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
        )
    return _sdpa_varlen_fallback(
        qkv, cu_seqlens, max_seqlen, dropout_p=dropout_p, softmax_scale=softmax_scale
    )
