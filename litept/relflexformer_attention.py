"""Dependency-free RelFlexformer-style attention prototype.

This module intentionally has no dependency on LitePT.  It implements the
finite-quadrature operator described by Algorithms 1 and 2 of the
RelFlexformer paper using direct trigonometric evaluation.  With a constant
number of frequencies this is O(N * S), and does not require a NUFFT package.

The paper specifies fixed normally distributed frequencies, ReLU Performer
features, and exp(-lambda * ||frequency||) Fourier modulation.  It does not
specify the quadrature coefficients used by the released experiments, so this
prototype exposes them and defaults to uniform coefficients.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def sample_fixed_frequencies(
    num_heads: int,
    num_frequencies: int = 8,
    coord_dim: int = 3,
    seed: int = 0,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Sample reproducible per-head frequencies from a standard normal."""
    if num_heads <= 0 or num_frequencies <= 0 or coord_dim <= 0:
        raise ValueError("frequency dimensions must all be positive")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return torch.randn(
        num_heads,
        num_frequencies,
        coord_dim,
        generator=generator,
        dtype=dtype,
    )


def _validate_inputs(
    values: torch.Tensor,
    coord: torch.Tensor,
    offset: torch.Tensor,
    frequencies: torch.Tensor,
    quadrature_weights: torch.Tensor,
) -> None:
    if values.ndim != 3:
        raise ValueError("values must have shape [N, H, C]")
    if coord.ndim != 2:
        raise ValueError("coord must have shape [N, R]")
    if offset.ndim != 1 or offset.numel() == 0:
        raise ValueError("offset must be a non-empty 1D tensor")
    if frequencies.ndim != 3:
        raise ValueError("frequencies must have shape [H, S, R]")
    if quadrature_weights.shape != frequencies.shape[:2]:
        raise ValueError("quadrature_weights must have shape [H, S]")
    if values.shape[0] != coord.shape[0]:
        raise ValueError("values and coord must contain the same number of tokens")
    if values.shape[1] != frequencies.shape[0]:
        raise ValueError("values and frequencies must contain the same number of heads")
    if coord.shape[1] != frequencies.shape[2]:
        raise ValueError("coord and frequencies must have the same coordinate dimension")
    if int(offset[-1]) != values.shape[0]:
        raise ValueError("the final offset must equal the token count")
    if bool(torch.any(offset[1:] <= offset[:-1])) or int(offset[0]) <= 0:
        raise ValueError("offset entries must be strictly increasing and positive")


def fourier_mask_multiply(
    values: torch.Tensor,
    coord: torch.Tensor,
    offset: torch.Tensor,
    frequencies: torch.Tensor,
    quadrature_weights: torch.Tensor,
    modulation_lambda: float,
) -> torch.Tensor:
    """Apply the finite Fourier mask independently to each point cloud.

    Args:
        values: Signal with shape [N, H, C].
        coord: Continuous coordinates with shape [N, R].
        offset: Cumulative cloud sizes with shape [B].
        frequencies: Fixed frequency vectors with shape [H, S, R].
        quadrature_weights: Coefficients ``a_s`` with shape [H, S].
        modulation_lambda: Lambda in exp(-lambda * ||frequency||).
    """
    _validate_inputs(values, coord, offset, frequencies, quadrature_weights)
    if modulation_lambda < 0:
        raise ValueError("modulation_lambda must be non-negative")

    compute_dtype = (
        torch.float32
        if values.dtype in (torch.float16, torch.bfloat16)
        else values.dtype
    )
    coord = coord.to(dtype=compute_dtype)
    frequencies = frequencies.to(device=values.device, dtype=compute_dtype)
    weights = quadrature_weights.to(device=values.device, dtype=compute_dtype)
    spectral_modulation = torch.exp(
        -modulation_lambda * torch.linalg.vector_norm(frequencies, dim=-1)
    )
    coefficients = weights * spectral_modulation

    output = torch.empty_like(values, dtype=compute_dtype)
    start = 0
    for end_tensor in offset:
        end = int(end_tensor)
        cloud_coord = coord[start:end]
        cloud_values = values[start:end].to(dtype=compute_dtype)

        phase = 2 * math.pi * torch.einsum(
            "nr,hsr->nhs", cloud_coord, frequencies
        )
        cosine = torch.cos(phase)
        sine = torch.sin(phase)
        cosine_spectrum = torch.einsum("nhs,nhc->hsc", cosine, cloud_values)
        sine_spectrum = torch.einsum("nhs,nhc->hsc", sine, cloud_values)
        output[start:end] = torch.einsum(
            "nhs,hs,hsc->nhc", cosine, coefficients, cosine_spectrum
        ) + torch.einsum(
            "nhs,hs,hsc->nhc", sine, coefficients, sine_spectrum
        )
        start = end
    return output


def finite_quadrature_mask(
    coord: torch.Tensor,
    frequencies: torch.Tensor,
    quadrature_weights: torch.Tensor,
    modulation_lambda: float,
) -> torch.Tensor:
    """Materialize the finite-quadrature mask for a tiny reference problem."""
    compute_dtype = coord.dtype
    frequencies = frequencies.to(device=coord.device, dtype=compute_dtype)
    weights = quadrature_weights.to(device=coord.device, dtype=compute_dtype)
    coefficients = weights * torch.exp(
        -modulation_lambda * torch.linalg.vector_norm(frequencies, dim=-1)
    )
    displacement = coord[:, None, :] - coord[None, :, :]
    phase = 2 * math.pi * torch.einsum(
        "ijr,hsr->hijs", displacement, frequencies
    )
    return torch.einsum("hijs,hs->hij", torch.cos(phase), coefficients)


def _stabilize_denominator(
    denominator: torch.Tensor,
    eps: float,
    reference: torch.Tensor | None = None,
) -> torch.Tensor:
    """Keep the row-sum sign while guarding small relative magnitudes."""
    if eps <= 0:
        raise ValueError("eps must be positive")
    floor = denominator.new_tensor(eps)
    if reference is not None:
        floor = eps * reference.abs().clamp_min(1.0)
    sign = torch.where(denominator < 0, -1, 1)
    return torch.where(
        denominator.abs() < floor,
        sign * floor,
        denominator,
    )


def dense_reference_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    coord: torch.Tensor,
    offset: torch.Tensor,
    frequencies: torch.Tensor,
    quadrature_weights: torch.Tensor,
    modulation_lambda: float = 1.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Slow O(N^2) reference for the same finite-quadrature operator."""
    if query.shape != key.shape or query.shape != value.shape:
        raise ValueError("query, key, and value must have the same [N, H, D] shape")
    outputs = []
    start = 0
    for end_tensor in offset:
        end = int(end_tensor)
        q = torch.relu(query[start:end])
        k = torch.relu(key[start:end])
        v = value[start:end]
        mask = finite_quadrature_mask(
            coord[start:end],
            frequencies,
            quadrature_weights,
            modulation_lambda,
        )
        base_scores = torch.einsum("nhd,mhd->hnm", q, k)
        scores = base_scores * mask
        denominator = _stabilize_denominator(
            scores.sum(dim=-1),
            eps,
            reference=base_scores.sum(dim=-1),
        )
        result = torch.einsum("hnm,mhd->nhd", scores, v)
        outputs.append(result / denominator.transpose(0, 1).unsqueeze(-1))
        start = end
    return torch.cat(outputs, dim=0)


def relflexformer_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    coord: torch.Tensor,
    offset: torch.Tensor,
    frequencies: torch.Tensor,
    quadrature_weights: torch.Tensor,
    modulation_lambda: float = 1.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """RelFlexformer masked ReLU attention without an explicit N x N matrix."""
    if query.shape != key.shape or query.shape != value.shape:
        raise ValueError("query, key, and value must have the same [N, H, D] shape")
    if query.ndim != 3:
        raise ValueError("query, key, and value must have shape [N, H, D]")

    compute_dtype = (
        torch.float32
        if query.dtype in (torch.float16, torch.bfloat16)
        else query.dtype
    )
    query_features = torch.relu(query.to(dtype=compute_dtype))
    key_features = torch.relu(key.to(dtype=compute_dtype))
    value = value.to(dtype=compute_dtype)
    key_value = key_features.unsqueeze(-1) * value.unsqueeze(-2)
    token_count, num_heads, feature_dim, value_dim = key_value.shape

    masked_key_value = fourier_mask_multiply(
        key_value.reshape(token_count, num_heads, feature_dim * value_dim),
        coord,
        offset,
        frequencies,
        quadrature_weights,
        modulation_lambda,
    ).reshape(token_count, num_heads, feature_dim, value_dim)
    masked_key = fourier_mask_multiply(
        key_features,
        coord,
        offset,
        frequencies,
        quadrature_weights,
        modulation_lambda,
    )

    numerator = torch.einsum("nhm,nhmd->nhd", query_features, masked_key_value)
    reference_denominator = torch.empty(
        token_count,
        num_heads,
        device=query.device,
        dtype=compute_dtype,
    )
    start = 0
    for end_tensor in offset:
        end = int(end_tensor)
        key_sum = key_features[start:end].sum(dim=0)
        reference_denominator[start:end] = torch.einsum(
            "nhm,hm->nh",
            query_features[start:end],
            key_sum,
        )
        start = end
    denominator = _stabilize_denominator(
        torch.einsum("nhm,nhm->nh", query_features, masked_key),
        eps,
        reference=reference_denominator,
    )
    return numerator / denominator.unsqueeze(-1)


class RelFlexformerAttention(nn.Module):
    """Stateful wrapper holding fixed frequencies as non-parameter buffers."""

    def __init__(
        self,
        num_heads: int,
        num_frequencies: int = 8,
        coord_dim: int = 3,
        modulation_lambda: float = 1.0,
        coord_scale: float = 1.0,
        eps: float = 1e-2,
        seed: int = 0,
    ) -> None:
        super().__init__()
        frequencies = sample_fixed_frequencies(
            num_heads=num_heads,
            num_frequencies=num_frequencies,
            coord_dim=coord_dim,
            seed=seed,
        )
        self.register_buffer("frequencies", frequencies, persistent=False)
        self.register_buffer(
            "quadrature_weights",
            torch.full((num_heads, num_frequencies), 1.0 / num_frequencies),
            persistent=False,
        )
        self.modulation_lambda = modulation_lambda
        if coord_scale <= 0:
            raise ValueError("coord_scale must be positive")
        self.coord_scale = coord_scale
        self.eps = eps

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        coord: torch.Tensor,
        offset: torch.Tensor,
    ) -> torch.Tensor:
        # RelFlex contains reductions over every token in a cloud.  CUDA
        # autocast would otherwise run its einsums in float16 even though the
        # operands are promoted in relflexformer_attention, which can overflow
        # during training.  Keep only this attention operator in full precision;
        # the caller converts its output back to the model activation dtype.
        with torch.autocast(device_type=query.device.type, enabled=False):
            return relflexformer_attention(
                query=query,
                key=key,
                value=value,
                coord=coord * self.coord_scale,
                offset=offset,
                frequencies=self.frequencies,
                quadrature_weights=self.quadrature_weights,
                modulation_lambda=self.modulation_lambda,
                eps=self.eps,
            )
