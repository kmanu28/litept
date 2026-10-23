"""CPU checks for the isolated RelFlexformer attention prototype."""

import torch

from litept.relflexformer_attention import (
    RelFlexformerAttention,
    dense_reference_attention,
    relflexformer_attention,
    sample_fixed_frequencies,
)


def make_inputs(dtype=torch.float64):
    torch.manual_seed(7)
    query = torch.randn(9, 2, 6, dtype=dtype, requires_grad=True)
    key = torch.randn(9, 2, 6, dtype=dtype, requires_grad=True)
    value = torch.randn(9, 2, 6, dtype=dtype, requires_grad=True)
    coord = 0.05 * torch.randn(9, 3, dtype=dtype)
    offset = torch.tensor([4, 9], dtype=torch.long)
    frequencies = sample_fixed_frequencies(
        num_heads=2,
        num_frequencies=8,
        seed=11,
        dtype=dtype,
    )
    weights = torch.full((2, 8), 1.0 / 8, dtype=dtype)
    return query, key, value, coord, offset, frequencies, weights


def test_fast_matches_dense():
    query, key, value, coord, offset, frequencies, weights = make_inputs()
    fast = relflexformer_attention(
        query, key, value, coord, offset, frequencies, weights
    )
    dense = dense_reference_attention(
        query, key, value, coord, offset, frequencies, weights
    )
    torch.testing.assert_close(fast, dense, rtol=1e-10, atol=1e-10)
    return (fast - dense).abs().max().item()


def test_gradients_match_dense():
    fast_inputs = make_inputs()
    query, key, value, coord, offset, frequencies, weights = fast_inputs
    fast_loss = relflexformer_attention(
        query, key, value, coord, offset, frequencies, weights
    ).square().mean()
    fast_gradients = torch.autograd.grad(fast_loss, (query, key, value))

    dense_inputs = make_inputs()
    query, key, value, coord, offset, frequencies, weights = dense_inputs
    dense_loss = dense_reference_attention(
        query, key, value, coord, offset, frequencies, weights
    ).square().mean()
    dense_gradients = torch.autograd.grad(dense_loss, (query, key, value))
    max_error = 0.0
    for fast_gradient, dense_gradient in zip(fast_gradients, dense_gradients):
        torch.testing.assert_close(
            fast_gradient, dense_gradient, rtol=1e-9, atol=1e-9
        )
        max_error = max(
            max_error, (fast_gradient - dense_gradient).abs().max().item()
        )
    return max_error


def test_clouds_are_isolated():
    query, key, value, coord, offset, frequencies, weights = make_inputs()
    batched = relflexformer_attention(
        query, key, value, coord, offset, frequencies, weights
    )
    first = relflexformer_attention(
        query[:4],
        key[:4],
        value[:4],
        coord[:4],
        torch.tensor([4]),
        frequencies,
        weights,
    )
    second = relflexformer_attention(
        query[4:],
        key[4:],
        value[4:],
        coord[4:],
        torch.tensor([5]),
        frequencies,
        weights,
    )
    separate = torch.cat((first, second))
    torch.testing.assert_close(batched, separate, rtol=1e-10, atol=1e-10)
    return (batched - separate).abs().max().item()


def test_zero_denominator_is_finite():
    module = RelFlexformerAttention(num_heads=2, seed=3).double()
    query = torch.full(
        (5, 2, 6), -1.0, dtype=torch.float64, requires_grad=True
    )
    key = torch.randn(5, 2, 6, dtype=torch.float64, requires_grad=True)
    value = torch.randn(5, 2, 6, dtype=torch.float64, requires_grad=True)
    coord = torch.randn(5, 3, dtype=torch.float64)
    output = module(query, key, value, coord, torch.tensor([5]))
    assert torch.isfinite(output).all()
    assert torch.count_nonzero(output) == 0
    output.sum().backward()
    assert query.grad is not None
    return output.abs().max().item()


def test_wrapper_adds_no_state_dict_entries():
    module = RelFlexformerAttention(num_heads=2, seed=3)
    assert not list(module.parameters())
    assert not module.state_dict()
    return len(module.state_dict())


def test_cuda_autocast_uses_fp32_math():
    if not torch.cuda.is_available():
        return float("nan")
    torch.manual_seed(13)
    module = RelFlexformerAttention(num_heads=2, seed=3).cuda()
    query = torch.randn(256, 2, 16, device="cuda").half().requires_grad_()
    key = torch.randn(256, 2, 16, device="cuda").half().requires_grad_()
    value = torch.randn(256, 2, 16, device="cuda").half().requires_grad_()
    coord = torch.randn(256, 3, device="cuda")
    offset = torch.tensor([128, 256], device="cuda")

    reference = module(
        query.float(), key.float(), value.float(), coord, offset
    )
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        output = module(query, key, value, coord, offset)
    assert output.dtype == torch.float32
    assert torch.isfinite(output).all()
    torch.testing.assert_close(output, reference, rtol=0, atol=0)
    output.square().mean().backward()
    for tensor in (query, key, value):
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all()
    return (output - reference).abs().max().item()


if __name__ == "__main__":
    checks = (
        ("forward max error", test_fast_matches_dense),
        ("gradient max error", test_gradients_match_dense),
        ("batch-isolation max error", test_clouds_are_isolated),
        ("zero-denominator max magnitude", test_zero_denominator_is_finite),
        ("state-dict entries", test_wrapper_adds_no_state_dict_entries),
        ("CUDA autocast-vs-FP32 max error", test_cuda_autocast_uses_fp32_math),
    )
    for name, check in checks:
        result = check()
        print(f"PASS: {name}: {result:.3e}")
