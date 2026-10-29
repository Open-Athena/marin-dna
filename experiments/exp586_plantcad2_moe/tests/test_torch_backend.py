import pytest

torch = pytest.importorskip("torch")

from exp586_moe.inference.torch_backend import fused_half_rope, rms_norm, short_conv


def test_short_conv_uses_lag_zero_first() -> None:
    values = torch.tensor([[[1.0], [2.0], [3.0], [4.0]]])
    weights = torch.tensor([[2.0], [3.0]])
    actual = short_conv(values, weights)
    assert torch.equal(actual, torch.tensor([[[2.0], [7.0], [12.0], [17.0]]]))


def test_rms_norm_preserves_input_dtype() -> None:
    values = torch.tensor([[1.0, 2.0]], dtype=torch.bfloat16)
    actual = rms_norm(values, torch.ones(2), 1e-5)
    assert actual.dtype == torch.bfloat16


def test_zero_position_rope_is_identity() -> None:
    values = torch.arange(16, dtype=torch.float32).reshape(1, 1, 2, 8)
    actual = fused_half_rope(values, 10_000)
    assert torch.equal(actual[..., 0, :], values[..., 0, :])
