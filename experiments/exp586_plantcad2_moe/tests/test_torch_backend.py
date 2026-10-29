import pytest

torch = pytest.importorskip("torch")

from exp586_moe.inference.torch_backend import (
    fused_half_rope,
    grouped_experts,
    rms_norm,
    short_conv,
)


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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="grouped GEMM requires CUDA")
def test_grouped_experts_matches_assignment_order() -> None:
    device = torch.device("cuda:0")
    routed = torch.arange(32, device=device, dtype=torch.bfloat16).reshape(4, 8) / 32
    selected = torch.tensor([[0, 2], [1, 0], [2, 1], [0, 2]], device=device)
    combine = torch.full((4, 2), 0.5, device=device)
    generator = torch.Generator(device=device).manual_seed(586)
    gate = torch.randn(
        3, 6, 8, device=device, dtype=torch.bfloat16, generator=generator
    )
    up = torch.randn(3, 6, 8, device=device, dtype=torch.bfloat16, generator=generator)
    down = torch.randn(
        3, 8, 6, device=device, dtype=torch.bfloat16, generator=generator
    )

    expected = torch.zeros_like(routed)
    for expert in range(3):
        rows, slots = torch.where(selected == expert)
        hidden = torch.nn.functional.silu(routed[rows] @ gate[expert].T) * (
            routed[rows] @ up[expert].T
        )
        expected.index_add_(
            0, rows, (hidden @ down[expert].T) * combine[rows, slots, None]
        )
    actual = grouped_experts(routed, selected, combine, gate, up, down)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
