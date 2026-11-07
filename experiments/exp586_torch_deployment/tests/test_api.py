"""Exercise selectable model implementations without optional GPU libraries."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from exp586_deployment import model
from exp586_deployment.model import MarinDNAMoE, depthwise_short_conv, short_conv
from exp586_deployment.tokenizer import dna_nucleotide_tokenizer


@pytest.fixture
def checkpoint(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    dna_nucleotide_tokenizer().save(str(tmp_path / "tokenizer.json"))
    monkeypatch.setattr(
        model.TorchConfig,
        "load",
        lambda _: SimpleNamespace(num_layers=0, head_dim=8, rope_theta=10000),
    )
    monkeypatch.setattr(model, "load_attention", lambda *_: None)
    monkeypatch.setattr(
        model,
        "load_weights",
        lambda *_: {
            "model.embed_tokens.weight": torch.arange(24).reshape(8, 3).float(),
            "lm_head.weight": torch.arange(24).reshape(8, 3).float() + 100,
        },
    )
    return tmp_path


def test_retained_vocabularies_preserve_selected_rows_and_order(
    checkpoint: Path,
) -> None:
    module = MarinDNAMoE(checkpoint, "cpu", input_vocab="NTAC", output_vocab="TAN")
    embedding = module._weight("model.embed_tokens.weight")
    head = module._weight("lm_head.weight")
    assert embedding.dtype == head.dtype == torch.bfloat16
    assert embedding.tolist() == [[12, 13, 14], [9, 10, 11], [0, 1, 2], [3, 4, 5]]
    assert head.tolist() == [[109, 110, 111], [100, 101, 102], [112, 113, 114]]


def test_full_vocab_and_selected_dtype_are_independent_options(
    checkpoint: Path,
) -> None:
    module = MarinDNAMoE(
        checkpoint, "cpu", input_vocab=None, output_vocab=None, dtype=torch.float16
    )
    assert module._weight("model.embed_tokens.weight").shape == (8, 3)
    assert module._weight("lm_head.weight").shape == (8, 3)
    assert module._weight("lm_head.weight").dtype == torch.float16


@pytest.mark.parametrize(
    "options",
    [
        {"input_vocab": "AACGT"},
        {"input_vocab": "acgt"},
        {"output_vocab": ""},
        {"output_vocab": "XYZ"},
        {"dtype": torch.float32},
        {"sconv_backend": "custom"},
        {"cache_rope": "yes"},
    ],
)
def test_invalid_options_fail_before_checkpoint_io(options: dict[str, object]) -> None:
    with pytest.raises((ValueError, TypeError)):
        MarinDNAMoE("/does/not/exist", "cpu", **options)


def test_cached_and_computed_rotary_tables_agree(checkpoint: Path) -> None:
    cached = MarinDNAMoE(checkpoint, "cpu", cache_rope=True)
    computed = MarinDNAMoE(checkpoint, "cpu", cache_rope=False)
    assert "rope_cos" in dict(cached.named_buffers())
    assert "rope_cos" not in dict(computed.named_buffers())
    x = torch.randn(2, 3, 7, 8)
    torch.testing.assert_close(
        cached._precomputed_half_rope(x),
        computed._precomputed_half_rope(x),
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize("length", [1, 2, 8, 13])
def test_selectable_convolution_preserves_lag_order(length: int) -> None:
    x, weight = torch.randn(2, length, 5), torch.randn(4, 5)
    torch.testing.assert_close(depthwise_short_conv(x, weight), short_conv(x, weight))


def test_attention_and_expert_providers_are_independent(
    checkpoint: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sentinel = object()
    monkeypatch.setattr(model, "load_fused_experts", lambda: sentinel)
    module = MarinDNAMoE(
        checkpoint, "cpu", attention_provider="flash_attn", moe_backend="vllm"
    )
    assert module.runtime.attention_provider == "flash_attn"
    assert module.runtime.moe_backend == "vllm"
    assert module.fused_experts is sentinel
