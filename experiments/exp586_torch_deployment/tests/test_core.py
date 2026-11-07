import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from exp586_deployment import providers
from exp586_deployment.benchmark import (
    fixture_batch,
    latency_summary,
    validation_batches,
)
from exp586_deployment.model import short_conv
from exp586_deployment.providers import RuntimeConfig


@pytest.mark.parametrize("length", [1, 2, 8, 13])
def test_short_conv_is_causal_and_matches_depthwise_conv(length: int) -> None:
    torch.manual_seed(586)
    x = torch.randn(2, length, 5)
    weight = torch.randn(4, 5)
    expected = F.conv1d(
        F.pad(x.transpose(1, 2), (3, 0)),
        weight.transpose(0, 1).flip(-1).unsqueeze(1),
        groups=5,
    ).transpose(1, 2)
    torch.testing.assert_close(short_conv(x, weight), expected)
    altered = x.clone()
    altered[:, -1] += 100
    torch.testing.assert_close(
        short_conv(x, weight)[:, :-1], short_conv(altered, weight)[:, :-1]
    )


def test_core_import_does_not_load_optional_providers_or_training() -> None:
    code = """import json,sys
import exp586_deployment.model
for prefix in ['vllm','flash_attn','flash_attn_3','jax','marin','levanter','wandb','exp586_moe']:
 assert not any(n==prefix or n.startswith(prefix+'.') for n in sys.modules),prefix
print('isolated')
"""
    assert (
        subprocess.check_output([sys.executable, "-c", code], text=True).strip()
        == "isolated"
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"attention_provider": "auto"},
        {"flash_attention_version": 4},
        {"moe_backend": "naive"},
    ],
)
def test_invalid_configuration_is_rejected(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        RuntimeConfig(**kwargs)


@pytest.mark.parametrize(
    "provider,version", [("standalone", 2), ("standalone", 3), ("vllm", 2), ("vllm", 3)]
)
def test_attention_adapter_preserves_gqa_and_window_contract(
    monkeypatch: pytest.MonkeyPatch, provider: str, version: int
) -> None:
    calls = []

    def attention(*args: object, **kwargs: object) -> torch.Tensor:
        calls.append((args, kwargs))
        return args[0].clone()

    monkeypatch.setattr(
        providers,
        "optional_module",
        lambda name: SimpleNamespace(flash_attn_varlen_func=attention),
    )
    operation = providers.load_attention(provider, version)
    q, k, v = (
        torch.randn(6, 8, 4).transpose(0, 1),
        torch.randn(8, 1, 4),
        torch.randn(8, 1, 4),
    )
    cu = torch.tensor([0, 4, 8], dtype=torch.int32)
    for left in [-1, 3]:
        result = operation(q, k, v, cu, 4, left)
        torch.testing.assert_close(result, q)
        assert result.is_contiguous()
        args, kwargs = calls[-1]
        assert args[1].shape[1] == 1
        assert kwargs["causal"] is True
        assert kwargs["window_size"] == (
            (None if left < 0 else [3, 0])
            if provider == "vllm"
            else ((-1, -1) if left < 0 else (3, 0))
        )
        if provider == "vllm":
            assert kwargs["fa_version"] == version
            assert args[3:] == (4, cu, 4, cu)
        else:
            assert args[3] is cu and args[4] is cu
            assert args[5:] == (4, 4)
        torch.library.opcheck(
            operation,
            (q, k, v, cu, 4, left),
            test_utils=("test_schema", "test_faketensor"),
        )


def test_forward_counts_full_windows_and_validation_has_one_shape() -> None:
    assert latency_summary([500, 500], 64)["forward_tokens_per_second"] == 1048576
    assert fixture_batch(["a", "b", "c"], 2, 2) == ["c", "a"]
    batches = list(validation_batches(["a", "b", "c"], ["a", "b", "c", "d"], 2))
    assert all(len(batch) == 2 for batch, n in batches)
    assert [row for batch, n in batches for row in batch[:n]] == ["a", "b", "c"]
