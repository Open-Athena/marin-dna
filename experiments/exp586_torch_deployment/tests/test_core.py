import subprocess
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from exp586_deployment import providers
from exp586_deployment.benchmark import (
    compiler_kwargs,
    fixture_batch,
    latency_summary,
    measurement_schedule,
    validation_batches,
)
from exp586_deployment.fast import combined_shared_experts
from exp586_deployment.model import MarinDNAMoE, short_conv
from exp586_deployment.providers import RuntimeConfig
from exp586_deployment.tokenizer import dna_nucleotide_tokenizer, encode_dna_batch


def test_nucleotide_tokenizer_preserves_single_base_ids_and_case() -> None:
    sequence = "AcGtN" * 1638 + "aC"
    tokenizer = dna_nucleotide_tokenizer()
    encoded = encode_dna_batch(tokenizer, [sequence.upper()])
    assert encoded.shape == (1, 8192)
    assert encoded.tolist() == [[0, 1, 2, 3, 4] * 1638 + [0, 1]]
    assert (
        tokenizer.encode(sequence, add_special_tokens=False).ids == encoded[0].tolist()
    )


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
from exp586_deployment import MarinDNAMoE, FastMarinDNAMoE
assert 'exp586_deployment.router_topk' not in sys.modules
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
    "provider,version", [("flash_attn", 2), ("flash_attn", 3), ("vllm", 2), ("vllm", 3)]
)
def test_attention_adapter_preserves_gqa_and_window_contract(
    monkeypatch: pytest.MonkeyPatch, provider: str, version: int
) -> None:
    calls = []
    imports = []

    def attention(*args: object, **kwargs: object) -> torch.Tensor:
        calls.append((args, kwargs))
        return args[0].clone()

    def optional_module(name: str) -> SimpleNamespace:
        imports.append(name)
        return SimpleNamespace(flash_attn_varlen_func=attention)

    monkeypatch.setattr(providers, "optional_module", optional_module)
    operation = providers.load_attention(provider, version)
    assert imports == [
        "vllm.vllm_flash_attn"
        if provider == "vllm"
        else "flash_attn"
        if version == 2
        else "flash_attn_interface"
    ]
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


def test_compile_defaults_keep_only_the_fullgraph_assertion() -> None:
    assert compiler_kwargs("default", "auto") == {"fullgraph": True}
    assert compiler_kwargs("reduce-overhead", "auto") == {
        "fullgraph": True,
        "mode": "reduce-overhead",
    }
    assert compiler_kwargs("default", "static") == {
        "fullgraph": True,
        "dynamic": False,
    }
    assert compiler_kwargs("reduce-overhead", "static") == {
        "fullgraph": True,
        "dynamic": False,
        "mode": "reduce-overhead",
    }


@pytest.mark.parametrize("batch", [8, 12, 16, 32, 64, 96, 128, 240, 256])
def test_timing_weights_every_real_window_equally(batch: int) -> None:
    fixture = [str(i) for i in range(512)]
    batches, iterations = measurement_schedule(len(fixture), batch, 32)
    inputs = [fixture_batch(fixture, batch, i * batch) for i in range(batches)]
    counts = Counter(
        window for i in range(iterations) for window in inputs[i % batches]
    )
    assert len(counts) == len(fixture)
    assert len(set(counts.values())) == 1
    assert iterations >= 32
    assert all(len(set(rows)) == batch for rows in inputs)
    if batch in {16, 32, 64, 128, 256}:
        assert iterations == 32


def test_hf_fetch_preserves_revision_and_explicit_model_settings(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fetched, constructed = [], []

    def fetch(repository: str, **kwargs: object) -> str:
        fetched.append((repository, kwargs))
        return str(tmp_path)

    def initialize(self: MarinDNAMoE, *args: object, **kwargs: object) -> None:
        torch.nn.Module.__init__(self)
        constructed.append((args, kwargs))

    monkeypatch.setattr("huggingface_hub.snapshot_download", fetch)
    monkeypatch.setattr(MarinDNAMoE, "__init__", initialize)
    model = MarinDNAMoE.from_pretrained(
        "plantcad/example",
        revision="immutable-revision",
        device="cpu",
        attention_provider="vllm",
        moe_backend="vllm",
        flash_attention_version=3,
    )
    assert isinstance(model, MarinDNAMoE)
    assert fetched[0][0] == "plantcad/example"
    assert fetched[0][1]["revision"] == "immutable-revision"
    assert "*.safetensors" in fetched[0][1]["allow_patterns"]
    assert constructed[0][0] == (tmp_path, "cpu")
    assert constructed[0][1]["attention_provider"] == "vllm"
    assert constructed[0][1]["moe_backend"] == "vllm"
    assert constructed[0][1]["flash_attention_version"] == 3
    assert constructed[0][1]["dtype"] == torch.bfloat16
    assert constructed[0][1]["input_vocab"] == "ACGTN"
    assert constructed[0][1]["output_vocab"] == "ACGT"
    MarinDNAMoE.from_pretrained(tmp_path, device="cpu")
    assert len(fetched) == 1


def test_missing_vllm_fails_before_checkpoint_load(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from exp586_deployment import model

    monkeypatch.setattr(model.TorchConfig, "load", lambda _: None)

    def missing(*args: object) -> None:
        raise RuntimeError("Selected provider requires importable vllm")

    monkeypatch.setattr(model, "load_attention", missing)
    with pytest.raises(RuntimeError, match="requires importable vllm"):
        MarinDNAMoE(tmp_path, "cpu", attention_provider="vllm", moe_backend="vllm")


@pytest.mark.parametrize("experts", [1, 2, 4])
def test_fast_shared_projection_preserves_sum_of_separate_experts(experts: int) -> None:
    torch.manual_seed(586)
    x = torch.randn(2, 3, 8)
    gate, up = torch.randn(experts, 5, 8), torch.randn(experts, 5, 8)
    down = torch.randn(experts, 8, 5)
    expected = sum(
        F.linear(F.silu(F.linear(x, gate[i])) * F.linear(x, up[i]), down[i])
        for i in range(experts)
    )
    actual = combined_shared_experts(
        x,
        torch.cat([weight for i in range(experts) for weight in (gate[i], up[i])]),
        torch.cat(list(down), dim=1),
        experts,
    )
    torch.testing.assert_close(actual, expected)
