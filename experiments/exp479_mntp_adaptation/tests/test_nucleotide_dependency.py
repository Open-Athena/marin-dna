from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

import exp479_mntp.nucleotide_dependency as dependency
from exp479_mntp.nucleotide_dependency import (
    Locus,
    mean_symmetrize,
    off_diagonal_spearman,
    orientation_dependency,
    plot_comparison,
)
from exp479_mntp.vep import LoadedArm


def test_mean_symmetrization_and_off_diagonal_correlation(tmp_path: Path) -> None:
    directed = np.zeros((255, 255), dtype=np.float32)
    directed[10, 20] = 2
    directed[20, 10] = 4
    symmetric = mean_symmetrize(directed)
    assert symmetric[10, 20] == symmetric[20, 10] == 3
    assert np.diag(symmetric).sum() == 0
    assert np.isclose(off_diagonal_spearman(symmetric, symmetric * 2), 1.0)

    output = tmp_path / "comparison.svg"
    plot_comparison(
        symmetric,
        symmetric * 2,
        locus=Locus("test", "1", 100, 200, "+"),
        correlation=1.0,
        output_path=output,
    )
    assert output.read_text(encoding="utf-8").startswith("<?xml")


class _Tokenizer:
    def __call__(
        self,
        sequence: str,
        *,
        add_special_tokens: bool,
        return_tensors: str,
    ) -> dict[str, torch.Tensor]:
        assert add_special_tokens
        assert return_tensors == "pt"
        lookup = {"A": 3, "C": 4, "G": 5, "T": 6}
        return {"input_ids": torch.tensor([[2, *(lookup[base] for base in sequence)]])}


def test_paired_baseline_causal_map_has_no_right_context_leakage(
    monkeypatch: object,
) -> None:
    monkeypatch.setattr(dependency, "NUCLEOTIDE_LENGTH", 5)  # type: ignore[attr-defined]
    monkeypatch.setattr(  # type: ignore[attr-defined]
        dependency,
        "assert_budget_reserve",
        lambda: None,
    )
    torch.manual_seed(479)
    config = Qwen3Config(
        vocab_size=8,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=8,
        max_position_embeddings=16,
        use_cache=False,
    )
    config._attn_implementation = "eager"
    arm = LoadedArm(
        model=Qwen3ForCausalLM(config).eval(),
        tokenizer=_Tokenizer(),
        canonical_ids=(3, 4, 5, 6),
        mask_token_id=7,
    )
    original_model_logits = dependency.model_logits
    observed_batch_sizes: list[int] = []

    def traced_model_logits(*args: object, **kwargs: object) -> torch.Tensor:
        input_ids = kwargs["input_ids"]
        assert isinstance(input_ids, torch.Tensor)
        observed_batch_sizes.append(len(input_ids))
        return original_model_logits(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(dependency, "model_logits", traced_model_logits)  # type: ignore[attr-defined]
    causal = orientation_dependency(
        arm,
        "ACGTA",
        batch_size=20,
        attention_mode="causal",
    )
    full = orientation_dependency(
        arm,
        "ACGTA",
        batch_size=20,
        attention_mode="full",
    )
    assert float(np.tril(causal, k=-1).max()) == 0.0
    assert float(np.tril(full, k=-1).max()) > 0.0
    assert observed_batch_sizes
    assert min(observed_batch_sizes) >= 2
    assert max(observed_batch_sizes) <= 20


def test_causal_dependency_does_not_require_a_mask_token(monkeypatch: object) -> None:
    monkeypatch.setattr(dependency, "NUCLEOTIDE_LENGTH", 5)  # type: ignore[attr-defined]
    monkeypatch.setattr(  # type: ignore[attr-defined]
        dependency,
        "assert_budget_reserve",
        lambda: None,
    )
    torch.manual_seed(479)
    config = Qwen3Config(
        vocab_size=7,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=8,
        max_position_embeddings=16,
        use_cache=False,
    )
    config._attn_implementation = "eager"
    arm = LoadedArm(
        model=Qwen3ForCausalLM(config).eval(),
        tokenizer=_Tokenizer(),
        canonical_ids=(3, 4, 5, 6),
        mask_token_id=None,
    )
    causal = orientation_dependency(
        arm,
        "ACGTA",
        batch_size=21,
        attention_mode="causal",
    )
    assert np.isfinite(causal).all()
    assert float(np.tril(causal, k=-1).max()) == 0.0
