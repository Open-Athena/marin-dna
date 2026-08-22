from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import torch
from Bio import bgzf
from pyfaidx import Fasta
from transformers import Qwen3Config, Qwen3ForCausalLM

from exp479_mntp.vep import (
    DatasetSpec,
    LoadedArm,
    assert_development_split,
    attach_reference_windows,
    load_variant_frame,
    reverse_complement,
    score_strand,
)


class CharacterTokenizer:
    mask_token_id = 7

    def __call__(
        self,
        sequences: str | list[str],
        *,
        return_tensors: str,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        del kwargs
        assert return_tensors == "pt"
        if isinstance(sequences, str):
            sequences = [sequences]
        lookup = {"N": 1, "A": 3, "C": 4, "G": 5, "T": 6}
        input_ids = torch.tensor([[2, *(lookup[base] for base in seq)] for seq in sequences])
        return {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)}


def _arm() -> LoadedArm:
    torch.manual_seed(479)
    config = Qwen3Config(
        vocab_size=8,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=300,
        use_cache=False,
    )
    config._attn_implementation = "eager"
    model = Qwen3ForCausalLM(config).eval()
    return LoadedArm(
        model=model,
        tokenizer=CharacterTokenizer(),
        canonical_ids=(3, 4, 5, 6),
        mask_token_id=7,
    )


def test_development_split_rejects_even_autosome() -> None:
    assert_development_split(pd.DataFrame({"chrom": ["1", "X"], "label": [0, 1]}), "ok")
    with pytest.raises(RuntimeError, match="held-out"):
        assert_development_split(pd.DataFrame({"chrom": ["2"], "label": [1]}), "bad")


def test_load_variant_frame_downloads_only_pinned_train_parquet(
    tmp_path: Path,
    monkeypatch: object,
) -> None:
    train_path = tmp_path / "train.parquet"
    pd.DataFrame(
        {
            "chrom": [1, 3],
            "pos": [10, 20],
            "ref": ["a", "c"],
            "alt": ["g", "t"],
            "label": [0, 1],
        }
    ).to_parquet(train_path, index=False)
    requested: list[dict[str, str]] = []

    def fake_download(**kwargs: str) -> str:
        requested.append(kwargs)
        return str(train_path)

    monkeypatch.setattr(  # type: ignore[attr-defined]
        "exp479_mntp.vep.hf_hub_download",
        fake_download,
    )
    spec = DatasetSpec(
        name="development",
        repo_id="example/evaluation",
        revision="a" * 40,
        protocol="minus_llr",
        evaluation="matched",
    )

    frame = load_variant_frame(spec)

    assert requested == [
        {
            "repo_id": "example/evaluation",
            "filename": "train.parquet",
            "repo_type": "dataset",
            "revision": "a" * 40,
        }
    ]
    assert frame["chrom"].tolist() == ["1", "3"]
    assert frame["ref"].tolist() == ["A", "C"]
    assert frame["alt"].tolist() == ["G", "T"]


def test_reverse_complement_preserves_unknowns_and_case() -> None:
    assert reverse_complement("ACGTNacgtn") == "nacgtNACGT"


def test_mntp_score_cannot_see_true_base_under_mask() -> None:
    first = "A" * 255
    second = first[:127] + "G" + first[128:]
    frame = pd.DataFrame(
        {
            "sequence": [first, second],
            "ref": ["A", "A"],
            "alt": ["C", "C"],
        }
    )
    scores = score_strand(_arm(), frame, objective="mntp", strand="fwd", batch_size=2)
    assert scores[0] == scores[1]


def test_attach_reference_windows_reads_bgzf_fasta(tmp_path: Path) -> None:
    fasta_path = tmp_path / "tiny.fa.bgz"
    with bgzf.BgzfWriter(str(fasta_path), "wb") as writer:
        writer.write(b">1\n" + b"A" * 255 + b"\n")
    with Fasta(fasta_path, as_raw=True, rebuild=True):
        pass

    frame = pd.DataFrame(
        {
            "chrom": ["1"],
            "pos": [128],
            "ref": ["A"],
            "alt": ["C"],
            "label": [1],
        }
    )
    result = attach_reference_windows(frame, fasta_path)
    assert result.loc[0, "sequence"] == "A" * 255


def test_mntp_score_masks_a_shifted_variant_position() -> None:
    variant_index = 63
    first = "A" * 255
    second = first[:variant_index] + "G" + first[variant_index + 1 :]
    frame = pd.DataFrame(
        {
            "sequence": [first, second],
            "ref": ["A", "A"],
            "alt": ["C", "C"],
        }
    )
    for strand in ("fwd", "rc"):
        scores = score_strand(
            _arm(),
            frame,
            objective="mntp",
            strand=strand,
            batch_size=2,
            variant_index=variant_index,
        )
        assert scores[0] == scores[1]


def test_attach_reference_windows_places_variant_at_requested_index(tmp_path: Path) -> None:
    fasta_path = tmp_path / "shifted.fa.bgz"
    with bgzf.BgzfWriter(str(fasta_path), "wb") as writer:
        writer.write(b">1\n" + b"A" * 383 + b"\n")
    with Fasta(fasta_path, as_raw=True, rebuild=True):
        pass

    frame = pd.DataFrame(
        {
            "chrom": ["1"],
            "pos": [192],
            "ref": ["A"],
            "alt": ["C"],
            "label": [1],
        }
    )
    for variant_index in (63, 127, 191):
        result = attach_reference_windows(
            frame,
            fasta_path,
            variant_index=variant_index,
        )
        assert len(result.loc[0, "sequence"]) == 255
        assert result.loc[0, "sequence"][variant_index] == "A"
