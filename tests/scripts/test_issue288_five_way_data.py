from __future__ import annotations

from collections import Counter
from typing import Any

import pytest
from datasets import Dataset

from scripts.issue288.five_way_data import (
    CONTEXT_TOKENS,
    SOURCES,
    WINDOW_BP,
    balanced_budget,
    interleave_five_sources,
    normalize_and_tokenize_row,
    reference_position,
    tokenize_source,
)


class FakeDnaTokenizer:
    bos_token_id = 2
    pad_token_id = 0
    unk_token_id = 1

    def __call__(self, sequence: str, **_: Any) -> dict[str, list[int]]:
        token_ids = {"a": 3, "c": 4, "g": 5, "t": 6}
        return {
            "input_ids": [self.bos_token_id]
            + [token_ids.get(base.lower(), self.unk_token_id) for base in sequence]
        }


def _genomes_row(index: int, strand: str = "+") -> dict[str, Any]:
    start = index * WINDOW_BP
    return {
        "id": f"NC_TEST.1:{start}-{start + WINDOW_BP}_{strand}",
        "seq": "AcgT" * 63 + "ACN",
    }


def _zoonomia_row(
    index: int,
    *,
    source_strand: str = "+",
    augmentation: str = "+",
) -> dict[str, Any]:
    start = index * WINDOW_BP
    return {
        "query_name": f"win_1_{index:09d}",
        "species": "Test_species",
        "t_chrom": "chr1",
        "t_start": start,
        "t_end": start + WINDOW_BP,
        "t_strand": source_strand,
        "t_src_size": 100_000,
        "sequence": "A" * WINDOW_BP,
        "augmentation": augmentation,
    }


def test_balanced_budgets_round_up_to_whole_five_way_windows() -> None:
    budget_1m = balanced_budget(1_000_000)
    assert budget_1m.windows_per_source == 785
    assert budget_1m.total_windows == 3_925
    assert budget_1m.actual_activations == 1_000_875

    budget_5m = balanced_budget(5_000_000)
    assert budget_5m.windows_per_source == 3_922
    assert budget_5m.total_windows == 19_610
    assert budget_5m.actual_activations == 5_000_550


@pytest.mark.parametrize(
    "source_index,row,expected_sequence_strand",
    [
        (0, _genomes_row(2, "+"), "+"),
        (0, _genomes_row(2, "-"), "-"),
        (3, _zoonomia_row(2, source_strand="+", augmentation="+"), "+"),
        (3, _zoonomia_row(2, source_strand="-", augmentation="+"), "-"),
        (3, _zoonomia_row(2, source_strand="+", augmentation="-"), "-"),
        (3, _zoonomia_row(2, source_strand="-", augmentation="-"), "+"),
    ],
)
def test_normalization_preserves_coordinate_orientation(
    source_index: int,
    row: dict[str, Any],
    expected_sequence_strand: str,
) -> None:
    normalized = normalize_and_tokenize_row(
        row,
        source=SOURCES[source_index],
        tokenizer=FakeDnaTokenizer(),  # type: ignore[arg-type]
    )
    assert normalized["sequence_strand"] == expected_sequence_strand
    assert len(normalized["input_ids"]) == CONTEXT_TOKENS
    assert normalized["input_ids"][0] == FakeDnaTokenizer.bos_token_id
    assert FakeDnaTokenizer.pad_token_id not in normalized["input_ids"]

    if expected_sequence_strand == "+":
        assert reference_position(normalized, 0) == normalized["start"]
        assert reference_position(normalized, WINDOW_BP - 1) == normalized["end"] - 1
    else:
        assert reference_position(normalized, 0) == normalized["end"] - 1
        assert reference_position(normalized, WINDOW_BP - 1) == normalized["start"]


def test_tokenization_counts_lowercase_and_ambiguous_bases() -> None:
    normalized = normalize_and_tokenize_row(
        _genomes_row(0),
        source=SOURCES[0],
        tokenizer=FakeDnaTokenizer(),  # type: ignore[arg-type]
    )
    assert normalized["lowercase_bases"] == 126
    assert normalized["unknown_tokens"] == 1


def test_rejects_wrong_window_length() -> None:
    row = _genomes_row(0)
    row["seq"] = "A" * (WINDOW_BP - 1)
    with pytest.raises(AssertionError):
        normalize_and_tokenize_row(
            row,
            source=SOURCES[0],
            tokenizer=FakeDnaTokenizer(),  # type: ignore[arg-type]
        )


def test_five_way_stream_is_exact_round_robin() -> None:
    tokenizer = FakeDnaTokenizer()
    streams = []
    for source_index, source in enumerate(SOURCES):
        if source.schema == "genomes_v5":
            rows = [_genomes_row(i) for i in range(4)]
        else:
            rows = [_zoonomia_row(i) for i in range(4)]
        raw = Dataset.from_list(rows).to_iterable_dataset(num_shards=1)
        streams.append(
            tokenize_source(
                raw,
                source,
                tokenizer,  # type: ignore[arg-type]
            )
        )

    mixed = list(interleave_five_sources(streams).take(15))
    expected_order = [source.name for source in SOURCES] * 3
    assert [row["source"] for row in mixed] == expected_order
    assert Counter(row["source"] for row in mixed) == Counter(
        {source.name: 3 for source in SOURCES}
    )
    assert all(len(row["input_ids"]) == CONTEXT_TOKENS for row in mixed)
