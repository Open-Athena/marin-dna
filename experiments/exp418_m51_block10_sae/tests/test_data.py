from __future__ import annotations

from datasets import IterableDataset

from data import (
    CONTEXT_TOKENS,
    N_SOURCES,
    SOURCES,
    WINDOW_BP,
    balanced_budget,
    interleave_five_sources,
    normalize_and_tokenize_row,
)


class FakeTokenizer:
    bos_token_id = 2
    pad_token_id = 0
    unk_token_id = 1

    def __call__(self, sequence: str, **_: object) -> dict[str, list[int]]:
        return {"input_ids": [self.bos_token_id] + [3] * len(sequence)}


def test_exact_budgets() -> None:
    wiring = balanced_budget(1_000_000)
    micro = balanced_budget(5_000_000)
    assert wiring.actual_activations == 1_000_875
    assert wiring.windows_per_source == 785
    assert micro.actual_activations == 5_000_550
    assert micro.windows_per_source == 3_922
    assert wiring.total_windows == wiring.windows_per_source * N_SOURCES
    assert micro.total_windows == micro.windows_per_source * N_SOURCES


def test_normalize_row_has_bos_plus_255_bases() -> None:
    row = normalize_and_tokenize_row(
        {"id": "chr1:0-255_+", "seq": "A" * WINDOW_BP},
        source=SOURCES[0],
        tokenizer=FakeTokenizer(),  # type: ignore[arg-type]
    )
    assert row["source"] == "cds"
    assert len(row["input_ids"]) == CONTEXT_TOKENS
    assert row["input_ids"][0] == 2


def test_round_robin_order_is_exact() -> None:
    streams = [
        IterableDataset.from_generator(
            lambda source=source: (
                {"input_ids": [index], "source": source.name} for index in range(3)
            )
        )
        for source in SOURCES
    ]
    # The experiment always consumes a bounded prefix, far before any source
    # exhausts.  Assert two complete groups rather than the terminal partial
    # group emitted by datasets' ``first_exhausted`` scheduling semantics.
    rows = list(interleave_five_sources(streams).take(2 * N_SOURCES))
    assert [row["source"] for row in rows] == [
        source.name for _ in range(2) for source in SOURCES
    ]
