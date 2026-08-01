from __future__ import annotations

from typing import ClassVar

from datasets import IterableDataset

from data import (
    CONTEXT_TOKENS,
    N_STREAMS,
    ORIENTATIONS,
    SOURCES,
    WINDOW_BP,
    interleave_source_orientations,
    normalize_and_tokenize_row,
)


class FakeTokenizer:
    bos_token_id = 2
    pad_token_id = 0
    unk_token_id = 1
    vocabulary: ClassVar[dict[str, int]] = {"A": 3, "C": 4, "G": 5, "T": 6}

    def __call__(self, sequence: str, **_: object) -> dict[str, list[int]]:
        return {
            "input_ids": [self.bos_token_id]
            + [
                self.vocabulary.get(base.upper(), self.unk_token_id)
                for base in sequence
            ]
        }


def test_forward_and_reverse_complement_share_record_id() -> None:
    sequence = "A" * 100 + "C" * 55 + "G" * 100
    row = {"id": "chr1:0-255_+", "seq": sequence}
    forward = normalize_and_tokenize_row(
        row,
        source=SOURCES[0],
        orientation="forward",
        tokenizer=FakeTokenizer(),  # type: ignore[arg-type]
    )
    reverse = normalize_and_tokenize_row(
        row,
        source=SOURCES[0],
        orientation="reverse_complement",
        tokenizer=FakeTokenizer(),  # type: ignore[arg-type]
    )
    assert forward["record_id"] == reverse["record_id"]
    assert len(forward["input_ids"]) == len(reverse["input_ids"]) == CONTEXT_TOKENS
    assert forward["input_ids"][1] == FakeTokenizer.vocabulary["A"]
    assert reverse["input_ids"][1] == FakeTokenizer.vocabulary["C"]
    assert forward["input_ids"] != reverse["input_ids"]


def test_source_orientation_interleave_is_exact() -> None:
    labels = [
        (source.name, orientation) for source in SOURCES for orientation in ORIENTATIONS
    ]
    streams = [
        IterableDataset.from_generator(
            lambda source=source, orientation=orientation: (
                {
                    "input_ids": [index],
                    "source": source,
                    "orientation": orientation,
                    "record_id": f"record-{index}",
                }
                for index in range(3)
            )
        )
        for source, orientation in labels
    ]
    rows = list(interleave_source_orientations(streams).take(2 * N_STREAMS))
    assert [(row["source"], row["orientation"]) for row in rows] == labels * 2


def test_fixed_window_and_stream_counts() -> None:
    assert WINDOW_BP == 255
    assert N_STREAMS == 10
    assert N_STREAMS == len(SOURCES) * len(ORIENTATIONS)
