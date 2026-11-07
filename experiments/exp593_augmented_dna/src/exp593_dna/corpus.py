"""Keep original train-row identity through the optimized raw DNA shuffle."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from importlib.resources import files
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from exp586_moe.corpus import RawWindows, training_dataset
from levanter.data.dataset import AsyncDataset

from exp593_dna.inputs import PreparedInputs
from exp593_dna.pipeline import RawWindow

SPLIT_ROWS = {"train": 2_638_656, "validation": 329_832}


class AnnotatedWindows(AsyncDataset[RawWindow]):
    """Join native sequence to exact flat-row metadata before shuffling."""

    def __init__(
        self,
        raw: RawWindows,
        metadata: pa.Table,
        split: str,
        *,
        expected_rows: int,
    ) -> None:
        if split not in SPLIT_ROWS:
            raise ValueError(
                "Only the approved train and validation splits are allowed"
            )
        row_ids = metadata.column("row").to_numpy()
        splits = set(metadata.column("split").to_pylist())
        if (
            metadata.num_rows != expected_rows
            or raw.ends[-1] != expected_rows
            or not np.array_equal(row_ids, np.arange(expected_rows))
            or splits != {split}
        ):
            raise ValueError("Raw DNA and metadata row/split identity differ")
        self.raw = raw
        self.metadata = metadata
        self.split = split
        self.expected_rows = expected_rows

    async def async_len(self) -> int:
        return self.expected_rows

    def is_finite(self) -> bool:
        return True

    async def get_batch(self, indices: Sequence[int]) -> Sequence[RawWindow]:
        sequences = await self.raw.get_batch(indices)
        if len(sequences) != len(indices):
            raise ValueError("Raw source batch length changed")
        result = []
        for index, dna in zip(indices, sequences, strict=True):
            record = self.metadata.slice(index, 1).to_pylist()[0]
            record["seq"] = dna
            result.append(RawWindow.from_dict(record))
        return result


def open_corpus(
    inputs: PreparedInputs,
    split: str,
    seed: int,
    *,
    preload: bool = False,
    progress: Callable[[int, int], None] | None = None,
) -> Any:
    """Retain exp586 block-shuffle semantics and optional decoded Parquet caching."""
    if split not in SPLIT_ROWS:
        raise ValueError("Split is outside the training experiment")
    inventory = json.loads(
        files("exp593_dna").joinpath("assets/corpus.json").read_text()
    )
    raw = RawWindows(inventory[split], cache_groups=4096, cache_files=preload)
    if preload:

        def warm_file(index: int) -> None:
            opened = raw.open_verified_file(index)
            raw.files[index] = opened
            for group in range(len(raw.records[index]["row_groups"])):
                raw.cache[index, group] = opened[1].read_row_group(
                    group, columns=["seq"]
                )["seq"]

        with ThreadPoolExecutor(max_workers=4) as pool:
            for completed, _ in enumerate(
                pool.map(warm_file, range(len(raw.records))), 1
            ):
                if progress is not None:
                    progress(completed, len(raw.records))
    names = inputs.names(f"metadata/{split}/", ".parquet")
    expected_shards = 44 if split == "train" else 6
    if len(names) != expected_shards:
        raise ValueError("Prepared metadata shard inventory differs")
    paths = [inputs.path(name) for name in names]
    metadata = pa.concat_tables([pq.read_table(path) for path in paths])
    dataset = AnnotatedWindows(raw, metadata, split, expected_rows=SPLIT_ROWS[split])
    return (
        training_dataset(dataset, seed) if split == "train" else dataset
    ).as_sync_dataset()
