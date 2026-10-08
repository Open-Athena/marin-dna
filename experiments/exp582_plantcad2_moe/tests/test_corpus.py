"""Check raw-row identity and parity with the original Levanter data wrappers."""

import asyncio
import json

import jax
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from haliax import Axis
from levanter.data.dataset import AsyncDataset
from levanter.data.text.datasets import (
    BlockShuffleConfig,
    DirectDatasetComponent,
    LmDataConfig,
)
from levanter.schedule import BatchSchedule

from exp582_moe import corpus
from exp582_moe.data import encode_windows, scratch_tokenizer


def parquet_record(path, sequences, group_size):
    pq.write_table(pa.table({"seq": sequences}), path, row_group_size=group_size)
    parquet = pq.ParquetFile(path)
    return {
        "uri": str(path),
        "rows": len(sequences),
        "row_groups": [
            parquet.metadata.row_group(i).num_rows
            for i in range(parquet.metadata.num_row_groups)
        ],
        "bytes": path.stat().st_size,
        "etag": None,  # Local fixture files have no object-store ETag.
    }


def test_pinned_corpus_inventory_covers_the_original_splits():
    manifest = json.loads(corpus.CORPUS_PATH.read_text())
    for split, count, rows in (("train", 44, 2_638_656), ("validation", 6, 329_832)):
        records = manifest[split]
        assert len(records) == count
        assert sum(record["rows"] for record in records) == rows
        assert len({record["uri"] for record in records}) == count
        for i, record in enumerate(records):
            assert record["uri"].endswith(
                f"/{split}/data-{i:05d}-of-{count:05d}.parquet"
            )
            assert sum(record["row_groups"]) == record["rows"]
            assert all(size > 0 for size in record["row_groups"])
            assert record["bytes"] > 0 and record["etag"]


def test_raw_rows_cross_groups_and_files_in_requested_order(tmp_path):
    expected = [f"window-{i}" for i in range(12)]
    records = [
        parquet_record(tmp_path / "first.parquet", expected[:7], 3),
        parquet_record(tmp_path / "second.parquet", expected[7:], 2),
    ]
    raw = corpus.RawWindows(records, cache_groups=2)
    assert asyncio.run(raw.async_len()) == 12
    assert raw.is_finite()
    indices = [11, 0, 2, 3, 6, 7, 8, 9, 3, 11]
    assert asyncio.run(raw.get_batch(indices)) == [expected[i] for i in indices]
    assert len(raw.cache) <= 2
    assert asyncio.run(raw.get_batch([])) == []
    assert asyncio.run(raw.get_batch([0, 7, 11])) == [expected[i] for i in [0, 7, 11]]
    for index in (-1, 12):
        with pytest.raises(IndexError):
            asyncio.run(raw.get_batch([index]))


@pytest.mark.parametrize("field", ["bytes", "etag", "rows", "row_groups"])
def test_raw_reader_rejects_changed_object_or_index_metadata(tmp_path, field):
    record = parquet_record(tmp_path / "data.parquet", ["A", "C", "G"], 2)
    bad = dict(record)
    if field == "etag":
        bad[field] = '"different-object"'
    elif field == "row_groups":
        # Same object and total rows, but wrong group boundaries silently misindex rows.
        bad[field] = [1, 2]
    else:
        bad[field] += 1
    with pytest.raises(ValueError):
        asyncio.run(corpus.RawWindows([bad]).get_batch([1]))


@pytest.mark.parametrize("cache_groups", [0, -1])
def test_raw_reader_requires_a_positive_cache(tmp_path, cache_groups):
    record = parquet_record(tmp_path / "data.parquet", ["A"], 1)
    with pytest.raises(ValueError):
        corpus.RawWindows([record], cache_groups=cache_groups)


def test_raw_reader_rejects_an_empty_inventory():
    with pytest.raises(ValueError):
        corpus.RawWindows([])


class IndexDataset(AsyncDataset[int]):
    """Expose row identities without allocating or fetching the real corpus."""

    def __init__(self, size):
        self.size = size

    def is_finite(self):
        return True

    async def async_len(self):
        return self.size

    async def get_batch(self, indices):
        assert all(0 <= i < self.size for i in indices)
        return list(indices)


def original_stream(raw, seed):
    # Run the actual public train_set, including its key splitting, mixture-block
    # permutation and restart policy. Strip only the final named-tensor conversion.
    config = LmDataConfig(
        components={"dna": DirectDatasetComponent(datasets={"train": raw})},
        shuffle=BlockShuffleConfig(
            io_block_size=256, window_blocks=512, perm_type="feistel"
        ),
        auto_build_caches=False,
    )
    return config.train_set(
        Axis("position", 8192), BatchSchedule(64), key=jax.random.PRNGKey(seed)
    ).dataset


@pytest.mark.parametrize("seed", [0, 1])
def test_training_order_matches_actual_levanter_at_boundaries_and_resume(seed):
    size = 2_638_656
    indices = [
        0,
        1,
        255,
        256,
        2047,
        2048,
        131071,
        131072,
        size - 65,
        size - 64,
        size - 1,
        size,
        size + 1,
        9 * size - 1,
        9 * size,
        10 * size - 1,
        2**32 + 1,
    ]
    raw = IndexDataset(size)
    reference = original_stream(raw, seed)
    actual = corpus.training_dataset(raw, seed)
    expected = asyncio.run(reference.get_batch(indices))
    assert asyncio.run(actual.get_batch(indices)) == expected
    assert not actual.is_finite()
    # A new process can seek directly by absolute occurrence, without replaying
    # earlier I/O or depending on how the original request was divided into batches.
    resumed = corpus.training_dataset(IndexDataset(size), seed)
    got = asyncio.run(resumed.get_batch(indices[8:]))
    assert got == expected[8:]
    pieces = [
        asyncio.run(actual.get_batch(indices[i : i + 3]))
        for i in range(0, len(indices), 3)
    ]
    assert [item for part in pieces for item in part] == expected


def test_distinct_data_seeds_change_order():
    raw = IndexDataset(8192)
    indices = list(range(64))
    first = asyncio.run(corpus.training_dataset(raw, 0).get_batch(indices))
    second = asyncio.run(corpus.training_dataset(raw, 1).get_batch(indices))
    assert first != second


@pytest.mark.parametrize("augment", [False, True])
def test_distributed_reader_fetches_only_local_absolute_occurrences(
    monkeypatch, augment
):
    class LocalSharding:
        def addressable_devices_indices_map(self, shape):
            assert shape == (8, 8192)
            return {0: (slice(2, 4), slice(None)), 1: (slice(6, 8), slice(None))}

    class Windows:
        def get_batch(self, indices):
            self.requested = indices
            return ["ACGT"[i % 4] * 8192 for i in indices]

    monkeypatch.setattr(corpus, "NamedSharding", lambda mesh, spec: LocalSharding())
    monkeypatch.setattr(corpus, "device_batch", lambda batch, mesh: batch)
    data = Windows()
    start = 2_638_656 + 64
    batch = corpus.distributed_batch(
        data, scratch_tokenizer(), None, start, 8, augment=augment
    )
    local = [2, 3, 6, 7]
    occurrences = [start + i for i in local]
    assert data.requested == occurrences
    expected = encode_windows(
        scratch_tokenizer(),
        ["ACGT"[i % 4] * 8192 for i in occurrences],
        occurrences=occurrences if augment else None,
    )
    np.testing.assert_array_equal(batch.token_ids[local], expected.token_ids)
    np.testing.assert_array_equal(batch.segment_ids[local], 0)
    np.testing.assert_array_equal(batch.loss_weights[local], expected.loss_weights)
    np.testing.assert_array_equal(batch.segment_ids[[0, 1, 4, 5]], -1)
    np.testing.assert_array_equal(batch.loss_weights[[0, 1, 4, 5]], 0)
    assert 7 not in batch.token_ids  # EOS remains vocabulary-only.
