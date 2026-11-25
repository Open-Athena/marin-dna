"""Indexed raw windows with exp472's existing Levanter shuffle/mixture semantics."""

import hashlib
import json
from bisect import bisect_right
from collections import OrderedDict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import fsspec
import jax
import numpy as np
import pyarrow.parquet as pq
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from levanter.data.dataset import AsyncDataset
from levanter.data.mixture import MixtureDataset
from levanter.utils.jax_utils import key_iterator

from exp586_moe.data import WindowBatch, encode_windows
from exp586_moe.gpu_smoke import device_batch

CORPUS_PATH = Path(__file__).with_name("corpus.json")


def corpus_digest() -> str:
    return hashlib.sha256(CORPUS_PATH.read_bytes()).hexdigest()


class RawWindows(AsyncDataset[str]):
    """Read only requested Parquet row groups, with a bounded process-local cache."""

    def __init__(self, records: list[dict[str, Any]], cache_groups: int = 256) -> None:
        if (
            not records
            or cache_groups <= 0
            or any(
                not r["row_groups"]
                or any(n <= 0 for n in r["row_groups"])
                or sum(r["row_groups"]) != r["rows"]
                for r in records
            )
        ):
            raise ValueError("Invalid corpus inventory or cache size")
        self.records = records
        self.ends = np.cumsum([r["rows"] for r in records]).tolist()
        self.group_ends = [np.cumsum(r["row_groups"]).tolist() for r in records]
        self.files: dict[int, tuple[Any, pq.ParquetFile]] = {}
        self.cache: OrderedDict[tuple[int, int], Any] = OrderedDict()
        self.cache_groups = cache_groups

    async def async_len(self) -> int:
        return self.ends[-1]

    def is_finite(self) -> bool:
        return True

    async def get_batch(self, indices: Sequence[int]) -> Sequence[str]:
        result = []
        for index in indices:
            if not 0 <= index < self.ends[-1]:
                raise IndexError(index)
            file_id = bisect_right(self.ends, index)
            row = index - (self.ends[file_id - 1] if file_id else 0)
            group = bisect_right(self.group_ends[file_id], row)
            offset = row - (self.group_ends[file_id][group - 1] if group else 0)
            key = (file_id, group)
            if key not in self.cache:
                if file_id not in self.files:
                    record = self.records[file_id]
                    fs, path = fsspec.core.url_to_fs(record["uri"])
                    info = fs.info(path)
                    if (
                        info["size"] != record["bytes"]
                        or info.get("ETag") != record["etag"]
                    ):
                        raise ValueError("Raw corpus object identity changed")
                    stream = fs.open(path, "rb")
                    parquet = pq.ParquetFile(stream)
                    if parquet.metadata.num_rows != record["rows"]:
                        raise ValueError("Raw corpus row count changed")
                    if [
                        parquet.metadata.row_group(i).num_rows
                        for i in range(parquet.metadata.num_row_groups)
                    ] != record["row_groups"]:
                        raise ValueError("Raw corpus row groups changed")
                    self.files[file_id] = stream, parquet
                self.cache[key] = self.files[file_id][1].read_row_group(
                    group, columns=["seq"]
                )["seq"]
                while len(self.cache) > self.cache_groups:
                    self.cache.popitem(last=False)
            self.cache.move_to_end(key)
            result.append(self.cache[key][offset].as_py())
        return result


def training_dataset(raw: AsyncDataset[str], seed: int) -> AsyncDataset[str]:
    """Match LmDataConfig.train_set's one-component key derivation and wrappers."""
    mix_key, shuffle_key = jax.random.split(jax.random.PRNGKey(seed))
    shuffled = raw.block_shuffle(
        io_block_size=256,
        window_blocks=512,
        key=next(key_iterator(shuffle_key)),
        perm_type="feistel",
    )
    return MixtureDataset({"dna": shuffled}, {"dna": 1.0}, block_size=2048, key=mix_key)


def open_dataset(split: str, seed: int = 0) -> Any:
    raw = RawWindows(json.loads(CORPUS_PATH.read_text())[split])
    return (training_dataset(raw, seed) if split == "train" else raw).as_sync_dataset()


def distributed_batch(
    dataset: Any,
    tokenizer: Any,
    mesh: jax.sharding.Mesh,
    start: int,
    batch_size: int,
    *,
    augment: bool,
) -> Any:
    """Load each process's examples only, preserving the global occurrence order."""
    shape = (batch_size, 8192)
    sharding = NamedSharding(mesh, P(("replica_dcn", "data", "expert"), None))
    indices = sorted(
        {
            i
            for part in sharding.addressable_devices_indices_map(shape).values()
            for i in range(*part[0].indices(batch_size))
        }
    )
    occurrences = [start + i for i in indices]
    sequences = dataset.get_batch(occurrences)
    encoded = encode_windows(
        tokenizer, sequences, occurrences=occurrences if augment else None
    )
    tokens = np.zeros(shape, dtype=np.int32)
    segments = np.full(shape, -1, dtype=np.int32)
    weights = np.zeros(shape, dtype=np.float32)
    tokens[indices], segments[indices], weights[indices] = (
        encoded.token_ids,
        encoded.segment_ids,
        encoded.loss_weights,
    )
    batch = WindowBatch(
        tokens, segments, weights, np.full(batch_size, 8192, dtype=np.int32)
    )
    return device_batch(batch, mesh)
