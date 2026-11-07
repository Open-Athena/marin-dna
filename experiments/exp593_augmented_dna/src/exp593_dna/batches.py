"""Shard augmented examples while retaining exact global token accounting."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import jax
import numpy as np
from exp586_moe.data import WindowBatch
from exp586_moe.gpu_smoke import device_batch
from jax.experimental import multihost_utils
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from exp593_dna.pipeline import KINDS, Augmenter, collate


@dataclass(frozen=True)
class TrainingBatch:
    """Model input plus host integers needed for scheduling and reproducibility."""

    model_batch: Any
    counts: dict[str, int]
    mixture: dict[str, int]
    bucket: int
    input_fingerprint: str


def distributed_batch(
    dataset: Any, augmenter: Augmenter, mesh: jax.sharding.Mesh, start: int
) -> TrainingBatch:
    """Load local rows only; sum their counts once across process ownership."""
    size = augmenter.config.global_batch_size
    if start < 0 or start % size:
        raise ValueError(
            "Training batches must begin on an absolute global-batch boundary"
        )
    bucket = augmenter.batch_bucket(start)
    shape = (size, bucket)
    sharding = NamedSharding(mesh, P(("replica_dcn", "data", "expert"), None))
    indices = sorted(
        {
            row
            for index in sharding.addressable_devices_indices_map(shape).values()
            for row in range(*index[0].indices(size))
        }
    )
    if not indices:
        raise ValueError("Every training process must own at least one example")
    occurrences = [start + i for i in indices]
    raw = dataset.get_batch(occurrences)
    examples = [
        augmenter.build(row, occurrence)
        for row, occurrence in zip(raw, occurrences, strict=True)
    ]
    local = collate(examples)
    if local["token_ids"].shape[1] != bucket:
        raise ValueError("Distributed input bucket differs across examples")
    tokens = np.zeros(shape, dtype=np.int32)
    weights = np.zeros(shape, dtype=np.float32)
    segments = np.full(shape, -1, dtype=np.int32)
    lengths = np.zeros(size, dtype=np.int32)
    tokens[indices] = local["token_ids"]
    weights[indices] = local["loss_weights"]
    segments[indices] = local["segment_ids"]
    lengths[indices] = local["lengths"]
    # Ownership plus per-row values make duplication or missing ranks fatal.
    details = np.zeros((size, 43), dtype=np.int32)
    for row, example in zip(indices, examples, strict=True):
        a = example.attributes
        details[row, :11] = (
            1,
            a["input_tokens"],
            a["dna_tokens"],
            a["text_tokens"],
            a["scored_tokens"],
            int(a["kind"].endswith("fim")),
            int(a["rc"]),
            a["raw_row"],
            KINDS.index(a["kind"]),
            int(a["annotation_guided"]),
            int(a["minimal_fim"]),
        )
        details[row, 11:] = np.frombuffer(
            hashlib.sha256(
                example.token_ids.astype("<i4").tobytes()
                + example.loss_weights.astype("<f4").tobytes()
            ).digest(),
            dtype=np.uint8,
        )
    gathered = np.asarray(multihost_utils.process_allgather(details, tiled=True))
    global_details = gathered.reshape(-1, size, details.shape[1]).sum(
        axis=0, dtype=np.int64
    )
    if not np.all(global_details[:, 0] == 1):
        raise ValueError("Distributed input rows have duplicate or missing owners")
    sums = global_details.sum(axis=0, dtype=np.int64)
    counts = {
        "dna_tokens": int(sums[2]),
        "text_tokens": int(sums[3]),
        "loss_targets": int(sums[4]),
        "examples": size,
        "fim_examples": int(sums[5]),
        "rc_examples": int(sums[6]),
    }
    if counts["dna_tokens"] + counts["text_tokens"] != int(sums[1]):
        raise ValueError("Global actual input accounting differs")
    mixture = {
        kind: int(np.count_nonzero(global_details[:, 8] == i))
        for i, kind in enumerate(KINDS)
    }
    mixture.update(annotation_guided=int(sums[9]), minimal_fim=int(sums[10]))
    fingerprint = hashlib.sha256(
        str(start).encode() + b":" + global_details.astype("<i8").tobytes()
    ).hexdigest()
    model_batch = device_batch(WindowBatch(tokens, segments, weights, lengths), mesh)
    return TrainingBatch(model_batch, counts, mixture, bucket, fingerprint)
