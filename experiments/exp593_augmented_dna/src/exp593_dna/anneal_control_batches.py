"""Exact exp586 causal-window distribution for the exp593 annealing control."""

from __future__ import annotations

import hashlib
from dataclasses import replace
from typing import Any

import numpy as np
from exp586_moe.data import (
    encode_windows,
    pretrained_dna_tokenizer,
    reverse_complement_selected,
)
from jax.experimental import multihost_utils
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from exp593_dna.anneal import AnnealTrial
from exp593_dna.batches import LocalBatch, TrainingBatch, assemble_distributed_batch
from exp593_dna.pipeline import load_registry

ASSEMBLIES = tuple(sorted(load_registry()))
WINDOW = 8192
BATCH = 64


class OldDistributionSource:
    """The exp586 shuffled dataset with unchanged exp586 tokenization and RC."""

    def __init__(self, shuffled: Any) -> None:
        self.shuffled = shuffled
        self.tokenizer = pretrained_dna_tokenizer()


def prepare(
    source: OldDistributionSource,
    mesh: Any,
    trial: AnnealTrial,
    *,
    stage_tokens: int,
    stage_examples: int,
) -> LocalBatch:
    if stage_examples % BATCH or stage_tokens != stage_examples * WINDOW:
        raise ValueError("Control requires complete old-distribution windows")
    if stage_tokens >= trial.horizon or trial.horizon % (BATCH * WINDOW):
        raise ValueError("Control horizon must consist of whole original batches")
    start = trial.source_examples + stage_examples
    shape = (BATCH, WINDOW)
    sharding = NamedSharding(mesh, P(("replica_dcn", "data", "expert"), None))
    indices = sorted(
        {
            row
            for index in sharding.addressable_devices_indices_map(shape).values()
            for row in range(*index[0].indices(BATCH))
        }
    )
    occurrences = [start + row for row in indices]
    raw = source.shuffled.get_batch(occurrences)
    encoded = encode_windows(
        source.tokenizer,
        [record.seq for record in raw],
        occurrences=occurrences,
    )
    tokens = np.zeros(shape, dtype=np.int32)
    weights = np.zeros(shape, dtype=np.float32)
    segments = np.full(shape, -1, dtype=np.int32)
    lengths = np.zeros(BATCH, dtype=np.int32)
    tokens[indices] = encoded.token_ids
    weights[indices] = encoded.loss_weights
    segments[indices] = encoded.segment_ids
    lengths[indices] = encoded.lengths
    details = np.zeros((BATCH, 44), dtype=np.int32)
    for row, record, occurrence, ids, loss in zip(
        indices, raw, occurrences, encoded.token_ids, encoded.loss_weights, strict=True
    ):
        details[row, :11] = (
            1,
            WINDOW,
            WINDOW,
            0,
            WINDOW - 1,
            0,
            int(reverse_complement_selected(occurrence)),
            record.row,
            0,
            0,
            0,
        )
        details[row, 11:43] = np.frombuffer(
            hashlib.sha256(
                ids.astype("<i4").tobytes() + loss.astype("<f4").tobytes()
            ).digest(),
            dtype=np.uint8,
        )
        details[row, 43] = ASSEMBLIES.index(record.assembly)
    return LocalBatch(tokens, weights, segments, lengths, details, start, WINDOW)


def assemble(local: LocalBatch, mesh: Any) -> TrainingBatch:
    batch = assemble_distributed_batch(local, mesh)
    gathered = np.asarray(multihost_utils.process_allgather(local.details, tiled=True))
    rows = gathered.reshape(-1, BATCH, 44).sum(axis=0, dtype=np.int64)
    if batch.counts != {
        "dna_tokens": BATCH * WINDOW,
        "text_tokens": 0,
        "loss_targets": BATCH * (WINDOW - 1),
        "examples": BATCH,
        "fim_examples": 0,
        "rc_examples": int(rows[:, 6].sum()),
    }:
        raise ValueError("Control batch differs from the original 1T objective")
    mixture = dict(batch.mixture)
    mixture[f"bucket/{WINDOW}"] = BATCH
    mixture.update(
        {
            f"species/{name}": int(np.count_nonzero(rows[:, 43] == i))
            for i, name in enumerate(ASSEMBLIES)
        }
    )
    return replace(batch, mixture=mixture)
