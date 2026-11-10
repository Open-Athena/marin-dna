"""Deterministic host prefetch and globally audited annotation-only mixtures."""

from __future__ import annotations

import hashlib
from dataclasses import replace
from typing import Any

import numpy as np
from jax.experimental import multihost_utils
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from exp593_dna.anneal import AnnealTrial, terminal_lengths
from exp593_dna.anneal_data import GuidedSource
from exp593_dna.annotations import LANDMARK_CLASSES
from exp593_dna.batches import LocalBatch, TrainingBatch, assemble_distributed_batch
from exp593_dna.pipeline import KINDS, collate, load_registry

ASSEMBLIES = tuple(sorted(load_registry()))


def prepare(
    source: GuidedSource,
    mesh: Any,
    trial: AnnealTrial,
    *,
    stage_tokens: int,
    stage_examples: int,
) -> LocalBatch:
    start = trial.source_examples + stage_examples
    if stage_examples % 64:
        raise ValueError("Annealing requires whole global batches")
    tail = terminal_lengths(trial.horizon - stage_tokens)
    probability = trial.fim_probability(stage_tokens)
    presentations = [
        source.augmenter.presentation(
            start + i, probability, length=tail[i] if tail else None
        )
        for i in range(64)
    ]
    bucket = max(p.bucket for p in presentations)
    presentations = [replace(p, bucket=bucket) for p in presentations]
    shape = (64, bucket)
    sharding = NamedSharding(mesh, P(("replica_dcn", "data", "expert"), None))
    indices = sorted(
        {
            row
            for index in sharding.addressable_devices_indices_map(shape).values()
            for row in range(*index[0].indices(64))
        }
    )
    examples = source.get_examples([presentations[i] for i in indices])
    local = collate(examples)
    tokens = np.zeros(shape, dtype=np.int32)
    weights = np.zeros(shape, dtype=np.float32)
    segments = np.full(shape, -1, dtype=np.int32)
    lengths = np.zeros(64, dtype=np.int32)
    tokens[indices], weights[indices] = local["token_ids"], local["loss_weights"]
    segments[indices], lengths[indices] = local["segment_ids"], local["lengths"]
    details = np.zeros((64, 49), dtype=np.int32)
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
        details[row, 11:43] = np.frombuffer(
            hashlib.sha256(
                example.token_ids.astype("<i4").tobytes()
                + example.loss_weights.astype("<f4").tobytes()
            ).digest(),
            dtype=np.uint8,
        )
        details[row, 43:] = (
            LANDMARK_CLASSES.index(a["landmark_class"]) + 1
            if a["landmark_class"]
            else 0,
            int(a["landmark_mode"] == "nearby"),
            a["gap_length"],
            ASSEMBLIES.index(a["assembly"]),
            int(a["right_flank_first"]),
            a["scored_tokens"] if a["annotation_guided"] else 0,
        )
    return LocalBatch(tokens, weights, segments, lengths, details, start, bucket)


def assemble(local: LocalBatch, mesh: Any) -> TrainingBatch:
    batch = assemble_distributed_batch(local, mesh)
    gathered = np.asarray(multihost_utils.process_allgather(local.details, tiled=True))
    rows = gathered.reshape(-1, 64, local.details.shape[1]).sum(axis=0, dtype=np.int64)
    mixture = dict(batch.mixture)
    mixture.update(
        {
            f"landmark/{name}": int(np.count_nonzero(rows[:, 43] == i + 1))
            for i, name in enumerate(LANDMARK_CLASSES)
        }
    )
    mixture.update(
        {
            f"species/{name}": int(np.count_nonzero(rows[:, 46] == i))
            for i, name in enumerate(ASSEMBLIES)
        }
    )
    mixture.update(
        nearby=int(rows[:, 44].sum()),
        overlap=int(batch.counts["fim_examples"] - rows[:, 44].sum()),
        right_flank_first=int(rows[:, 47].sum()),
        fim_targets=int(rows[:, 48].sum()),
    )
    mixture.update(
        {
            f"gap/{gap}": int(np.count_nonzero(rows[:, 45] == gap))
            for gap in (1, 2, 3, 16, 32, 64, 128, 256, 512)
        }
    )
    if batch.counts["fim_examples"] != mixture["annotation_guided"]:
        raise ValueError("The global FIM batch includes an unguided example")
    return replace(batch, mixture=mixture)
