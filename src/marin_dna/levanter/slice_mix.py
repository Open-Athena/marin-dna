# Copyright The MarinDNA Authors
# SPDX-License-Identifier: Apache-2.0

"""Epoch-reshuffling stopgap for single-dataset training via slice-mixing.

Background (issue #284, marin#2869). When one dataset is trained for many
epochs, levanter's data path applies a *single, fixed* block-shuffle permutation
(``PermutationDataset`` / ``BlockShufflingDataset``) and does **not** reshuffle on
restart — so the model sees the identical batch sequence every epoch. For a
*single-dataset specialist* (one training component, no other mixture components
to interleave with) this is unmitigated: e.g. the order-cohort TSS arm trains
~20.8 epochs over one ~1.97M-row dataset, showing the same ordering ~21x.

The stopgap (Eric Czech's gist, https://gist.github.com/eric-czech/a98dba4416613a208b9af943f6121f75):
slice the single (already-tokenized) dataset into ``K`` disjoint contiguous
components, block-shuffle each independently, and re-mix them with weights equal
to their size fractions. ``MixtureDataset`` (``randomize_blocks=True``,
``stop_strategy="restart"``) assigns each block a *fresh* dataset-id permutation
via ``jax.random.fold_in(key, block_index)``; because the block index keeps
increasing across epochs, the way the ``K`` slices are interleaved differs every
epoch, so the batch *composition* almost never repeats — using only the cheap,
fast block-shuffle (no expensive global per-epoch reshuffle). Each slice's own
internal order is a fixed cycle; the cross-epoch variation comes from the
re-mixing of the slices. That is exactly the behavior #284 sets out to test.

Two pieces:

* ``slice_mix_components`` — pure dataset logic (slice + block-shuffle + size
  weights) over any ``AsyncDataset``. Unit-tested against ``ListAsyncDataset``.
* ``SliceMixLmDataConfig`` — an ``LmDataConfig`` subclass that overrides
  ``train_set`` to slice-mix its single training component. The reshuffle MUST
  live in ``train_set`` (not a pre-train worker hook) because constructing the
  slices touches JAX eagerly (``BlockShufflingDataset.__init__`` calls
  ``jax.random.PRNGKey`` / ``jax.device_put`` / ``jax.random.split``), and
  ``levanter.main.train_lm.main`` only calls ``data.train_set(...)`` *after*
  ``levanter.initialize`` (i.e. after ``jax.distributed.initialize``). Building
  the slices any earlier raises "jax.distributed.initialize() must be called
  before any JAX calls that might initialise the XLA backend". Validation
  components (zero weight) use the inherited cache-backed path unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax
from jaxtyping import PRNGKeyArray
from levanter.data.dataset import AsyncDataset
from levanter.data.mixture import MixtureDataset
from levanter.data.text import LmDataConfig
from levanter.data.text.datasets import NamedLmDataset
from levanter.utils.thread_utils import blocking_wait

if TYPE_CHECKING:
    from haliax import Axis
    from levanter.models.lm_model import LmExample
    from levanter.schedule import BatchSchedule

# Match the baseline's hierarchical block-shuffle granularity
# (``levanter.data.text.DEFAULT_LM_DATA_SHUFFLE``) so each slice is shuffled
# exactly as the no-reshuffle baseline shuffles the whole dataset — the only
# intended difference between the arms is the per-epoch re-mixing, not the
# within-slice shuffle strength.
DEFAULT_IO_BLOCK_SIZE = 256
DEFAULT_WINDOW_BLOCKS = 512


def slice_mix_components(
    base: AsyncDataset,
    *,
    num_slices: int,
    key: PRNGKeyArray,
    name_prefix: str,
    io_block_size: int = DEFAULT_IO_BLOCK_SIZE,
    window_blocks: int = DEFAULT_WINDOW_BLOCKS,
) -> tuple[dict[str, AsyncDataset], dict[str, float]]:
    """Slice ``base`` into ``num_slices`` disjoint block-shuffled components.

    Splits ``[0, n)`` into ``num_slices`` contiguous, non-overlapping ranges
    (sizes differ by at most one element), wraps each in an independent
    ``BlockShufflingDataset`` (via ``AsyncDataset.block_shuffle``), and assigns
    each a mixture weight equal to its fraction of ``n`` (so the expected
    per-epoch coverage is uniform and the weights sum to exactly 1.0).

    NOTE: constructs ``BlockShufflingDataset``s, which eagerly touch JAX — call
    only after ``jax.distributed.initialize`` (i.e. from within ``train_set``).

    Args:
        base: The full (already-tokenized) training dataset to reshuffle.
        num_slices: Number of disjoint slices ``K`` to mix back together.
        key: PRNG key; split ``num_slices`` ways, one per slice's block-shuffle.
        name_prefix: Component-name prefix; slices are ``f"{name_prefix}__slice{i}"``.
        io_block_size: Block-shuffle I/O block size (per slice).
        window_blocks: Block-shuffle window size in blocks (per slice).

    Returns:
        ``(datasets, weights)`` keyed by slice component name. ``datasets`` maps
        to the block-shuffled ``AsyncDataset`` slices; ``weights`` to their size
        fractions.
    """
    if num_slices < 1:
        raise ValueError(f"num_slices must be >= 1, got {num_slices}")
    n = blocking_wait(base.async_len())
    if n < num_slices:
        raise ValueError(f"dataset length {n} is smaller than num_slices {num_slices}")

    # Contiguous cuts covering [0, n) exactly, so the slice sizes sum to n and
    # the weights sum to exactly 1.0.
    cuts = [round(i * n / num_slices) for i in range(num_slices + 1)]
    assert cuts[0] == 0 and cuts[-1] == n
    keys = jax.random.split(key, num_slices)

    datasets: dict[str, AsyncDataset] = {}
    weights: dict[str, float] = {}
    for i in range(num_slices):
        start, end = cuts[i], cuts[i + 1]
        assert end > start, f"empty slice {i}: [{start}, {end}) (n={n}, K={num_slices})"
        shuffled = base.slice_dataset(start, end).block_shuffle(
            io_block_size=io_block_size,
            window_blocks=window_blocks,
            key=keys[i],
        )
        name = f"{name_prefix}__slice{i}"
        datasets[name] = shuffled
        weights[name] = (end - start) / n
    return datasets, weights


def _single_training_component_name(weights: object) -> str:
    """Return the sole component name with nonzero training weight.

    Slice-mixing is defined for a *single*-dataset specialist. Refuse anything
    else loudly rather than silently reshuffling only one arm of a real mixture.
    """
    if not isinstance(weights, dict):
        raise ValueError(
            "SliceMixLmDataConfig expects fixed dict train_weights with exactly "
            f"one nonzero entry; got {type(weights).__name__}"
        )
    nonzero = [name for name, w in weights.items() if w and w > 0]
    if len(nonzero) != 1:
        raise ValueError(
            "SliceMixLmDataConfig expects exactly one training component (single-"
            f"dataset specialist); found {len(nonzero)}: {sorted(nonzero)}"
        )
    return nonzero[0]


@dataclass(frozen=True)
class SliceMixLmDataConfig(LmDataConfig):
    """``LmDataConfig`` that reshuffles its single training component each epoch.

    Identical to ``LmDataConfig`` except ``train_set`` slices the one
    (nonzero-weight) training component into ``num_slices`` block-shuffled pieces
    re-mixed by ``MixtureDataset`` (size-proportional weights). Construct it by
    copying a normal ``lm_mixture_data_config(...)`` output's fields and adding
    the knobs below; it serializes/cloudpickles like any dataclass, so the marin
    executor + iris worker handle it transparently. Validation components (zero
    weight) use the inherited cache-backed path; only ``train_set`` changes.
    """

    num_slices: int = 8
    slice_mix_seed: int = 0
    slice_io_block_size: int = DEFAULT_IO_BLOCK_SIZE
    slice_window_blocks: int = DEFAULT_WINDOW_BLOCKS

    def train_set(
        self,
        Pos: "Axis",
        batch_schedule: "BatchSchedule",
        *,
        key: PRNGKeyArray,
    ) -> AsyncDataset["LmExample"]:
        # This override only supports the single-dataset specialist; refuse the
        # train_sets() features it bypasses rather than silently ignore them.
        for attr in (
            "num_validation_sequences",
            "max_train_batches",
            "experiment_budget",
            "target_budget",
        ):
            if getattr(self, attr) is not None:
                raise NotImplementedError(
                    f"SliceMixLmDataConfig.train_set does not support {attr!r}"
                )

        train_name = _single_training_component_name(self.train_weights)
        # Build the single training component's token-seq dataset exactly as the
        # base path would: build_caches loads the existing cache (no rebuild),
        # build_token_datasets routes through the DNA-patched dataset_for_component
        # so per-token loss weights are preserved. (Both return only the nonzero-
        # weight components for the "train" split.) Runs post jax-init.
        caches = self.build_caches("train")
        base = self.build_token_datasets(caches, Pos, split="train")[train_name]

        slice_datasets, slice_weights = slice_mix_components(
            base,
            num_slices=self.num_slices,
            key=jax.random.PRNGKey(self.slice_mix_seed),
            name_prefix=train_name,
            io_block_size=self.slice_io_block_size,
            window_blocks=self.slice_window_blocks,
        )
        mixture = MixtureDataset(
            datasets=slice_datasets,
            weights=slice_weights,
            stop_strategy=self.stop_strategy,
            key=key,
            block_size=self.mixture_block_size,
        )
        return NamedLmDataset(mixture, Pos)
