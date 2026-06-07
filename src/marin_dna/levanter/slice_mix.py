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
* ``inject_slice_mix`` — worker-side glue: given a *materialized* marin
  ``TrainLmOnPodConfig`` whose training data is one cache-backed component, load
  that component's token-sequence dataset (reusing levanter's own
  ``build_caches`` + ``build_token_datasets``, which route through the DNA-patched
  ``dataset_for_component`` so per-token loss weights are preserved), slice-mix
  it, and return a new pod config whose training data is the ``K``
  ``DirectDatasetComponent`` slices (validation components untouched,
  ``shuffle=False`` since the slices are already block-shuffled).

The injection must run on the training worker *after* the executor has
materialized the tokenize cache (the cache must exist to be sliced) and *before*
levanter builds the data loader — see the experiment's worker entrypoint.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, cast

import jax
from haliax import Axis
from jaxtyping import PRNGKeyArray
from levanter.data.dataset import AsyncDataset
from levanter.data.text import DirectDatasetComponent, LmDataConfig
from levanter.utils.thread_utils import blocking_wait

if TYPE_CHECKING:
    from levanter.main.train_lm import TrainLmConfig
    from marin.training.training import TrainLmOnPodConfig

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
            "inject_slice_mix expects fixed dict train_weights with exactly one "
            f"nonzero entry; got {type(weights).__name__}"
        )
    nonzero = [name for name, w in weights.items() if w and w > 0]
    if len(nonzero) != 1:
        raise ValueError(
            "inject_slice_mix expects exactly one training component (single-"
            f"dataset specialist); found {len(nonzero)}: {sorted(nonzero)}"
        )
    return nonzero[0]


def inject_slice_mix(
    pod_config: "TrainLmOnPodConfig",
    *,
    num_slices: int,
    seed: int = 0,
    io_block_size: int = DEFAULT_IO_BLOCK_SIZE,
    window_blocks: int = DEFAULT_WINDOW_BLOCKS,
    mixture_block_size: int | None = None,
) -> "TrainLmOnPodConfig":
    """Return a copy of ``pod_config`` with its training data slice-mixed.

    Must be called on the worker **after** ``materialize`` (the tokenize cache
    must already exist) and **before** levanter builds the data loader. The
    training data config must have exactly one cache-backed training component
    (nonzero ``train_weights``); validation components (zero weight) are carried
    through untouched.

    The single training component's token-sequence dataset is built via the
    config's own ``build_caches`` + ``build_token_datasets`` (so it goes through
    the DNA-patched ``dataset_for_component`` and keeps the per-token loss
    weights), then sliced into ``num_slices`` block-shuffled
    ``DirectDatasetComponent``s mixed by size fraction. ``shuffle`` is set to
    ``False`` because the slices are already block-shuffled; ``stop_strategy``
    (RESTART) and ``mixture_block_size`` are inherited unless overridden.

    Args:
        pod_config: A *materialized* ``TrainLmOnPodConfig`` (concrete cache paths).
        num_slices: Number of disjoint slices ``K`` to mix back together.
        seed: Seed for the per-slice block-shuffle permutations (resume-stable:
            the worker rebuilds the identical slices on restart).
        io_block_size: Per-slice block-shuffle I/O block size.
        window_blocks: Per-slice block-shuffle window size in blocks.
        mixture_block_size: Override the mixture block size; ``None`` keeps the
            config's existing value.

    Returns:
        A new ``TrainLmOnPodConfig`` with the slice-mixed training data.
    """
    # marin types ``TrainLmOnPodConfig.train_config`` as bare ``object``; it holds
    # a levanter ``TrainLmConfig`` here, so narrow it for attribute access.
    train_config = cast("TrainLmConfig", pod_config.train_config)
    data = train_config.data
    if not isinstance(data, LmDataConfig):
        raise TypeError(f"expected LmDataConfig, got {type(data).__name__}")

    train_name = _single_training_component_name(data.train_weights)

    # Reuse levanter's own loaders so the base dataset is byte-identical to what
    # training would otherwise consume: build_caches loads the existing cache
    # (no rebuild), build_token_datasets routes through the (DNA-patched)
    # dataset_for_component → per-token loss weights are preserved. For the
    # "train" split these return only the nonzero-weight component(s).
    Pos = Axis("position", train_config.train_seq_len)
    caches = data.build_caches("train")
    bases = data.build_token_datasets(caches, Pos, split="train")
    base = bases[train_name]

    slice_datasets, slice_weights = slice_mix_components(
        base,
        num_slices=num_slices,
        key=jax.random.PRNGKey(seed),
        name_prefix=train_name,
        io_block_size=io_block_size,
        window_blocks=window_blocks,
    )

    # Replace the single training component with the K block-shuffled slices;
    # carry every other (validation, zero-weight) component through unchanged.
    val_components = {
        name: comp for name, comp in data.components.items() if name != train_name
    }
    new_components: dict = {
        name: DirectDatasetComponent(datasets={"train": ds})
        for name, ds in slice_datasets.items()
    }
    new_components.update(val_components)

    new_weights = dict(slice_weights)
    new_weights.update({name: 0.0 for name in val_components})

    new_data = dataclasses.replace(
        data,
        components=new_components,
        train_weights=new_weights,
        shuffle=False,  # slices are already block-shuffled
        mixture_block_size=mixture_block_size
        if mixture_block_size is not None
        else data.mixture_block_size,
    )
    new_train_config = dataclasses.replace(train_config, data=new_data)
    return dataclasses.replace(pod_config, train_config=new_train_config)
