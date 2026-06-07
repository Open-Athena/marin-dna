# Copyright The MarinDNA Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the epoch-reshuffling slice-mix helper (issue #284).

These exercise the pure, dataset-level logic (``slice_mix_components``) plus the
end-to-end re-mix behavior through ``MixtureDataset`` — the same invariant Eric's
gist asserts: batch orderings differ across epochs. The marin/levanter glue in
``inject_slice_mix`` needs a materialized cache + pod config, so it's covered by
the experiment's TPU smoke test rather than here.
"""

import pytest

pytest.importorskip("levanter", reason="install with `uv sync --extra marin` to run")

import jax  # noqa: E402
from levanter.data.dataset import ListAsyncDataset  # noqa: E402
from levanter.data.mixture import MixtureDataset  # noqa: E402
from levanter.utils.thread_utils import blocking_wait  # noqa: E402

from marin_dna.levanter.slice_mix import SliceMixLmDataConfig, slice_mix_components  # noqa: E402

# Small block-shuffle granularity so even a tiny test dataset is genuinely
# reshuffled within each slice.
_IO_BLOCK = 4
_WINDOW = 2


def _read_all(ds) -> list:
    n = blocking_wait(ds.async_len())
    return list(blocking_wait(ds.get_batch(list(range(n)))))


def test_slice_mix_components_partition():
    """Slices are disjoint, cover the whole dataset, and weights are size-proportional."""
    n, k = 1000, 8
    base = ListAsyncDataset(list(range(n)))
    datasets, weights = slice_mix_components(
        base,
        num_slices=k,
        key=jax.random.PRNGKey(0),
        name_prefix="t",
        io_block_size=_IO_BLOCK,
        window_blocks=_WINDOW,
    )

    assert set(datasets) == set(weights) == {f"t__slice{i}" for i in range(k)}
    assert weights == pytest.approx({name: 1 / k for name in weights}, abs=1e-3)
    assert sum(weights.values()) == pytest.approx(1.0)

    # Disjoint cover: the union of all slices is exactly range(n), with no overlap
    # (block-shuffle only reorders within a slice — it never drops or duplicates).
    seen: list[int] = []
    for name in datasets:
        elems = _read_all(datasets[name])
        assert len(set(elems)) == len(elems), f"{name} has duplicates"
        seen.extend(elems)
    assert sorted(seen) == list(range(n))


def test_slice_mix_epoch_reshuffling():
    """Across two epochs the batch ordering differs (the gist's invariant)."""
    n, k = 1000, 8
    base = ListAsyncDataset(list(range(n)))
    datasets, weights = slice_mix_components(
        base,
        num_slices=k,
        key=jax.random.PRNGKey(0),
        name_prefix="t",
        io_block_size=_IO_BLOCK,
        window_blocks=_WINDOW,
    )
    mixture = MixtureDataset(
        datasets=datasets,
        weights=weights,
        block_size=16,
        key=jax.random.PRNGKey(1),
        stop_strategy="restart",
    )

    # Two "epochs" worth of draws (the mixture is infinite under restart; one
    # conceptual epoch ≈ n draws given size-proportional weights).
    items = list(blocking_wait(mixture.get_batch(list(range(2 * n)))))
    assert all(0 <= x < n for x in items), "mixture drew an out-of-range element"

    epoch1, epoch2 = items[0:10], items[n : n + 10]
    assert epoch1 != epoch2, "epochs are identical — no reshuffling happened"


def test_slice_mix_rejects_bad_num_slices():
    base = ListAsyncDataset(list(range(4)))
    with pytest.raises(ValueError):
        slice_mix_components(
            base, num_slices=0, key=jax.random.PRNGKey(0), name_prefix="t"
        )
    with pytest.raises(ValueError):
        slice_mix_components(
            base, num_slices=8, key=jax.random.PRNGKey(0), name_prefix="t"
        )


def test_slice_mix_config_wraps_lmdataconfig():
    """exp284's construction pattern: copy a plain LmDataConfig's fields + knobs."""
    import dataclasses

    from levanter.data.text import LmDataConfig

    base = LmDataConfig(tokenizer="gpt2", mixture_block_size=512)
    cfg = SliceMixLmDataConfig(
        **{f.name: getattr(base, f.name) for f in dataclasses.fields(base)},
        num_slices=8,
        slice_mix_seed=3,
    )
    assert isinstance(cfg, LmDataConfig)
    assert cfg.num_slices == 8 and cfg.slice_mix_seed == 3
    # parent fields carried through verbatim
    assert cfg.tokenizer == "gpt2" and cfg.mixture_block_size == 512
    assert cfg.stop_strategy == base.stop_strategy
