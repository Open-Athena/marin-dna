"""Lightweight tests for train.py CLI helpers."""

from marin_dna.pipelines.enhancer_segmentation.train import _as_batches_arg


def test_as_batches_arg_full_pass_through() -> None:
    """1.0 must stay a float (Lightning: 100% of batches), NOT become int(1)."""
    assert _as_batches_arg(1.0) == 1.0
    assert isinstance(_as_batches_arg(1.0), float)


def test_as_batches_arg_fraction_below_one() -> None:
    assert _as_batches_arg(0.1) == 0.1
    assert isinstance(_as_batches_arg(0.1), float)


def test_as_batches_arg_explicit_int_count() -> None:
    """Values > 1 with no fractional part become integer batch counts."""
    assert _as_batches_arg(10.0) == 10
    assert isinstance(_as_batches_arg(10.0), int)


def test_as_batches_arg_non_integer_above_one_stays_float() -> None:
    assert _as_batches_arg(2.5) == 2.5
    assert isinstance(_as_batches_arg(2.5), float)
