import pytest

from exp586_moe.inference.optimized_torch import padded_batches


def _sequence(base: str) -> str:
    return base * 8192


def test_padded_batches_preserves_rows_and_fixed_shape() -> None:
    sequences = [_sequence("A"), _sequence("C"), _sequence("G")]

    batches = list(padded_batches(sequences, 2))

    assert batches == [
        (sequences[:2], 2),
        ([sequences[2], sequences[2]], 1),
    ]


def test_padded_batches_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="positive"):
        list(padded_batches([_sequence("A")], 0))
    with pytest.raises(ValueError, match="ACGTN"):
        list(padded_batches([_sequence("X")], 1))
