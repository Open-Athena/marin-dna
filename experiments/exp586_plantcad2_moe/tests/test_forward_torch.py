import pytest

from exp586_moe.inference.forward_torch import (
    fixture_batch,
    latency_summary,
    validation_batches,
)


def test_forward_units_count_full_sequences() -> None:
    result = latency_summary([250.0, 1000.0, 500.0], batch_size=8)
    assert result["median_forward_ms"] == 500.0
    assert result["forward_sequences_per_second"] == 16.0
    assert result["forward_tokens_per_second"] == 131072.0


@pytest.mark.parametrize("values", [[], [0.0], [-1.0], [float("nan")], [float("inf")]])
def test_forward_rejects_invalid_event_times(values: list[float]) -> None:
    with pytest.raises(ValueError):
        latency_summary(values, 1)


def test_rotating_fixture_keeps_unique_complete_windows() -> None:
    sequences = ["A" * 8192, "C" * 8192, "G" * 8192]
    assert fixture_batch(sequences, 2, 2) == [sequences[2], sequences[0]]
    with pytest.raises(ValueError):
        fixture_batch(sequences, 4, 0)


@pytest.mark.parametrize("batch_size", [1, 2, 3, 4, 8])
def test_validation_uses_one_compile_shape_and_exact_scored_rows(
    batch_size: int,
) -> None:
    sequences = [str(i) for i in range(8)]
    validation = sequences[:5]
    batches = list(validation_batches(validation, sequences, batch_size))
    assert all(
        len(batch) == batch_size and len(set(batch)) == batch_size
        for batch, _ in batches
    )
    assert [
        row for batch, real_rows in batches for row in batch[:real_rows]
    ] == validation
