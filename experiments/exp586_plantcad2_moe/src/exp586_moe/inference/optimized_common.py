"""Backend-independent helpers for fixed-shape DNA inference."""

from collections.abc import Iterator, Sequence

from exp586_moe.inference.contract import validate_sequences


def padded_batches(
    sequences: Sequence[str], batch_size: int
) -> Iterator[tuple[list[str], int]]:
    """Yield fixed-shape batches and the number of real rows in each batch."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    for start in range(0, len(sequences), batch_size):
        batch = list(sequences[start : start + batch_size])
        validate_sequences(batch)
        real_rows = len(batch)
        while len(batch) < batch_size:
            batch.extend(batch[: min(len(batch), batch_size - len(batch))])
        yield batch, real_rows
