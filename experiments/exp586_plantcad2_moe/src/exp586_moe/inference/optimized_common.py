"""Backend-independent helpers for fixed-shape DNA inference."""

from collections.abc import Iterator, Sequence

from exp586_moe.inference.contract import validate_sequences


def _unique_fillers(batch: list[str], count: int) -> list[str]:
    """Create valid, disposable rows for a fixed-shape partial batch."""
    if count == 0:
        return []
    template = batch[0]
    used = set(batch)
    fillers: list[str] = []
    alphabet = "ACGTN"
    candidate_index = 0
    while len(fillers) < count:
        digits = candidate_index
        prefix = list(template[:8])
        for position in range(len(prefix)):
            prefix[position] = alphabet[digits % len(alphabet)]
            digits //= len(alphabet)
        candidate = "".join(prefix) + template[len(prefix) :]
        candidate_index += 1
        if candidate not in used:
            fillers.append(candidate)
            used.add(candidate)
    return fillers


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
        batch.extend(_unique_fillers(batch, batch_size - len(batch)))
        yield batch, real_rows
