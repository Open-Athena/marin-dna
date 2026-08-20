"""Small DNA helpers needed by the isolated scoring environment."""

_DNA_REVERSE_COMPLEMENT = str.maketrans("ACGTNacgtn", "TGCANtgcan")


def reverse_complement(sequence: str) -> str:
    """Reverse-complement DNA while preserving case and unknown characters."""
    return sequence.translate(_DNA_REVERSE_COMPLEMENT)[::-1]
