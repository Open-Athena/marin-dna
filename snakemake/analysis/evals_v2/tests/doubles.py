"""Small protocol doubles shared by eval model tests."""

from typing import ClassVar


class DnaTokenizerStub:
    """Identity-hashable tokenizer double for one-token DNA bases."""

    _BASE_IDS: ClassVar[dict[str, int]] = {"A": 4, "C": 5, "G": 6, "T": 7}

    def __init__(self, *, bos: bool, eos: bool) -> None:
        self.bos_token_id = 2 if bos else None
        self.eos_token_id = 3 if eos else None
        self.mask_token_id = None

    def encode(self, sequence: str) -> list[int]:
        ids = [self._BASE_IDS[base] for base in sequence.upper()]
        if self.bos_token_id is not None:
            ids.insert(0, self.bos_token_id)
        if self.eos_token_id is not None:
            ids.append(self.eos_token_id)
        return ids
