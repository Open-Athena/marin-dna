"""One-window DNA examples, deterministic strand augmentation, and explicit padding."""

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from huggingface_hub import hf_hub_download
from tokenizers import AddedToken, Tokenizer, models, normalizers, pre_tokenizers

DATASET_REVISION = "4a444fff5520b992aa978d92a5af509a81977098"
LANGUAGE_TOKENIZER = "marin-community/marin-tokenizer"
LANGUAGE_TOKENIZER_REVISION = "a5ca45f2feb6c959bd87b81689aa7279b5bdcaa2"
DNA_TOKENIZER = "kuleshov-group/PlantCAD2-Small-l24-d0768"
DNA_TOKENIZER_REVISION = "f756c255cb76e9f538c3acec04acf4214ed03fb3"
AUGMENTATION_SEED = 472
WINDOW_BASES = 8192
SCRATCH_VOCAB = {
    "[PAD]": 0,
    "[MASK]": 1,
    "[UNK]": 2,
    "a": 3,
    "c": 4,
    "g": 5,
    "t": 6,
    "[EOS]": 7,
}
_COMPLEMENT = str.maketrans("acgt", "tgca")


def scratch_tokenizer() -> Tokenizer:
    """Build a character tokenizer with the historical IDs and vocabulary-only EOS."""
    tokenizer = Tokenizer(models.WordLevel(SCRATCH_VOCAB, unk_token="[UNK]"))
    tokenizer.normalizer = normalizers.Lowercase()
    tokenizer.pre_tokenizer = pre_tokenizers.Split("", behavior="isolated")
    tokenizer.add_special_tokens(
        [
            AddedToken(t, normalized=False, special=True)
            for t in ("[PAD]", "[MASK]", "[UNK]", "[EOS]")
        ]
    )
    return tokenizer


def language_tokenizer() -> Tokenizer:
    """Load the unchanged, pinned language tokenizer; callers disable added tokens."""
    return Tokenizer.from_file(
        hf_hub_download(
            LANGUAGE_TOKENIZER, "tokenizer.json", revision=LANGUAGE_TOKENIZER_REVISION
        )
    )


def pretrained_dna_tokenizer() -> Tokenizer:
    """Use original language IDs with one lowercase ASCII letter per DNA token.

    Pre-token boundaries prevent every cross-character BPE merge, including
    whole-word vocabulary matches when the source uses ``ignore_merges``.
    The original tokenizer remains available separately for text evaluation.
    """
    tokenizer = language_tokenizer()
    tokenizer.normalizer = normalizers.Lowercase()
    tokenizer.pre_tokenizer = pre_tokenizers.Split("", behavior="isolated")
    tokenizer.post_processor = None
    return tokenizer


def tokenizer_digest(tokenizer: Tokenizer) -> str:
    """Bind segmentation, normalization, vocabulary IDs and special-token rules."""
    return hashlib.sha256(tokenizer.to_str().encode()).hexdigest()


def reverse_complement_selected(index: int) -> bool:
    """Reproduce exp472's Bernoulli draw for an absolute training occurrence."""
    if index < 0:
        raise ValueError("Training occurrence must be nonnegative")
    rng = np.random.default_rng(
        np.random.SeedSequence([AUGMENTATION_SEED, index & 0xFFFFFFFF, index >> 32])
    )
    return bool(rng.random() < 0.5)


def reverse_complement(sequence: str) -> str:
    """Complement canonical bases; other characters retain exp472's UNK semantics."""
    return sequence.lower().translate(_COMPLEMENT)[::-1]


def prepare_window(sequence: str, *, occurrence: int | None) -> str:
    """Normalize one original window and optionally apply training augmentation."""
    if (
        len(sequence) != WINDOW_BASES
        or not sequence.isascii()
        or not sequence.isalpha()
    ):
        raise ValueError("Expected one original 8192-base ASCII DNA window")
    normalized = sequence.lower()
    if occurrence is not None and reverse_complement_selected(occurrence):
        return reverse_complement(normalized)
    return normalized


@dataclass(frozen=True)
class WindowBatch:
    token_ids: np.ndarray
    segment_ids: np.ndarray
    loss_weights: np.ndarray
    lengths: np.ndarray

    @property
    def input_tokens(self) -> int:
        return int(self.lengths.sum(dtype=np.int64))

    @property
    def loss_targets(self) -> int:
        return int(self.loss_weights.sum(dtype=np.float64))


def encode_windows(
    tokenizer: Tokenizer,
    sequences: Sequence[str],
    *,
    occurrences: Sequence[int] | None = None,
    padded_length: int = WINDOW_BASES,
    pad_id: int = 0,
) -> WindowBatch:
    """Keep each window separate and mask padding in both attention and next-token loss."""
    if not sequences:
        raise ValueError("A batch must contain at least one window")
    if occurrences is not None and len(occurrences) != len(sequences):
        raise ValueError("Every training example needs its absolute occurrence index")
    strings = [
        prepare_window(seq, occurrence=None if occurrences is None else occurrences[i])
        for i, seq in enumerate(sequences)
    ]
    encoded = tokenizer.encode_batch(strings, add_special_tokens=False)
    lengths = np.asarray([len(example.ids) for example in encoded], dtype=np.int32)
    if np.any(lengths != WINDOW_BASES):
        raise ValueError("DNA encoding must produce exactly one token per base")
    if np.any(lengths < 2) or np.any(lengths > padded_length):
        raise ValueError(
            f"Token lengths {lengths.tolist()} do not fit {padded_length}; truncation is forbidden"
        )
    tokens = np.full((len(sequences), padded_length), pad_id, dtype=np.int32)
    segments = np.full(tokens.shape, -1, dtype=np.int32)
    loss = np.zeros(tokens.shape, dtype=np.float32)
    for row, example in enumerate(encoded):
        length = len(example.ids)
        tokens[row, :length] = example.ids
        segments[row, :length] = 0
        loss[row, : length - 1] = 1
    return WindowBatch(tokens, segments, loss, lengths)
