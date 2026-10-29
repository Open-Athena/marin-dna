"""Immutable model, input, and output contracts for inference benchmarks."""

from dataclasses import dataclass
from enum import StrEnum

from tokenizers import Tokenizer

RUN_ID = "exp586-plantcad2-d768-pretrained-lrm1-seed0-v1"
CHECKPOINT = (
    "s3://marin-us-east-02a/MarinDNA/exp586_plantcad2_moe/checkpoints/"
    f"{RUN_ID}/2026.10.08/checkpoints/step-412290"
)
CHECKPOINT_METADATA_RAW_SHA256 = (
    "f6028295e3790bf43606917a78e5dade8ce0eb12e19dd2a8e3cab886ae86ab34"
)
CHECKPOINT_METADATA_DIGEST = (
    "275e68e8a521333a4dd5c7bf2aa8f9c79c11517677a8af47ce7abc1aa5004a89"
)
TOKENIZER_SHA256 = "d1cf7a63f3e0e0778a7ef137020e6f39d04be1e6da14fe22d6283eb36cddae5e"

FIXTURE_SOURCE = (
    "s3://marin-us-east-02a/MarinDNA/exp582_plantcad2_moe/evaluations/"
    "final-v1/inputs/tasks/conservation_within_andropogoneae/"
    "test-00000-of-00001.parquet"
)
FIXTURE_SOURCE_SHA256 = (
    "b390611a3332e35f3b8d18f30d4d79954926cfe6abe17da3333794335d87223b"
)
FIXTURE_SOURCE_ROWS = 38_060
FIXTURE_MANIFEST_SHA256 = (
    "f2f83a2803f8f43f154abd2eac18a02ea34e0ec2a41138a5e7463e3e5b3778e8"
)

INPUT_ALPHABET = "ACGTN"
OUTPUT_ALPHABET = "ACGT"
TOTAL_PARAMETERS = 1_591_848_448
ACTIVE_PARAMETERS = 60_555_264
WINDOW_BASES = 8192


class OutputMode(StrEnum):
    """Inference outputs with distinct normalization costs."""

    ACGT_LOGITS = "acgt_logits"
    FULL_VOCAB_TARGET_LOG_PROBS = "full_vocab_target_log_probs"


@dataclass(frozen=True)
class TokenContract:
    """IDs retained from the pretrained language tokenizer."""

    input_ids: dict[str, int]
    output_ids: dict[str, int]


def token_contract(tokenizer: Tokenizer) -> TokenContract:
    """Resolve isolated character IDs and reject hidden BPE or special tokens."""
    encoded: dict[str, int] = {}
    for base in INPUT_ALPHABET:
        result = tokenizer.encode(base, add_special_tokens=False)
        if len(result.ids) != 1:
            raise ValueError(f"{base} did not encode to exactly one token")
        encoded[base] = result.ids[0]
    combined = tokenizer.encode(INPUT_ALPHABET, add_special_tokens=False)
    if combined.ids != [encoded[base] for base in INPUT_ALPHABET]:
        raise ValueError("Combined DNA input used BPE merges or inserted tokens")
    if len(set(encoded.values())) != len(encoded):
        raise ValueError("DNA characters do not have distinct token IDs")
    return TokenContract(
        input_ids=encoded,
        output_ids={base: encoded[base] for base in OUTPUT_ALPHABET},
    )


def validate_sequences(sequences: list[str]) -> None:
    """Enforce the exact benchmark workload before any backend sees it."""
    if not sequences:
        raise ValueError("The benchmark fixture is empty")
    if len(set(sequences)) != len(sequences):
        raise ValueError("Benchmark sequences must be unique")
    allowed = set(INPUT_ALPHABET)
    for index, sequence in enumerate(sequences):
        if len(sequence) != WINDOW_BASES:
            raise ValueError(f"Sequence {index} is not {WINDOW_BASES} bases")
        if set(sequence) - allowed:
            raise ValueError(f"Sequence {index} contains a character outside ACGTN")

