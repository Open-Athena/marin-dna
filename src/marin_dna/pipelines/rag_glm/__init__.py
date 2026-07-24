"""Issue #402 retrieval-augmented genomic language-model prototype."""

from marin_dna.pipelines.rag_glm.dataset import (
    BASES_PER_SLOT,
    DOCUMENT_TOKENS,
    HUMAN_SEGMENT_START,
    HUMAN_VARIANT_TOKEN_INDEX,
    PROVISIONAL_SPECIES_ORDER,
    SEQUENCE_BOUNDARY,
    SPECIES_ORDER_VERSION,
)

__all__ = [
    "BASES_PER_SLOT",
    "DOCUMENT_TOKENS",
    "HUMAN_SEGMENT_START",
    "HUMAN_VARIANT_TOKEN_INDEX",
    "PROVISIONAL_SPECIES_ORDER",
    "SEQUENCE_BOUNDARY",
    "SPECIES_ORDER_VERSION",
]
