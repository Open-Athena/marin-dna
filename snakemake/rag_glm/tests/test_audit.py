"""Tests for the issue #402 species audit."""

import polars as pl
from marin_dna_rag_glm.audit import (
    pairwise_identity_table,
    projection_audit_table,
)
from marin_dna_rag_glm.dataset import (
    BASES_PER_SLOT,
    PROVISIONAL_SPECIES_ORDER,
)


def test_projection_audit_counts_missing_and_ambiguous_sequences() -> None:
    anchors = ["a", "b", "c"]
    rows = pl.DataFrame(
        [
            {"query_name": "a", "species": "sp1", "sequence": "A" * BASES_PER_SLOT},
            {"query_name": "b", "species": "sp1", "sequence": "N" + "A" * 254},
            {"query_name": "a", "species": "sp2", "sequence": "C" * BASES_PER_SLOT},
        ]
    )
    audit = projection_audit_table(
        rows, sample_anchor_ids=anchors, species=["sp1", "sp2", "sp3"]
    ).sort("species")
    assert audit["n_projected"].to_list() == [2, 1, 0]
    assert audit["n_ambiguous_windows"].to_list() == [1, 0, 0]
    assert audit["n_ambiguous_bases"].to_list() == [1, 0, 0]
    assert audit["projection_success"].to_list() == [2 / 3, 1 / 3, 0.0]


def test_pairwise_identity_excludes_ambiguous_bases() -> None:
    species_a, species_b = PROVISIONAL_SPECIES_ORDER[:2]
    rows = pl.DataFrame(
        [
            {
                "query_name": "a",
                "species": species_a,
                "sequence": "A" * 254 + "N",
            },
            {
                "query_name": "a",
                "species": species_b,
                "sequence": "A" * 253 + "CN",
            },
        ]
    )
    identity = pairwise_identity_table(
        rows,
        species_order=(
            species_a,
            species_b,
            *PROVISIONAL_SPECIES_ORDER[2:],
        ),
    )
    pair = identity.filter(
        (pl.col("species_a") == species_a) & (pl.col("species_b") == species_b)
    ).row(0, named=True)
    assert pair["n_shared_anchors"] == 1
    assert pair["n_compared_bases"] == 254
    assert pair["same_position_identity"] == 253 / 254
