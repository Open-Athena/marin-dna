"""Tests for fixed-layout RAG document construction."""

import polars as pl
import pytest

from marin_dna.pipelines.rag_glm.dataset import (
    BASES_PER_SLOT,
    DOCUMENT_TOKENS,
    DOCUMENT_TOKENS_WITHOUT_CLS,
    HUMAN_SEGMENT_START,
    HUMAN_VARIANT_TOKEN_INDEX,
    MISSING_SEQUENCE,
    PROVISIONAL_SPECIES_ORDER,
    SEQUENCE_BOUNDARY,
    add_document_reverse_complements,
    assemble_document,
    assemble_fixed_layout_documents,
    reverse_complement_document_slots,
    split_training_validation,
    stable_anchor_rank,
)


def _sequence(base: str) -> str:
    return base * BASES_PER_SLOT


def _projection_row(
    anchor_id: str,
    species: str,
    sequence: str,
    *,
    chrom: str,
    start: int,
) -> dict:
    return {
        "query_name": anchor_id,
        "species": species,
        "t_chrom": f"chr{chrom}",
        "t_start": start,
        "t_end": start + BASES_PER_SLOT,
        "t_strand": "+",
        "t_src_size": 1_000_000,
        "sequence": sequence,
    }


def _projection_frame(n_chr18: int = 3) -> pl.DataFrame:
    rows: list[dict] = []
    anchors = [
        ("win_1_000000001", "1", 100),
        ("win_2_000000001", "2", 200),
        *[
            (f"win_18_{index:09d}", "18", 1_000 + index * 128)
            for index in range(n_chr18)
        ],
    ]
    for anchor_id, chrom, start in anchors:
        for slot, species in enumerate(PROVISIONAL_SPECIES_ORDER):
            if species == "Tolypeutes_matacus" and chrom == "2":
                continue
            rows.append(
                _projection_row(
                    anchor_id,
                    species,
                    _sequence("ACGT"[slot % 4]),
                    chrom=chrom,
                    start=start,
                )
            )
    return pl.DataFrame(rows)


def test_token_accounting_constants() -> None:
    assert DOCUMENT_TOKENS_WITHOUT_CLS == 2_047
    assert DOCUMENT_TOKENS == 2_048
    assert HUMAN_SEGMENT_START == 1_793
    assert HUMAN_VARIANT_TOKEN_INDEX == 1_920


def test_assemble_document_has_exact_layout() -> None:
    slots = tuple(_sequence(base) for base in "ACGTACGT")
    document = assemble_document(slots)
    assert document.count(SEQUENCE_BOUNDARY) == 7
    assert document.split(SEQUENCE_BOUNDARY) == list(slots)


def test_assemble_document_rejects_wrong_window_length() -> None:
    slots = [_sequence("A")] * 8
    slots[3] = "A" * (BASES_PER_SLOT - 1)
    with pytest.raises(AssertionError, match="every sequence"):
        assemble_document(slots)


def test_whole_document_reverse_complement_preserves_slot_order() -> None:
    slots = (
        "A" * BASES_PER_SLOT,
        "C" * BASES_PER_SLOT,
        "G" * BASES_PER_SLOT,
        "T" * BASES_PER_SLOT,
        "N" * BASES_PER_SLOT,
        ("ACG" * 85),
        ("TGC" * 85),
        "A" * BASES_PER_SLOT,
    )
    reverse = reverse_complement_document_slots(slots)
    assert reverse[0] == "T" * BASES_PER_SLOT
    assert reverse[4] == "N" * BASES_PER_SLOT
    assert reverse_complement_document_slots(reverse) == slots


def test_assemble_fixed_layout_fills_missing_slot_and_keeps_human_final() -> None:
    documents = assemble_fixed_layout_documents(_projection_frame())
    row = documents.filter(pl.col("anchor_id") == "win_2_000000001").row(0, named=True)
    assert row["chrom"] == "2"
    assert row["start"] == 200
    assert row["end"] == 455
    assert row["available_2"] is False
    assert row["quality_pass_2"] is False
    assert row["sequence_2"] == MISSING_SEQUENCE
    assert row["available_7"] is True
    assert row["sequence_7"] == _sequence("T")
    assert row["seq"].split(SEQUENCE_BOUNDARY)[7] == row["sequence_7"]


def test_assemble_fixed_layout_rejects_duplicate_species_rows() -> None:
    rows = _projection_frame()
    duplicate = rows.row(0, named=True)
    with pytest.raises(AssertionError, match="at most one canonical row"):
        assemble_fixed_layout_documents(pl.concat([rows, pl.DataFrame([duplicate])]))


def test_split_is_chromosome_disjoint_and_deterministic() -> None:
    documents = assemble_fixed_layout_documents(_projection_frame(n_chr18=6))
    train_a, validation_a = split_training_validation(
        documents, validation_size=3, validation_seed=42
    )
    train_b, validation_b = split_training_validation(
        documents, validation_size=3, validation_seed=42
    )
    assert set(train_a["chrom"]) == {"1", "2"}
    assert set(validation_a["chrom"]) == {"18"}
    assert validation_a["anchor_id"].to_list() == validation_b["anchor_id"].to_list()
    assert set(validation_a["augmentation"]) == {"+"}


def test_stable_anchor_rank_changes_with_seed() -> None:
    anchor = "win_18_000000001"
    assert stable_anchor_rank(anchor, 42) == stable_anchor_rank(anchor, 42)
    assert stable_anchor_rank(anchor, 42) != stable_anchor_rank(anchor, 43)


def test_reverse_complement_augmentation_is_document_wide_and_involutive() -> None:
    documents = assemble_fixed_layout_documents(_projection_frame())
    training = documents.filter(pl.col("chrom") != "18")
    augmented = add_document_reverse_complements(training)
    assert augmented.height == 2 * training.height

    anchor_id = "win_1_000000001"
    forward = augmented.filter(
        (pl.col("anchor_id") == anchor_id) & (pl.col("augmentation") == "+")
    ).row(0, named=True)
    reverse = augmented.filter(
        (pl.col("anchor_id") == anchor_id) & (pl.col("augmentation") == "-")
    ).row(0, named=True)
    reverse_slots = tuple(reverse[f"sequence_{slot}"] for slot in range(8))
    assert reverse_slots == reverse_complement_document_slots(
        tuple(forward[f"sequence_{slot}"] for slot in range(8))
    )
    assert reverse["seq"] == assemble_document(reverse_slots)
