"""Deterministic chromosome-18 train/validation split construction."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import polars as pl


@dataclass(frozen=True)
class DatasetSplits:
    """Split frames plus persisted validation selection metadata."""

    train: pl.DataFrame
    validation: pl.DataFrame
    selection_manifest: pl.DataFrame
    species_counts: pl.DataFrame
    realized_token_count: int


def _stable_digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def add_stable_row_ids(rows: pl.DataFrame) -> pl.DataFrame:
    """Add reproducible row IDs from biological identity fields."""
    required = {
        "query_name",
        "source_chrom",
        "source_start",
        "source_end",
        "species",
        "alignment_source",
    }
    missing = required - set(rows.columns)
    assert not missing, f"rows missing ID columns: {sorted(missing)}"
    with_augmentation = (
        rows
        if "augmentation" in rows.columns
        else rows.with_columns(pl.lit("+").alias("augmentation"))
    )
    identity = [
        "query_name",
        "source_chrom",
        "source_start",
        "source_end",
        "species",
        "alignment_source",
        "augmentation",
    ]
    ids = [
        _stable_digest("\t".join(str(row[column]) for column in identity))[:24]
        for row in with_augmentation.select(identity).to_dicts()
    ]
    result = with_augmentation.with_columns(pl.Series("row_id", ids))
    assert result["row_id"].n_unique() == result.height, (
        "biological row identity is not unique"
    )
    return result


def _allocate_species_quotas(
    availability: dict[str, int], max_rows: int
) -> dict[str, int]:
    """Water-fill quotas so unconstrained species differ by at most one row."""
    assert max_rows >= 0
    if not availability:
        return {}
    species = sorted(availability)
    assert all(availability[name] > 0 for name in species)
    assert max_rows >= len(species), (
        f"validation cap {max_rows} cannot represent {len(species)} species"
    )
    target = min(max_rows, sum(availability.values()))
    quotas = {name: 1 for name in species}
    remaining = target - len(species)
    while remaining > 0:
        eligible = [name for name in species if quotas[name] < availability[name]]
        assert eligible, "quota allocation exhausted availability too early"
        for name in eligible:
            if remaining == 0:
                break
            quotas[name] += 1
            remaining -= 1
    return quotas


def assign_train_validation_splits(
    rows: pl.DataFrame,
    *,
    validation_chrom: str = "chr18",
    max_validation_rows: int = 16_384,
    seed: int = 42,
    bases_per_sequence: int = 255,
) -> DatasetSplits:
    """Apply the source-chromosome split and stratified validation cap.

    All chromosome-18 rows leave training before sampling.  Only original
    orientation rows (``augmentation == '+'``) are eligible for validation;
    unsampled chromosome-18 rows and their reverse complements are discarded.
    """
    assert max_validation_rows > 0
    assert bases_per_sequence > 0
    required = {"query_name", "source_chrom", "species"}
    missing = required - set(rows.columns)
    assert not missing, f"dataset rows missing columns: {sorted(missing)}"
    with_ids = rows if "row_id" in rows.columns else add_stable_row_ids(rows)
    if "augmentation" not in with_ids.columns:
        with_ids = with_ids.with_columns(pl.lit("+").alias("augmentation"))
    assert with_ids["row_id"].n_unique() == with_ids.height
    assert set(with_ids["augmentation"].unique().to_list()) <= {"+", "-"}

    train = with_ids.filter(pl.col("source_chrom") != validation_chrom)
    candidates = with_ids.filter(
        (pl.col("source_chrom") == validation_chrom) & (pl.col("augmentation") == "+")
    )
    availability = {
        str(species): int(count)
        for species, count in candidates.group_by("species").len().iter_rows()
    }
    quotas = _allocate_species_quotas(availability, max_validation_rows)

    selected_rows: list[dict[str, object]] = []
    selection_rows: list[dict[str, object]] = []
    for species in sorted(quotas):
        species_rows = candidates.filter(pl.col("species") == species).to_dicts()
        ranked = sorted(
            species_rows,
            key=lambda row: _stable_digest(f"{seed}\t{row['row_id']}"),
        )
        for rank, row in enumerate(ranked[: quotas[species]], start=1):
            digest = _stable_digest(f"{seed}\t{row['row_id']}")
            selected_rows.append(row)
            selection_rows.append(
                {
                    "row_id": str(row["row_id"]),
                    "query_name": str(row["query_name"]),
                    "species": species,
                    "seed": seed,
                    "species_rank": rank,
                    "selection_digest": digest,
                }
            )

    validation = (
        pl.DataFrame(selected_rows, schema=with_ids.schema)
        if selected_rows
        else pl.DataFrame(schema=with_ids.schema)
    )
    selection_schema = {
        "row_id": pl.String,
        "query_name": pl.String,
        "species": pl.String,
        "seed": pl.Int64,
        "species_rank": pl.Int64,
        "selection_digest": pl.String,
    }
    selection_manifest = (
        pl.DataFrame(selection_rows, schema=selection_schema)
        if selection_rows
        else pl.DataFrame(schema=selection_schema)
    )
    species_counts = pl.DataFrame(
        [
            {
                "species": species,
                "eligible_rows": availability[species],
                "selected_rows": quotas[species],
            }
            for species in sorted(quotas)
        ],
        schema={
            "species": pl.String,
            "eligible_rows": pl.Int64,
            "selected_rows": pl.Int64,
        },
    )

    assert (train["source_chrom"] != validation_chrom).all()
    assert (validation["source_chrom"] == validation_chrom).all()
    assert (validation["augmentation"] == "+").all()
    assert set(train["row_id"].to_list()).isdisjoint(validation["row_id"].to_list())
    assert set(train["query_name"].to_list()).isdisjoint(
        validation["query_name"].to_list()
    )
    assert set(validation["species"].to_list()) == set(availability)
    assert validation.height == min(candidates.height, max_validation_rows)
    if candidates.height >= max_validation_rows:
        assert validation.height == max_validation_rows

    realized_token_count = validation.height * (bases_per_sequence + 1)
    return DatasetSplits(
        train=train.sort("row_id"),
        validation=validation.sort("row_id"),
        selection_manifest=selection_manifest.sort("species", "species_rank"),
        species_counts=species_counts,
        realized_token_count=realized_token_count,
    )
