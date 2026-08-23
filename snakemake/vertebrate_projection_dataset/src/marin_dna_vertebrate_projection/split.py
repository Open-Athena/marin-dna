"""Deterministic uniform row-random validation selection."""

import hashlib

import polars as pl

VALIDATION_IDENTITY_COLUMNS = (
    "query_name",
    "source_chrom",
    "source_start",
    "source_end",
    "species",
    "alignment_source",
)
_SELECTION_HASH = "selection_hash"


def _identity_value(row: dict[str, object]) -> str:
    return "\t".join(str(row[column]) for column in VALIDATION_IDENTITY_COLUMNS)


def _stable_digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def validation_identity_expression() -> pl.Expr:
    """Return the stable biological row identity used at the split boundary."""
    return pl.concat_str(
        [pl.col(column).cast(pl.String) for column in VALIDATION_IDENTITY_COLUMNS],
        separator="\t",
    )


def validation_selection_hash(seed: int) -> pl.Expr:
    """Return the seeded uniform hash used to rank original-orientation rows."""
    return validation_identity_expression().hash(seed=seed).alias(_SELECTION_HASH)


def select_uniform_validation_rows(
    rows: pl.DataFrame | pl.LazyFrame,
    *,
    validation_rows: int,
    seed: int,
) -> pl.DataFrame:
    """Select an order-invariant uniform sample without replacement.

    Selection is row-level. Species projections from the same human anchor may
    fall on opposite sides of the split. The input must contain only original-
    orientation rows because validation selection precedes augmentation.
    """
    assert validation_rows > 0
    lazy = rows.lazy() if isinstance(rows, pl.DataFrame) else rows
    schema = lazy.collect_schema()
    missing = set(VALIDATION_IDENTITY_COLUMNS) - set(schema.names())
    assert not missing, f"dataset rows missing identity columns: {sorted(missing)}"
    assert "augmentation" not in schema, (
        "validation selection must precede reverse-complement augmentation"
    )

    selected = (
        lazy.select(
            *VALIDATION_IDENTITY_COLUMNS,
            validation_selection_hash(seed),
        )
        .bottom_k(
            validation_rows,
            by=[_SELECTION_HASH, *VALIDATION_IDENTITY_COLUMNS],
        )
        .collect(engine="streaming")
        .sort(_SELECTION_HASH, *VALIDATION_IDENTITY_COLUMNS)
    )
    assert selected.height == validation_rows, (
        f"validation requires {validation_rows} rows, found {selected.height}"
    )
    assert selected.select(*VALIDATION_IDENTITY_COLUMNS).is_unique().all(), (
        "biological row identity is not unique"
    )

    identities = [_identity_value(row) for row in selected.to_dicts()]
    return (
        selected.with_columns(
            pl.Series(
                "row_id",
                [_stable_digest(identity)[:24] for identity in identities],
            ),
            pl.Series(
                "selection_digest",
                [_stable_digest(f"{seed}\t{identity}") for identity in identities],
            ),
            pl.lit(seed, dtype=pl.Int64).alias("seed"),
        )
        .with_row_index("selection_rank", offset=1)
        .select(
            "row_id",
            *VALIDATION_IDENTITY_COLUMNS,
            "seed",
            "selection_rank",
            _SELECTION_HASH,
            "selection_digest",
        )
    )
