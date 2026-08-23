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
VALIDATION_COMPOSITION_DIMENSIONS = (
    "source_chrom",
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


def validation_selection_hash(seed: int, selection_salt: str) -> pl.Expr:
    """Return the seeded uniform hash used to rank original-orientation rows."""
    return (
        pl.concat_str(
            pl.lit(selection_salt), validation_identity_expression(), separator="\t"
        )
        .hash(seed=seed)
        .alias(_SELECTION_HASH)
    )


def select_uniform_validation_rows(
    rows: pl.DataFrame | pl.LazyFrame,
    *,
    validation_rows: int,
    seed: int,
    selection_salt: str,
) -> pl.DataFrame:
    """Select an order-invariant uniform sample without replacement.

    Selection is row-level. Species projections from the same human anchor may
    fall on opposite sides of the split. The input must contain only original-
    orientation rows because validation selection precedes augmentation.
    """
    assert validation_rows > 0
    assert selection_salt
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
            validation_selection_hash(seed, selection_salt),
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
                [
                    _stable_digest(f"{seed}\t{selection_salt}\t{identity}")
                    for identity in identities
                ],
            ),
            pl.lit(seed, dtype=pl.Int64).alias("seed"),
            pl.lit(selection_salt).alias("selection_salt"),
        )
        .with_row_index("selection_rank", offset=1)
        .select(
            "row_id",
            *VALIDATION_IDENTITY_COLUMNS,
            "seed",
            "selection_salt",
            "selection_rank",
            _SELECTION_HASH,
            "selection_digest",
        )
    )


def build_validation_composition(
    eligible_rows: pl.DataFrame | pl.LazyFrame,
    selected_rows: pl.DataFrame | pl.LazyFrame,
) -> pl.DataFrame:
    """Summarize exact eligible and selected counts for split auditing."""
    eligible_lazy = (
        eligible_rows.lazy()
        if isinstance(eligible_rows, pl.DataFrame)
        else eligible_rows
    )
    selected_lazy = (
        selected_rows.lazy()
        if isinstance(selected_rows, pl.DataFrame)
        else selected_rows
    )
    frames: list[pl.DataFrame] = []
    for dimension in VALIDATION_COMPOSITION_DIMENSIONS:
        eligible_counts = (
            eligible_lazy.group_by(dimension)
            .len(name="eligible_rows")
            .collect(engine="streaming")
        )
        selected_counts = (
            selected_lazy.group_by(dimension)
            .len(name="selected_rows")
            .collect(engine="streaming")
        )
        frames.append(
            eligible_counts.join(selected_counts, on=dimension, how="left")
            .with_columns(
                pl.col("selected_rows").fill_null(0),
                pl.lit(dimension).alias("dimension"),
                pl.col(dimension).cast(pl.String).alias("value"),
            )
            .select("dimension", "value", "eligible_rows", "selected_rows")
        )
    return pl.concat(frames).sort("dimension", "value")
