"""Deterministic anchor sampling for the issue #473 projection-policy pilot."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import polars as pl

from marin_dna_vertebrate_projection.issue_473.policy import ANCHOR_COLUMNS

CONSERVATION_SCORE_COLUMN = "proportion_conserved"
PILOT_STRATUM_COLUMNS = (
    "region_label",
    "source_chrom",
    "conservation_quantile",
)


@dataclass(frozen=True)
class PilotSample:
    """Selected anchors and audit tables for one stratified pilot sample."""

    anchors: pl.DataFrame
    selection_manifest: pl.DataFrame
    stratum_counts: pl.DataFrame


def build_scored_anchor_catalog(
    labels: pl.DataFrame,
    scored_windows: pl.DataFrame,
    *,
    min_proportion_conserved: float,
    target_length: int = 255,
) -> pl.DataFrame:
    """Join retained anchor labels to their original conservation scores."""
    assert 0.0 <= min_proportion_conserved <= 1.0
    assert target_length > 0
    label_columns = {"name", "chrom", "start", "end", "label"}
    score_columns = {"name", CONSERVATION_SCORE_COLUMN}
    missing_labels = label_columns - set(labels.columns)
    missing_scores = score_columns - set(scored_windows.columns)
    assert not missing_labels, f"labels missing columns: {sorted(missing_labels)}"
    assert not missing_scores, (
        f"scored windows missing columns: {sorted(missing_scores)}"
    )
    assert labels["name"].n_unique() == labels.height
    assert scored_windows["name"].n_unique() == scored_windows.height

    scores = scored_windows.select("name", CONSERVATION_SCORE_COLUMN)
    assert scores[CONSERVATION_SCORE_COLUMN].is_between(0.0, 1.0).all()
    catalog = (
        labels.join(scores, on="name", how="left", validate="1:1")
        .select(
            pl.col("name").alias("query_name"),
            (pl.lit("chr") + pl.col("chrom").cast(pl.String)).alias("source_chrom"),
            pl.col("start").cast(pl.Int64).alias("source_start"),
            pl.col("end").cast(pl.Int64).alias("source_end"),
            pl.col("label").alias("region_label"),
            pl.col(CONSERVATION_SCORE_COLUMN),
        )
        .sort("source_chrom", "source_start", "query_name")
    )
    assert catalog[CONSERVATION_SCORE_COLUMN].null_count() == 0
    assert (catalog[CONSERVATION_SCORE_COLUMN] >= min_proportion_conserved).all()
    assert catalog["query_name"].n_unique() == catalog.height
    assert (catalog["source_start"] >= 0).all()
    assert (catalog["source_end"] - catalog["source_start"] == target_length).all()
    return catalog


def _selection_digest(seed: int, query_name: str) -> str:
    return hashlib.sha256(f"{seed}\t{query_name}".encode()).hexdigest()


def _selection_key(seed: int, query_name: str) -> int:
    return int.from_bytes(
        hashlib.sha256(f"{seed}\t{query_name}".encode()).digest()[:8], "big"
    )


def _allocate_stratum_quotas(
    availability: dict[tuple[str, int], int],
    max_rows: int,
) -> dict[tuple[str, int], int]:
    """Water-fill quotas after assigning one row to every observed stratum."""
    assert max_rows >= 0
    if not availability:
        return {}
    strata = sorted(availability)
    assert all(availability[stratum] > 0 for stratum in strata)
    assert max_rows >= len(strata), (
        f"regional cap {max_rows} cannot represent {len(strata)} strata"
    )
    target = min(max_rows, sum(availability.values()))
    quotas = {stratum: 1 for stratum in strata}
    remaining = target - len(strata)
    while remaining > 0:
        eligible = [
            stratum for stratum in strata if quotas[stratum] < availability[stratum]
        ]
        assert eligible, "quota allocation exhausted availability too early"
        for stratum in eligible:
            if remaining == 0:
                break
            quotas[stratum] += 1
            remaining -= 1
    return quotas


def sample_projection_pilot_anchors(
    anchors: pl.DataFrame,
    *,
    regions: tuple[str, ...],
    max_per_region: int = 10_000,
    conservation_quantiles: int = 5,
    seed: int = 473,
) -> PilotSample:
    """Sample each functional region across chromosome and score quantile.

    Conservation quantiles are equal-count bins computed within each region,
    with score and ``query_name`` providing deterministic tie-breaking. The
    sampler first represents every observed chromosome-by-quantile stratum,
    then water-fills the remaining regional budget as evenly as availability
    permits. Within each stratum, SHA-256 ordering selects rows reproducibly.
    """
    assert regions and len(set(regions)) == len(regions)
    assert max_per_region > 0
    assert conservation_quantiles > 0
    required = {*ANCHOR_COLUMNS, CONSERVATION_SCORE_COLUMN}
    missing = required - set(anchors.columns)
    assert not missing, f"anchors missing columns: {sorted(missing)}"
    assert anchors["query_name"].n_unique() == anchors.height
    assert anchors[CONSERVATION_SCORE_COLUMN].is_between(0.0, 1.0).all()

    candidates = anchors.filter(pl.col("region_label").is_in(regions))
    observed_regions = set(candidates["region_label"].unique().to_list())
    assert observed_regions == set(regions), (
        f"missing requested regions: {sorted(set(regions) - observed_regions)}"
    )

    quantified: list[pl.DataFrame] = []
    for region in regions:
        region_rows = candidates.filter(pl.col("region_label") == region).sort(
            CONSERVATION_SCORE_COLUMN, "query_name"
        )
        effective_quantiles = min(conservation_quantiles, region_rows.height)
        quantile = [
            (index * effective_quantiles) // region_rows.height + 1
            for index in range(region_rows.height)
        ]
        quantified.append(
            region_rows.with_columns(
                pl.Series("conservation_quantile", quantile, dtype=pl.Int64)
            )
        )
    with_quantiles = pl.concat(quantified)

    count_rows: list[dict[str, object]] = []
    for region in regions:
        availability = {
            (str(chrom), int(quantile)): int(count)
            for chrom, quantile, count in (
                with_quantiles.filter(pl.col("region_label") == region)
                .group_by("source_chrom", "conservation_quantile")
                .len()
                .iter_rows()
            )
        }
        quotas = _allocate_stratum_quotas(availability, max_per_region)
        for (chrom, quantile), eligible in sorted(availability.items()):
            count_rows.append(
                {
                    "region_label": region,
                    "source_chrom": chrom,
                    "conservation_quantile": quantile,
                    "eligible_anchors": eligible,
                    "selected_anchors": quotas[(chrom, quantile)],
                }
            )
    stratum_counts = pl.DataFrame(count_rows).sort(*PILOT_STRATUM_COLUMNS)

    ranked = (
        with_quantiles.with_columns(
            pl.col("query_name")
            .map_elements(
                lambda name: _selection_key(seed, str(name)),
                return_dtype=pl.UInt64,
            )
            .alias("selection_key")
        )
        .sort(*PILOT_STRATUM_COLUMNS, "selection_key", "query_name")
        .with_columns(
            pl.col("query_name")
            .cum_count()
            .over(*PILOT_STRATUM_COLUMNS)
            .cast(pl.Int64)
            .alias("stratum_rank")
        )
        .join(
            stratum_counts.select(
                *PILOT_STRATUM_COLUMNS,
                pl.col("selected_anchors").alias("stratum_quota"),
            ),
            on=PILOT_STRATUM_COLUMNS,
            how="left",
            validate="m:1",
        )
    )
    selected = ranked.filter(pl.col("stratum_rank") <= pl.col("stratum_quota"))
    selected = selected.with_columns(
        pl.col("query_name")
        .map_elements(
            lambda name: _selection_digest(seed, str(name)),
            return_dtype=pl.String,
        )
        .alias("selection_digest")
    )
    anchor_columns = [
        *ANCHOR_COLUMNS,
        CONSERVATION_SCORE_COLUMN,
        "conservation_quantile",
    ]
    sampled_anchors = selected.select(anchor_columns).sort(
        "region_label", "source_chrom", "source_start", "query_name"
    )
    manifest = selected.select(
        *anchor_columns,
        pl.lit(seed, dtype=pl.Int64).alias("seed"),
        "stratum_rank",
        "selection_digest",
    ).sort(*PILOT_STRATUM_COLUMNS, "stratum_rank")

    selected_by_region = {
        str(region): int(count)
        for region, count in sampled_anchors.group_by("region_label").len().iter_rows()
    }
    eligible_by_region = {
        str(region): int(count)
        for region, count in candidates.group_by("region_label").len().iter_rows()
    }
    for region in regions:
        assert selected_by_region[region] == min(
            eligible_by_region[region], max_per_region
        )
    assert sampled_anchors["query_name"].n_unique() == sampled_anchors.height
    assert (stratum_counts["selected_anchors"] > 0).all()
    assert stratum_counts["selected_anchors"].sum() == sampled_anchors.height
    return PilotSample(
        anchors=sampled_anchors,
        selection_manifest=manifest,
        stratum_counts=stratum_counts,
    )
