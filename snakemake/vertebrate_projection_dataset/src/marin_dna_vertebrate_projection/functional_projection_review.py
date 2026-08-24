"""Bounded deterministic projection review for the five issue #517 arms."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from marin_dna_vertebrate_projection.functional_anchors import FUNCTIONAL_ARMS


def _identity_hash(seed: int) -> pl.Expr:
    return pl.concat_str(
        [
            pl.col("query_name"),
            pl.col("species"),
            pl.col("alignment_source"),
            pl.col("t_chrom"),
            pl.col("t_start").cast(pl.String),
        ],
        separator="\t",
    ).hash(seed=seed)


def write_functional_projection_review(
    sequences_path: str | Path,
    rejected_paths: list[str],
    sample_path: str | Path,
    rejected_sample_path: str | Path,
    report_path: str | Path,
    *,
    seed: int,
    rows_per_arm: int,
    fragmented_rows: int,
    rejected_rows_per_reason: int,
) -> None:
    """Write accepted/rejected samples without loading the full projection table."""
    if rows_per_arm <= 0 or fragmented_rows < 0 or rejected_rows_per_reason <= 0:
        raise ValueError("invalid functional inspection sample size")
    sequences = pl.scan_parquet(sequences_path)
    samples: list[pl.DataFrame] = []
    for arm in FUNCTIONAL_ARMS:
        arm_sample = (
            sequences.filter(pl.col("region_label") == arm)
            .with_columns(_identity_hash(seed).alias("_sample_hash"))
            .bottom_k(rows_per_arm, by="_sample_hash")
            .drop("_sample_hash")
            .collect(engine="streaming")
        )
        if arm_sample.is_empty():
            raise AssertionError(f"projection inspection lacks recovered {arm} rows")
        samples.append(
            arm_sample.with_columns(pl.lit(f"stable_{arm}").alias("review_reason"))
        )
    if fragmented_rows:
        fragmented = (
            sequences.filter(pl.col("fragment_count") > 1)
            .with_columns(_identity_hash(seed).alias("_sample_hash"))
            .bottom_k(fragmented_rows, by="_sample_hash")
            .drop("_sample_hash")
            .collect(engine="streaming")
            .with_columns(pl.lit("fragmented_mapping").alias("review_reason"))
        )
        if not fragmented.is_empty():
            samples.append(fragmented)
    accepted_sample = (
        pl.concat(samples, how="vertical")
        .unique(subset=["query_name", "species"])
        .sort("region_label", "query_name", "species")
    )
    if not (accepted_sample["sequence"].str.len_bytes() == 255).all():
        raise AssertionError("inspection sample contains a non-255-bp sequence")

    rejected = pl.concat(
        [pl.scan_parquet(path) for path in rejected_paths], how="vertical"
    )
    reasons = (
        rejected.select("rejection_reason")
        .unique()
        .collect(engine="streaming")["rejection_reason"]
        .sort()
        .to_list()
    )
    rejected_samples = []
    for reason in reasons:
        rejected_samples.append(
            rejected.filter(pl.col("rejection_reason") == reason)
            .with_columns(
                pl.concat_str([pl.col("query_name"), pl.col("species")], separator="\t")
                .hash(seed=seed)
                .alias("_sample_hash")
            )
            .sort("fragment_count", "_sample_hash", descending=[True, False])
            .limit(rejected_rows_per_reason)
            .drop("_sample_hash")
            .collect(engine="streaming")
        )
    rejected_sample = (
        pl.concat(rejected_samples, how="vertical")
        if rejected_samples
        else pl.DataFrame(schema=pl.read_parquet_schema(rejected_paths[0]))
    )
    for path in [sample_path, rejected_sample_path, report_path]:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    accepted_sample.write_csv(sample_path, separator="\t")
    rejected_sample.write_csv(rejected_sample_path, separator="\t")
    arm_lines = [
        f"| {arm} | {accepted_sample.filter(pl.col('region_label') == arm).height} |"
        for arm in FUNCTIONAL_ARMS
    ]
    Path(report_path).write_text(
        "# Functional projection inspection\n\n"
        "Status: **pending human review**.\n\n"
        "| Arm | Sampled accepted rows |\n"
        "|---|---:|\n" + "\n".join(arm_lines) + "\n\n"
        f"Sampled rejection rows: {rejected_sample.height}.\n\n"
        "- [ ] Check source and projected intervals in the relevant genome browsers.\n"
        "- [ ] Check negative-strand sequence orientation and center-base placement.\n"
        "- [ ] Review every sampled explicit rejection reason.\n"
        "- [ ] Record approval or exclusions in issue #517 before publication.\n"
    )
