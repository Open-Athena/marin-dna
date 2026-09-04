"""Deterministic human-review artifacts for functional anchors."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from marin_dna_vertebrate_projection.functional_anchors import FUNCTIONAL_ARMS


def _deterministic_head(frame: pl.DataFrame, count: int, seed: int) -> pl.DataFrame:
    if frame.height <= count:
        return frame
    return (
        frame.with_columns(pl.col("query_name").hash(seed=seed).alias("_sample_hash"))
        .sort("_sample_hash", "query_name")
        .head(count)
        .drop("_sample_hash")
    )


def write_preprojection_review(
    projection_path: str | Path,
    training_path: str | Path,
    deferred_path: str | Path,
    sample_path: str | Path,
    report_path: str | Path,
    *,
    rows_per_arm_and_band: int = 3,
    seed: int = 517,
) -> None:
    """Write a deterministic preprojection sample and pending review checklist."""
    if rows_per_arm_and_band <= 0:
        raise ValueError("rows_per_arm_and_band must be positive")
    training = pl.read_parquet(training_path).with_columns(
        pl.lit("training_ge_0.20").alias("conservation_band")
    )
    deferred = pl.read_parquet(deferred_path).with_columns(
        pl.lit("deferred_0.10_to_0.20").alias("conservation_band")
    )
    projection_names = set(pl.read_parquet(projection_path)["query_name"])
    assert set(training["query_name"]) | set(deferred["query_name"]) == projection_names
    samples: list[pl.DataFrame] = []
    for arm in FUNCTIONAL_ARMS:
        for frame in [training, deferred]:
            arm_rows = frame.filter(pl.col("source_arm") == arm)
            if not arm_rows.is_empty():
                samples.append(
                    _deterministic_head(arm_rows, rows_per_arm_and_band, seed)
                )
    if not samples:
        raise AssertionError("preprojection review sample is empty")
    sample = pl.concat(samples, how="vertical").sort(
        "source_arm", "conservation_band", "query_name"
    )
    for path in [sample_path, report_path]:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    sample.write_csv(sample_path, separator="\t")
    count_lines = []
    for arm in FUNCTIONAL_ARMS:
        count_lines.append(
            f"| {arm} | "
            f"{training.filter(pl.col('source_arm') == arm).height} | "
            f"{deferred.filter(pl.col('source_arm') == arm).height} |"
        )
    Path(report_path).write_text(
        "# Functional-anchor preprojection review\n\n"
        "Status: **pending human review**.\n\n"
        "The annotation source is the complete Ensembl GRCh38 release 115 GTF; "
        "RefSeq and canonical-transcript-only filtering are not used.\n\n"
        "All reported coordinates are 0-based and half-open.\n\n"
        "| Arm | Training anchors (>=0.20) | Deferred anchors ([0.10, 0.20)) |\n"
        "|---|---:|---:|\n" + "\n".join(count_lines) + "\n\n"
        "Review `preprojection_sample.tsv` in a genome browser and reconcile "
        "the tabular audit artifacts before approving projection.\n\n"
        "- [ ] Confirm human composition and conservation quantiles by arm.\n"
        "- [ ] Confirm chromosome, construction-loss, and ownership-loss tables.\n"
        "- [ ] Confirm development overlap uses only odd autosomes/X, excludes "
        "complete mature-miRNA match groups, and converts 1-based VEP positions "
        "at the boundary.\n"
        "- [ ] Confirm CDS, 3′ UTR, TSS/5′ UTR, ncRNA, and enhancer identities.\n"
        "- [ ] Confirm centered dELS/pELS windows have no annotated exon overlap.\n"
        "- [ ] Confirm priority ownership agrees with visible overlaps.\n"
        "- [ ] Record exclusions or approve the smoke projection in issue #517.\n"
    )
