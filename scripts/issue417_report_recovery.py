#!/usr/bin/env python3
"""Report issue #417 rejection and backend-recovery rates from preserved QC."""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl


REGION_LABELS = [
    ("cds", "CDS"),
    ("utr3", "3′ UTR"),
    ("ncrna_exon", "ncRNA exon"),
    ("tss_region_and_utr5", "TSS + 5′ UTR"),
    ("ccre_non_promoter", "non-promoter cCRE"),
    ("background", "background"),
]


def _count(frame: pl.DataFrame) -> int:
    value = frame["count"].sum()
    return 0 if value is None else int(value)


def _rejection_split(rejections: pl.DataFrame) -> tuple[int, int]:
    explicit = _count(rejections.filter(pl.col("rejection_reason") != "no_mapping"))
    unmapped = _count(rejections.filter(pl.col("rejection_reason") == "no_mapping"))
    return explicit, unmapped


def _recovery_row(
    anchors: pl.DataFrame,
    *,
    label: str,
) -> dict[str, int | float | str]:
    assert anchors.height > 0
    requested_mammals = anchors["requested_mammal_species"].unique().to_list()
    requested_non_mammals = anchors["requested_non_mammal_species"].unique().to_list()
    assert len(requested_mammals) == len(requested_non_mammals) == 1
    mammal_denominator = anchors.height * int(requested_mammals[0])
    non_mammal_denominator = anchors.height * int(requested_non_mammals[0])
    accepted_mammals = int(anchors["accepted_mammal_projections"].sum())
    accepted_non_mammals = int(anchors["accepted_non_mammal_projections"].sum())
    return {
        "label": label,
        "anchors": anchors.height,
        "accepted_mammals": accepted_mammals,
        "requested_mammals": mammal_denominator,
        "mammal_percent": 100 * accepted_mammals / mammal_denominator,
        "accepted_non_mammals": accepted_non_mammals,
        "requested_non_mammals": non_mammal_denominator,
        "non_mammal_percent": 100 * accepted_non_mammals / non_mammal_denominator,
    }


def _format_fraction(accepted: int, requested: int, percent: float) -> str:
    return f"{accepted:,} / {requested:,} = {percent:.2f}%"


def render_report(
    full_per_anchor_path: Path,
    full_rejections_path: Path,
    zrs_per_anchor_path: Path,
    zrs_rejections_path: Path,
) -> str:
    """Render a Markdown breakdown with fail-fast accounting checks."""
    full = pl.read_parquet(full_per_anchor_path)
    rejections = pl.read_parquet(full_rejections_path)
    assert full["query_name"].n_unique() == full.height

    requested_total_values = full["requested_total_species"].unique().to_list()
    assert len(requested_total_values) == 1
    requested_total = full.height * int(requested_total_values[0])
    accepted_total = int(full["accepted_total_projections"].sum())
    explicit_rejections, unmapped = _rejection_split(rejections)
    assert accepted_total + explicit_rejections + unmapped == requested_total

    recovery = [_recovery_row(full, label="all")]
    for region_label, display_label in REGION_LABELS:
        recovery.append(
            _recovery_row(
                full.filter(pl.col("region_label") == region_label),
                label=display_label,
            )
        )

    reason_counts = (
        rejections.filter(pl.col("rejection_reason") != "no_mapping")
        .group_by("rejection_reason")
        .agg(pl.col("count").sum())
        .sort("count", descending=True)
    )

    zrs = pl.read_parquet(zrs_per_anchor_path).filter(
        pl.col("query_name").str.starts_with("zrs_")
    )
    zrs_rejections = pl.read_parquet(zrs_rejections_path).filter(
        pl.col("query_name").str.starts_with("zrs_")
    )
    assert zrs.height == 2
    assert zrs["requested_mammal_species"].unique().to_list() == [2]
    assert zrs["requested_non_mammal_species"].unique().to_list() == [5]

    lines = [
        "## Projection accounting",
        "",
        f"- accepted: {accepted_total:,} / {requested_total:,} "
        f"({100 * accepted_total / requested_total:.2f}%)",
        f"- explicitly rejected: {explicit_rejections:,} "
        f"({100 * explicit_rejections / requested_total:.2f}%)",
        f"- no mapping: {unmapped:,} ({100 * unmapped / requested_total:.2f}%)",
        "",
        "Explicit rejection reasons:",
        "",
    ]
    lines.extend(
        f"- `{reason}`: {int(count):,}" for reason, count in reason_counts.iter_rows()
    )
    lines.extend(
        [
            "",
            "## Recovery by anchor cohort",
            "",
            "| Cohort | Anchors | Mammals (107 requested/anchor) | "
            "Non-mammalian vertebrates (28 requested/anchor) |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in recovery:
        lines.append(
            f"| {row['label']} | {row['anchors']:,} | "
            f"{_format_fraction(int(row['accepted_mammals']), int(row['requested_mammals']), float(row['mammal_percent']))} | "
            f"{_format_fraction(int(row['accepted_non_mammals']), int(row['requested_non_mammals']), float(row['non_mammal_percent']))} |"
        )
    lines.extend(
        [
            "",
            "`cds_mammals_only` uses the CDS anchor row above but excludes "
            "non-mammals by definition.",
            "",
            "## ZRS smoke-QC sidecar",
            "",
            "The ZRS sidecar requested two mammal and five non-mammal controls, "
            "not the full 107+28 target cohort.",
            "",
            "| Locus | Mammals | Non-mammals | Explicit rejects | No mapping |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in zrs.sort("query_name").iter_rows(named=True):
        name = str(row["query_name"])
        locus_rejections = zrs_rejections.filter(pl.col("query_name") == name)
        explicit, no_mapping = _rejection_split(locus_rejections)
        accepted_mammals = int(row["accepted_mammal_projections"])
        requested_mammals = int(row["requested_mammal_species"])
        accepted_non_mammals = int(row["accepted_non_mammal_projections"])
        requested_non_mammals = int(row["requested_non_mammal_species"])
        assert (
            accepted_mammals + accepted_non_mammals + explicit + no_mapping
            == requested_mammals + requested_non_mammals
        )
        lines.append(
            f"| `{name}` | "
            f"{_format_fraction(accepted_mammals, requested_mammals, 100 * accepted_mammals / requested_mammals)} | "
            f"{_format_fraction(accepted_non_mammals, requested_non_mammals, 100 * accepted_non_mammals / requested_non_mammals)} | "
            f"{explicit:,} | {no_mapping:,} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-per-anchor", type=Path, required=True)
    parser.add_argument("--full-rejections", type=Path, required=True)
    parser.add_argument("--zrs-per-anchor", type=Path, required=True)
    parser.add_argument("--zrs-rejections", type=Path, required=True)
    args = parser.parse_args()
    print(
        render_report(
            args.full_per_anchor,
            args.full_rejections,
            args.zrs_per_anchor,
            args.zrs_rejections,
        ),
        end="",
    )


if __name__ == "__main__":
    main()
