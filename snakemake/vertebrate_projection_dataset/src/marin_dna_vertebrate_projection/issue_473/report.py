"""Complete post-projection summaries for issue #473.

This module is deliberately separate from the producing workflow. It reads a
complete producer snapshot and summarizes accepted-row geometry and sequence
properties plus exact accepted/rejected/no-mapping accounting without
materializing the requested anchor-by-species grid.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import polars as pl

POLICY_WIDTHS = {"full_window": 255, "center_1": 1}
PAIR_KEYS = ("query_name", "species")


def _require_columns(frame: pl.LazyFrame, required: set[str], label: str) -> None:
    missing = required - set(frame.collect_schema().names())
    assert not missing, f"{label} missing columns: {sorted(missing)}"


def summarize_accepted_rows(
    accepted: pl.LazyFrame,
    *,
    policy: str,
    landmark_width: int,
    target_length: int = 255,
) -> pl.DataFrame:
    """Summarize accepted projection geometry and sequence composition."""
    assert policy in POLICY_WIDTHS
    assert landmark_width == POLICY_WIDTHS[policy]
    required = {
        *PAIR_KEYS,
        "region_label",
        "alignment_source",
        "clade",
        "t_start",
        "t_end",
        "t_strand",
        "pre_resize_t_start",
        "pre_resize_t_end",
        "fragment_count",
        "aligned_bases",
        "sequence",
    }
    _require_columns(accepted, required, f"{policy} accepted rows")
    sequence = pl.col("sequence")
    sequence_length = sequence.str.len_chars()
    lowered = sequence.str.to_lowercase()
    canonical = lowered.str.count_matches("[acgt]")
    gc = lowered.str.count_matches("[gc]")
    lowercase = sequence.str.count_matches("[a-z]")
    prepared = accepted.filter(
        pl.col("alignment_source") != "human_reference"
    ).with_columns(
        pl.lit(policy).alias("projection_policy"),
        (pl.col("pre_resize_t_end") - pl.col("pre_resize_t_start")).alias(
            "pre_resize_span"
        ),
        (pl.col("aligned_bases") / float(landmark_width)).alias(
            "landmark_aligned_fraction"
        ),
        (
            (pl.col("pre_resize_t_end") - pl.col("pre_resize_t_start"))
            / float(landmark_width)
        ).alias("span_source_width_ratio"),
        sequence_length.alias("sequence_length"),
        (1.0 - canonical / sequence_length).alias("ambiguous_base_fraction"),
        (lowercase / sequence_length).alias("repeat_masked_fraction"),
        (gc / sequence_length).alias("gc_fraction"),
        (pl.col("t_start") < 0).alias("negative_target_start"),
        (pl.col("t_end") <= pl.col("t_start")).alias("nonpositive_target_interval"),
        ((pl.col("t_end") - pl.col("t_start")) != target_length).alias(
            "bad_emitted_interval_length"
        ),
        (~pl.col("t_strand").is_in(["+", "-"])).alias("bad_target_strand"),
    )
    grouping = [
        "projection_policy",
        "region_label",
        "alignment_source",
        "species",
        "clade",
    ]
    summary = (
        prepared.group_by(grouping)
        .agg(
            pl.len().cast(pl.Int64).alias("accepted_rows"),
            pl.col("fragment_count").mean().alias("mean_fragment_count"),
            pl.col("fragment_count").median().alias("median_fragment_count"),
            pl.col("fragment_count").quantile(0.9).alias("q90_fragment_count"),
            pl.col("aligned_bases").mean().alias("mean_aligned_bases"),
            pl.col("aligned_bases").median().alias("median_aligned_bases"),
            pl.col("aligned_bases").quantile(0.1).alias("q10_aligned_bases"),
            pl.col("aligned_bases").quantile(0.9).alias("q90_aligned_bases"),
            pl.col("landmark_aligned_fraction")
            .mean()
            .alias("mean_landmark_aligned_fraction"),
            pl.col("pre_resize_span").mean().alias("mean_pre_resize_span"),
            pl.col("pre_resize_span").median().alias("median_pre_resize_span"),
            pl.col("pre_resize_span").quantile(0.1).alias("q10_pre_resize_span"),
            pl.col("pre_resize_span").quantile(0.9).alias("q90_pre_resize_span"),
            pl.col("span_source_width_ratio")
            .mean()
            .alias("mean_span_source_width_ratio"),
            pl.col("span_source_width_ratio")
            .median()
            .alias("median_span_source_width_ratio"),
            pl.col("sequence_length").min().alias("min_sequence_length"),
            pl.col("sequence_length").max().alias("max_sequence_length"),
            pl.col("ambiguous_base_fraction")
            .mean()
            .alias("mean_ambiguous_base_fraction"),
            pl.col("repeat_masked_fraction")
            .mean()
            .alias("mean_repeat_masked_fraction"),
            pl.col("gc_fraction").mean().alias("mean_gc_fraction"),
            (pl.col("t_strand") == "+").sum().cast(pl.Int64).alias("plus_strand_rows"),
            (pl.col("t_strand") == "-").sum().cast(pl.Int64).alias("minus_strand_rows"),
            pl.col("negative_target_start")
            .sum()
            .cast(pl.Int64)
            .alias("negative_target_start_rows"),
            pl.col("nonpositive_target_interval")
            .sum()
            .cast(pl.Int64)
            .alias("nonpositive_target_interval_rows"),
            pl.col("bad_emitted_interval_length")
            .sum()
            .cast(pl.Int64)
            .alias("bad_emitted_interval_length_rows"),
            pl.col("bad_target_strand")
            .sum()
            .cast(pl.Int64)
            .alias("bad_target_strand_rows"),
        )
        .with_columns(
            pl.lit(
                "source_landmark_only; emitted-window coverage is sampled separately"
            ).alias("aligned_fraction_scope")
        )
        .sort(grouping)
        .collect(engine="streaming")
    )
    invalid = summary.select(
        pl.col("negative_target_start_rows").sum(),
        pl.col("nonpositive_target_interval_rows").sum(),
        pl.col("bad_emitted_interval_length_rows").sum(),
        pl.col("bad_target_strand_rows").sum(),
        (pl.col("min_sequence_length") != target_length).sum(),
        (pl.col("max_sequence_length") != target_length).sum(),
    ).row(0)
    assert invalid == (0, 0, 0, 0, 0, 0), f"{policy} invalid accepted rows: {invalid}"
    return summary


def _selected_species(species_manifest: pl.DataFrame) -> pl.DataFrame:
    required = {"scientific_name", "backend", "clade", "selected"}
    missing = required - set(species_manifest.columns)
    assert not missing, f"species manifest missing columns: {sorted(missing)}"
    selected = species_manifest.filter(pl.col("selected")).select(
        pl.col("scientific_name").alias("species"),
        pl.col("backend").alias("alignment_source"),
        "clade",
    )
    assert selected.height > 0 and selected["species"].n_unique() == selected.height
    return selected


def build_outcome_counts(
    anchors: pl.DataFrame,
    species_manifest: pl.DataFrame,
    accepted_by_policy: Mapping[str, pl.LazyFrame],
    rejected_by_policy: Mapping[str, pl.LazyFrame],
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Build exact scoped accounting without expanding the requested grid."""
    assert set(accepted_by_policy) == set(POLICY_WIDTHS)
    assert set(rejected_by_policy) == set(POLICY_WIDTHS)
    required_anchors = {"query_name", "region_label"}
    missing = required_anchors - set(anchors.columns)
    assert not missing, f"anchors missing columns: {sorted(missing)}"
    assert anchors["query_name"].n_unique() == anchors.height
    anchor_identity = anchors.select("query_name", "region_label")
    anchor_counts = anchor_identity.group_by("region_label").len(name="requested_rows")
    species = _selected_species(species_manifest)
    policies = pl.DataFrame({"projection_policy": sorted(POLICY_WIDTHS)})
    requested = policies.join(anchor_counts, how="cross").join(species, how="cross")

    outcome_parts: list[pl.DataFrame] = []
    for policy in sorted(POLICY_WIDTHS):
        accepted = accepted_by_policy[policy]
        _require_columns(
            accepted,
            {"query_name", "species", "region_label", "alignment_source", "clade"},
            f"{policy} accepted rows",
        )
        accepted_counts = (
            accepted.filter(pl.col("alignment_source") != "human_reference")
            .group_by("region_label", "species")
            .agg(pl.len().cast(pl.Int64).alias("count"))
            .with_columns(
                pl.lit(policy).alias("projection_policy"),
                pl.lit("accepted").alias("outcome"),
            )
            .select("projection_policy", "region_label", "species", "outcome", "count")
            .collect(engine="streaming")
        )
        rejected = rejected_by_policy[policy]
        _require_columns(
            rejected,
            {"query_name", "species", "rejection_reason"},
            f"{policy} rejected rows",
        )
        rejection_counts = (
            rejected.select("query_name", "species", "rejection_reason")
            .join(anchor_identity.lazy(), on="query_name", how="left", validate="m:1")
            .with_columns(
                (pl.lit("rejected:") + pl.col("rejection_reason")).alias("outcome")
            )
            .group_by("region_label", "species", "outcome")
            .agg(pl.len().cast(pl.Int64).alias("count"))
            .with_columns(pl.lit(policy).alias("projection_policy"))
            .select("projection_policy", "region_label", "species", "outcome", "count")
            .collect(engine="streaming")
        )
        explicit = rejection_counts.group_by(
            "projection_policy", "region_label", "species"
        ).agg(pl.col("count").sum().alias("explicit_rejections"))
        accepted_total = accepted_counts.rename({"count": "accepted_rows"}).select(
            "projection_policy", "region_label", "species", "accepted_rows"
        )
        policy_requested = requested.filter(pl.col("projection_policy") == policy)
        no_mapping = (
            policy_requested.join(
                accepted_total,
                on=["projection_policy", "region_label", "species"],
                how="left",
            )
            .join(
                explicit,
                on=["projection_policy", "region_label", "species"],
                how="left",
            )
            .with_columns(
                pl.col("accepted_rows").fill_null(0),
                pl.col("explicit_rejections").fill_null(0),
            )
            .with_columns(
                (
                    pl.col("requested_rows")
                    - pl.col("accepted_rows")
                    - pl.col("explicit_rejections")
                )
                .cast(pl.Int64)
                .alias("count"),
                pl.lit("no_mapping").alias("outcome"),
            )
        )
        assert no_mapping.filter(pl.col("count") < 0).is_empty()
        outcome_parts.extend(
            [
                accepted_counts,
                rejection_counts,
                no_mapping.select(
                    "projection_policy", "region_label", "species", "outcome", "count"
                ),
            ]
        )

    outcomes = (
        pl.concat(outcome_parts, how="vertical")
        .join(species, on="species", how="left", validate="m:1")
        .sort(
            "projection_policy",
            "region_label",
            "alignment_source",
            "clade",
            "species",
            "outcome",
        )
    )
    accounting = outcomes.group_by("projection_policy", "region_label", "species").agg(
        pl.col("count").sum().alias("observed_rows")
    )
    checked = requested.join(
        accounting,
        on=["projection_policy", "region_label", "species"],
        how="left",
        validate="1:1",
    )
    assert checked["observed_rows"].null_count() == 0
    assert (checked["observed_rows"] == checked["requested_rows"]).all()

    summary = (
        outcomes.with_columns(
            (pl.col("outcome") == "accepted").alias("is_accepted"),
            (pl.col("outcome") == "no_mapping").alias("is_no_mapping"),
        )
        .group_by("projection_policy", "region_label")
        .agg(
            pl.col("count").sum().alias("requested_pairs"),
            pl.col("count").filter(pl.col("is_accepted")).sum().alias("accepted_pairs"),
            pl.col("count")
            .filter(pl.col("is_no_mapping"))
            .sum()
            .alias("no_mapping_pairs"),
            pl.col("count")
            .filter(~pl.col("is_accepted") & ~pl.col("is_no_mapping"))
            .sum()
            .alias("rejected_pairs"),
        )
        .join(anchor_counts, on="region_label", how="left", validate="m:1")
        .with_columns(
            (pl.col("accepted_pairs") / pl.col("requested_pairs")).alias(
                "recovery_fraction"
            ),
            (pl.col("accepted_pairs") / pl.col("requested_rows")).alias(
                "mean_accepted_species_per_anchor"
            ),
        )
        .sort("region_label", "projection_policy")
    )
    assert (
        summary["accepted_pairs"]
        + summary["rejected_pairs"]
        + summary["no_mapping_pairs"]
        == summary["requested_pairs"]
    ).all()
    return outcomes, summary


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(4 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _write_report(summary: pl.DataFrame, output: Path) -> None:
    lines = [
        "# Issue #473 post-projection summary",
        "",
        "All counts are over the exact fixed anchor-by-target-species grid.",
        "Coordinates are 0-based and half-open.",
        "",
        "| Region | Policy | Requested | Accepted | Rejected | No mapping | Recovery | Mean species/anchor |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary.iter_rows(named=True):
        lines.append(
            "| {region_label} | {projection_policy} | {requested_pairs:,} | "
            "{accepted_pairs:,} | {rejected_pairs:,} | {no_mapping_pairs:,} | "
            "{recovery_fraction:.4f} | {mean_accepted_species_per_anchor:.2f} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            (
                "`accepted_metrics.parquet` reports fragment, source-landmark "
                "alignment, pre-resize span, target interval, strand, ambiguity, "
                "repeat masking, and GC summaries by region/backend/species/clade."
            ),
            "",
            (
                "Source-landmark aligned fraction is not emitted-window coverage. "
                "Exact emitted-window coverage is reported only by the sampled HAL "
                "trace."
            ),
        ]
    )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_report(
    *,
    anchors_path: Path,
    species_manifest_path: Path,
    accepted_paths: Mapping[str, Path],
    rejection_paths: Mapping[str, Sequence[Path]],
    output_dir: Path,
    producer_commit: str,
    producer_config_sha256: str,
) -> None:
    """Run the complete bounded-memory post-projection analysis."""
    assert len(producer_commit) == 40 and len(producer_config_sha256) == 64
    anchors = pl.read_parquet(anchors_path)
    species_manifest = pl.read_csv(species_manifest_path, separator="\t")
    accepted = {
        policy: pl.scan_parquet(path) for policy, path in accepted_paths.items()
    }
    rejected = {
        policy: pl.concat(
            [
                pl.scan_parquet(path).select(
                    "query_name", "species", "rejection_reason"
                )
                for path in paths
            ],
            how="vertical",
        )
        for policy, paths in rejection_paths.items()
    }
    assert all(rejection_paths.values()), "each policy needs rejection evidence"
    accepted_metrics = pl.concat(
        [
            summarize_accepted_rows(
                accepted[policy],
                policy=policy,
                landmark_width=POLICY_WIDTHS[policy],
            )
            for policy in sorted(POLICY_WIDTHS)
        ],
        how="vertical",
    )
    outcomes, summary = build_outcome_counts(
        anchors,
        species_manifest,
        accepted,
        rejected,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "accepted_metrics": output_dir / "accepted_metrics.parquet",
        "outcome_counts": output_dir / "outcome_counts.parquet",
        "region_policy_summary": output_dir / "region_policy_summary.parquet",
        "report": output_dir / "report.md",
    }
    accepted_metrics.write_parquet(outputs["accepted_metrics"])
    outcomes.write_parquet(outputs["outcome_counts"])
    summary.write_parquet(outputs["region_policy_summary"])
    _write_report(summary, outputs["report"])
    manifest = {
        "producer_commit": producer_commit,
        "producer_config_sha256": producer_config_sha256,
        "coordinate_system": "0-based half-open",
        "policies": POLICY_WIDTHS,
        "accepted_paths": {key: str(value) for key, value in accepted_paths.items()},
        "rejection_path_counts": {
            key: len(value) for key, value in rejection_paths.items()
        },
        "outputs": {
            key: {"path": path.name, "sha256": _sha256(path)}
            for key, path in outputs.items()
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchors", type=Path, required=True)
    parser.add_argument("--species-manifest", type=Path, required=True)
    parser.add_argument("--full-accepted", type=Path, required=True)
    parser.add_argument("--center-accepted", type=Path, required=True)
    parser.add_argument("--full-rejection", type=Path, action="append", required=True)
    parser.add_argument("--center-rejection", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--producer-commit", required=True)
    parser.add_argument("--producer-config-sha256", required=True)
    args = parser.parse_args()
    run_report(
        anchors_path=args.anchors,
        species_manifest_path=args.species_manifest,
        accepted_paths={
            "full_window": args.full_accepted,
            "center_1": args.center_accepted,
        },
        rejection_paths={
            "full_window": args.full_rejection,
            "center_1": args.center_rejection,
        },
        output_dir=args.output_dir,
        producer_commit=args.producer_commit,
        producer_config_sha256=args.producer_config_sha256,
    )


if __name__ == "__main__":
    main()
