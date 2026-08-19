"""Cross-policy projection summaries for issue #473."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from marin_dna_vertebrate_projection.issue_473.policy import policy_by_name


def _scalar_stats(
    path: str | Path, policy_name: str, target_length: int
) -> dict[str, object]:
    policy = policy_by_name(policy_name)
    target_span = pl.col("pre_resize_t_end") - pl.col("pre_resize_t_start")
    stats = (
        pl.scan_parquet(path)
        .filter(pl.col("alignment_source") != "human_reference")
        .select(
            pl.len().cast(pl.Int64).alias("accepted_pairs"),
            (pl.col("aligned_bases") / policy.landmark_width)
            .mean()
            .alias("mean_mapped_landmark_fraction"),
            target_span.mean().alias("mean_pre_resize_target_span"),
            pl.max_horizontal(pl.lit(0), target_length - target_span)
            .mean()
            .alias("mean_emitted_flank_bases"),
        )
        .collect(engine="streaming")
        .row(0, named=True)
    )
    mapped_fraction = stats["mean_mapped_landmark_fraction"]
    assert mapped_fraction is None or 0.0 <= float(mapped_fraction) <= 1.0
    return stats


def _qc_totals(path: str | Path) -> dict[str, int]:
    stats = (
        pl.scan_parquet(path)
        .select(
            pl.col("accepted_total_projections")
            .sum()
            .cast(pl.Int64)
            .alias("accepted_pairs"),
            pl.col("requested_total_species")
            .sum()
            .cast(pl.Int64)
            .alias("requested_pairs"),
            pl.col("no_mapping_count").sum().cast(pl.Int64).alias("no_mapping_pairs"),
        )
        .collect(engine="streaming")
        .row(0, named=True)
    )
    accepted = int(stats["accepted_pairs"])
    requested = int(stats["requested_pairs"])
    no_mapping = int(stats["no_mapping_pairs"])
    explicit_rejections = requested - accepted - no_mapping
    assert accepted >= 0 and no_mapping >= 0 and explicit_rejections >= 0
    return {
        "accepted_pairs": accepted,
        "requested_pairs": requested,
        "no_mapping_pairs": no_mapping,
        "explicit_rejection_pairs": explicit_rejections,
    }


def _accepted_keys(path: str | Path, flag_name: str) -> pl.LazyFrame:
    return (
        pl.scan_parquet(path)
        .filter(pl.col("alignment_source") != "human_reference")
        .select("query_name", "species")
        .unique()
        .with_columns(pl.lit(True).alias(flag_name))
    )


def write_policy_comparison(
    sequence_paths: dict[str, str],
    per_anchor_paths: dict[str, str],
    summary_path: str | Path,
    pairwise_path: str | Path,
    *,
    baseline_policy: str = "full_window",
    target_length: int = 255,
) -> None:
    """Compare recovery and mapping evidence against the full-window baseline."""
    assert sequence_paths and set(sequence_paths) == set(per_anchor_paths)
    assert baseline_policy in sequence_paths
    assert target_length > 0

    summary_rows: list[dict[str, object]] = []
    for policy_name, sequence_path in sequence_paths.items():
        qc = _qc_totals(per_anchor_paths[policy_name])
        mapping = _scalar_stats(sequence_path, policy_name, target_length)
        assert int(mapping["accepted_pairs"]) == qc["accepted_pairs"]
        summary_rows.append(
            {
                "projection_policy": policy_name,
                **qc,
                "recovered_fraction": (
                    qc["accepted_pairs"] / qc["requested_pairs"]
                    if qc["requested_pairs"]
                    else 0.0
                ),
                "mean_mapped_landmark_fraction": mapping[
                    "mean_mapped_landmark_fraction"
                ],
                "mean_pre_resize_target_span": mapping["mean_pre_resize_target_span"],
                "mean_emitted_flank_bases": mapping["mean_emitted_flank_bases"],
            }
        )

    baseline_keys = _accepted_keys(
        sequence_paths[baseline_policy], "_baseline_accepted"
    )
    pairwise_rows: list[dict[str, object]] = []
    for policy_name, sequence_path in sequence_paths.items():
        if policy_name == baseline_policy:
            continue
        policy_keys = _accepted_keys(sequence_path, "_policy_accepted")
        counts = (
            baseline_keys.join(
                policy_keys,
                on=["query_name", "species"],
                how="full",
                coalesce=True,
            )
            .with_columns(
                pl.col("_baseline_accepted").fill_null(False),
                pl.col("_policy_accepted").fill_null(False),
            )
            .select(
                (pl.col("_baseline_accepted") & pl.col("_policy_accepted"))
                .sum()
                .cast(pl.Int64)
                .alias("accepted_by_both"),
                (pl.col("_baseline_accepted") & ~pl.col("_policy_accepted"))
                .sum()
                .cast(pl.Int64)
                .alias("baseline_only"),
                (~pl.col("_baseline_accepted") & pl.col("_policy_accepted"))
                .sum()
                .cast(pl.Int64)
                .alias("policy_only"),
                pl.len().cast(pl.Int64).alias("accepted_union"),
            )
            .collect(engine="streaming")
            .row(0, named=True)
        )
        union = int(counts["accepted_union"])
        both = int(counts["accepted_by_both"])
        pairwise_rows.append(
            {
                "baseline_policy": baseline_policy,
                "projection_policy": policy_name,
                **counts,
                "accepted_jaccard": both / union if union else 1.0,
            }
        )

    summary = pl.DataFrame(summary_rows).sort("projection_policy")
    pairwise = pl.DataFrame(pairwise_rows).sort("projection_policy")
    summary_output = Path(summary_path)
    pairwise_output = Path(pairwise_path)
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    pairwise_output.parent.mkdir(parents=True, exist_ok=True)
    summary.write_parquet(summary_output)
    pairwise.write_parquet(pairwise_output)
